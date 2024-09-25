#include "PPOLearner.h"

#include "../Util/TorchFuncs.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>

using namespace torch;

Tensor _CopyParams(nn::Module* mod) {
    return torch::nn::utils::parameters_to_vector(mod->parameters()).cpu();
}

void _CopyModelParamsHalf(nn::Module* from, nn::Module* to) {
    RG_NOGRAD;
    try {
        auto fromParams = from->parameters();
        auto toParams = to->parameters();

        if (fromParams.size() != toParams.size()) {
            RG_ERR_CLOSE("_CopyModelParamsHalf(): from and to modules have different number of parameters");
        }

        for (size_t i = 0; i < fromParams.size(); i++) {
            auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE);
            toParams[i].copy_(scaledParams, true);
        }
    }
    catch (const std::exception& e) {
        RG_ERR_CLOSE("_CopyModelParamsHalf() exception: " << e.what());
    }
}

RLGPC::PPOLearner::PPOLearner(int obsSpaceSize, int actSpaceSize, PPOLearnerConfig _config, Device _device)
    : config(_config), device(_device), minibatchThreadPool(nullptr) {

    if (config.miniBatchSize == 0)
        config.miniBatchSize = config.batchSize;

    if (config.batchSize % config.miniBatchSize != 0)
        RG_ERR_CLOSE("PPOLearner: config.batchSize must be a multiple of config.miniBatchSize");

    policy = new DiscretePolicy(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device, config.policyTemperature);
    valueNet = new ValueEstimator(obsSpaceSize, config.criticLayerSizes, device);

    if (config.halfPrecModels) {
        policyHalf = new DiscretePolicy(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device);
        valueNetHalf = new ValueEstimator(obsSpaceSize, config.criticLayerSizes, device);

        _CopyModelParamsHalf(policy, policyHalf);
        _CopyModelParamsHalf(valueNet, valueNetHalf);

        policyHalf->to(RG_HALFPERC_TYPE);
        valueNetHalf->to(RG_HALFPERC_TYPE);
    }
    else {
        policyHalf = nullptr;
        valueNetHalf = nullptr;
    }

    policyOptimizer = new optim::Adam(policy->parameters(), optim::AdamOptions(config.policyLR));
    valueOptimizer = new optim::Adam(valueNet->parameters(), optim::AdamOptions(config.criticLR));
    valueLossFn = nn::MSELoss();

    if (config.measureGradientNoise) {
        noiseTrackerPolicy = new GradNoiseTracker(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
        noiseTrackerValueNet = new GradNoiseTracker(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
    }
    else {
        noiseTrackerPolicy = nullptr;
        noiseTrackerValueNet = nullptr;
    }
}

RLGPC::PPOLearner::~PPOLearner() {
    delete policy;
    delete valueNet;
    delete policyOptimizer;
    delete valueOptimizer;
    if (policyHalf)
        delete policyHalf;
    if (valueNetHalf)
        delete valueNetHalf;
    if (noiseTrackerPolicy)
        delete noiseTrackerPolicy;
    if (noiseTrackerValueNet)
        delete noiseTrackerValueNet;
    if (minibatchThreadPool)
        delete minibatchThreadPool;
}

void RLGPC::PPOLearner::Learn(ExperienceBuffer* expBuffer, Report& report) {

    bool autocast = config.autocastLearn;

    if (autocast) {
#ifndef RG_CUDA_SUPPORT
        RG_ERR_CLOSE("Autocast not supported on non-CUDA!")
#endif
    }

    static amp::GradScaler* gradScaler = nullptr;
#ifdef RG_CUDA_SUPPORT
    if (autocast && !gradScaler) {
        RG_LOG("Creating grad scaler...");
        gradScaler = new amp::GradScaler();
    }
#endif

    int numIterations = 0;
    int numMinibatchIterations = 0;
    float meanEntropy = 0.0f;
    float meanDivergence = 0.0f;
    float meanValLoss = 0.0f;
    float meanRatio = 0.0f;
    FList clipFractions;

    // Save parameters first
    auto policyBefore = _CopyParams(policy);
    auto criticBefore = _CopyParams(valueNet);

    bool trainPolicy = config.policyLR != 0;
    bool trainCritic = config.criticLR != 0;

    Timer totalTimer;
    for (int epoch = 0; epoch < config.epochs; epoch++) {

        // Get randomly-ordered timesteps for PPO
        auto batches = expBuffer->GetAllBatchesShuffled(config.batchSize);

        for (auto& batch : batches) {
            auto batchActs = batch.actions;
            auto batchOldProbs = batch.logProbs;
            auto batchObs = batch.states;
            auto batchTargetValues = batch.values;
            auto batchAdvantages = batch.advantages;

            batchActs = batchActs.view({ config.batchSize, -1 });
            policyOptimizer->zero_grad();
            valueOptimizer->zero_grad();

            // Synchronization primitives
            std::mutex threadLockMutex;
            std::mutex threadUpdateMutex;
            std::condition_variable threadCV;
            std::atomic<int> threadCounter = 0;

            auto fnRunMinibatch = [&](int start, int stop) {

                float batchSizeRatio = (stop - start) / static_cast<float>(config.batchSize);

                // Send everything to the device and enforce correct shapes
                auto acts = batchActs.slice(0, start, stop).to(device, true, true);
                auto obs = batchObs.slice(0, start, stop).to(device, true, true);

                auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);
                auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
                auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

                Timer timer;
                if (autocast) RG_AUTOCAST_ON();
                auto vals = valueNet->Forward(obs); // 11%
                {
                    std::lock_guard<std::mutex> lock(threadUpdateMutex);
                    report.Accum("PPO Value Estimate Time", timer.Elapsed());
                }

                timer.Reset();
                torch::Tensor logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
                if (trainPolicy) {
                    // Get policy log probs & entropy
                    DiscretePolicy::BackpropResult bpResult = policy->GetBackpropData(obs, acts); // 13%

                    logProbs = bpResult.actionLogProbs;
                    entropy = bpResult.entropy;

                    logProbs = logProbs.view_as(oldProbs);
                    {
                        std::lock_guard<std::mutex> lock(threadUpdateMutex);
                        report.Accum("PPO Backprop Data Time", timer.Elapsed());
                    }

                    // Compute PPO loss
                    ratio = exp(logProbs - oldProbs);
                    {
                        std::lock_guard<std::mutex> lock(threadUpdateMutex);
                        meanRatio += ratio.mean().detach().cpu().item<float>();
                    }
                    clipped = clamp(
                        ratio, 1 - config.clipRange, 1 + config.clipRange
                    );

                    // Compute policy loss
                    policyLoss = -min(
                        ratio * advantages, clipped * advantages
                    ).mean();
                    ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio;
                }

                torch::Tensor valueLoss;
                if (trainCritic) {
                    // Compute value loss
                    vals = vals.view_as(targetValues);
                    valueLoss = valueLossFn(vals, targetValues) * batchSizeRatio;
                }

                if (autocast) RG_AUTOCAST_OFF();

                float kl = 0.0f;
                if (trainPolicy) {
                    // Compute KL divergence & clip fraction using SB3 method for reporting
                    float clipFraction;
                    {
                        RG_NOGRAD;

                        auto logRatio = logProbs - oldProbs;
                        auto klTensor = (exp(logRatio) - 1) - logRatio;
                        kl = klTensor.mean().detach().cpu().item<float>();

                        clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).cpu().item<float>();
                        {
                            std::lock_guard<std::mutex> lock(threadUpdateMutex);
                            clipFractions.push_back(clipFraction);
                        }
                    }
                }

                // Gradient computations
                if (autocast) {
                    if (trainPolicy)
                        gradScaler->scale(ppoLoss).backward();
                    if (trainCritic)
                        gradScaler->scale(valueLoss).backward();
                }
                else {
                    if (trainPolicy)
                        ppoLoss.backward(); // 29%
                    if (trainCritic)
                        valueLoss.backward(); // 24%
                }

                {
                    std::lock_guard<std::mutex> lock(threadUpdateMutex);

                    report.Accum("PPO Gradient Time", timer.Elapsed());

                    if (trainCritic)
                        meanValLoss += valueLoss.cpu().detach().item<float>();
                    if (trainPolicy) {
                        meanDivergence += kl;
                        meanEntropy += entropy.cpu().detach().item<float>();
                    }
                    numMinibatchIterations += 1;
                }

                {
                    std::lock_guard<std::mutex> lock(threadLockMutex);
                    threadCounter--;
                    threadCV.notify_all();
                }
                };

            if (this->device.is_cpu()) {

                if (!this->minibatchThreadPool) {
                    int numThreads = std::thread::hardware_concurrency();
                    numThreads += numThreads / 2; // Slightly faster
                    this->minibatchThreadPool = new ThreadPool(numThreads);
                }

                // Use multithreaded PPO learn
                int realMinibatchSize = config.batchSize / this->minibatchThreadPool->threads.size();

                for (int mbs = 0; mbs < config.batchSize; mbs += realMinibatchSize) {
                    int start = mbs;
                    int stop = start + realMinibatchSize;
                    stop = RS_MIN(stop, config.batchSize);

                    this->minibatchThreadPool->StartJob(std::bind(fnRunMinibatch, start, stop));
                }

                while (this->minibatchThreadPool->GetNumRunningJobs() > 0)
                    RG_SLEEP(1);

            }
            else {
                for (int mbs = 0; mbs < config.batchSize; mbs += config.miniBatchSize) {
                    int start = mbs;
                    int stop = std::min(start + config.miniBatchSize, config.batchSize);
                    fnRunMinibatch(start, stop);
                }
            }

            if (config.measureGradientNoise) {
                if (trainPolicy)
                    noiseTrackerPolicy->Update(policy->seq);
                if (trainCritic)
                    noiseTrackerValueNet->Update(valueNet->seq);
            }

            if (trainPolicy)
                nn::utils::clip_grad_norm_(policy->parameters(), 0.5f);
            if (trainCritic)
                nn::utils::clip_grad_norm_(valueNet->parameters(), 0.5f);

            if (autocast) {
                if (trainPolicy)
                    gradScaler->step(*policyOptimizer);
                if (trainCritic)
                    gradScaler->step(*valueOptimizer);
            }
            else {
                if (trainPolicy)
                    policyOptimizer->step();
                if (trainCritic)
                    valueOptimizer->step();
            }

            if (policyHalf)
                _CopyModelParamsHalf(policy, policyHalf);
            if (valueNetHalf)
                _CopyModelParamsHalf(valueNet, valueNetHalf);

            if (autocast)
                gradScaler->update();
            numIterations += 1;
        }
    }

    numIterations = RS_MAX(numIterations, 1);
    numMinibatchIterations = RS_MAX(numMinibatchIterations, 1);

    // Compute averages for the metrics that will be reported
    meanEntropy /= numMinibatchIterations;
    meanDivergence /= numMinibatchIterations;
    meanValLoss /= numMinibatchIterations;
    meanRatio /= numMinibatchIterations;

    float meanClip = 0.0f;
    if (!clipFractions.empty()) {
        for (float f : clipFractions)
            meanClip += f;
        meanClip /= clipFractions.size();
    }

    // Compute magnitude of updates made to the policy and value estimator
    auto policyAfter = _CopyParams(policy);
    auto criticAfter = _CopyParams(valueNet);

    float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
    float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

    float totalTime = totalTimer.Elapsed();

    // Assemble and return report
    cumulativeModelUpdates += numIterations;
    report["PPO Batch Consumption Time"] = totalTime / numIterations;
    report["Cumulative Model Updates"] = cumulativeModelUpdates;
    report["Policy Entropy"] = meanEntropy;
    report["Mean KL Divergence"] = meanDivergence;
    report["Mean Ratio"] = meanRatio;
    report["Value Function Loss"] = meanValLoss;
    report["SB3 Clip Fraction"] = meanClip;
    report["Policy Update Magnitude"] = policyUpdateMagnitude;
    report["Value Function Update Magnitude"] = criticUpdateMagnitude;
    report["PPO Learn Time"] = totalTimer.Elapsed();

    if (config.measureGradientNoise) {
        if (noiseTrackerPolicy->lastNoiseScale != 0)
            report["Grad Noise Policy"] = noiseTrackerPolicy->lastNoiseScale;
        if (noiseTrackerValueNet->lastNoiseScale != 0)
            report["Grad Noise Value Net"] = noiseTrackerValueNet->lastNoiseScale;
    }

    policyOptimizer->zero_grad();
    valueOptimizer->zero_grad();
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
    std::vector<uint64_t> result;

    for (size_t i = 0; i < seq->size(); i++)
        for (const auto& param : seq[i]->parameters())
            result.push_back(param.numel());

    return result;
}

constexpr const char* MODEL_FILE_NAMES[] = {
    "PPO_POLICY.lt",
    "PPO_CRITIC.lt",
};

constexpr const char* OPTIM_FILE_NAMES[] = {
    "PPO_POLICY_OPTIM.lt",
    "PPO_CRITIC_OPTIM.lt",
};

void TorchLoadSaveSeq(torch::nn::Sequential seq, std::filesystem::path path, c10::Device device, bool load) {
    if (load) {
        auto streamIn = std::ifstream(path, std::ios::binary);
        streamIn >> std::noskipws;

        if (!streamIn.good())
            RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

        auto sizesBefore = GetSeqSizes(seq);

        try {
            torch::load(seq, streamIn, device);
        }
        catch (const std::exception& e) {
            RG_ERR_CLOSE(
                "Failed to load model, checkpoint may be corrupt or of different model arch.\n"
                << "Exception: " << e.what()
            );
        }

        // Verify model size
        auto sizesAfter = GetSeqSizes(seq);
        if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
            std::stringstream stream;
            stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";

            for (int i = 0; i < 2; i++) {
                stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
                for (uint64_t size : (i ? sizesAfter : sizesBefore))
                    stream << size << ' ';

                stream << " ]";
                if (i == 0)
                    stream << ",\n";
            }

            RG_ERR_CLOSE(stream.str());
        }

    }
    else {
        auto streamOut = std::ofstream(path, std::ios::binary);
        if (!streamOut) {
            RG_ERR_CLOSE("Failed to open file for writing: " << path);
        }
        torch::save(seq, streamOut);
    }
}

void TorchLoadSaveAll(RLGPC::PPOLearner* learner, std::filesystem::path folderPath, bool load) {

    if (load) {
        if (!std::filesystem::exists(folderPath / MODEL_FILE_NAMES[0]))
            RG_ERR_CLOSE("PPOLearner: Failed to find file \"" << MODEL_FILE_NAMES[0] << "\" in " << folderPath << ".")
    }

    TorchLoadSaveSeq(learner->policy->seq, folderPath / MODEL_FILE_NAMES[0], learner->device, load);

    if (!load || std::filesystem::exists(folderPath / MODEL_FILE_NAMES[1]))
        TorchLoadSaveSeq(learner->valueNet->seq, folderPath / MODEL_FILE_NAMES[1], learner->device, load);

    if (load) {
        if (learner->policyHalf)
            _CopyModelParamsHalf(learner->policy, learner->policyHalf);
        if (learner->valueNetHalf)
            _CopyModelParamsHalf(learner->valueNet, learner->valueNetHalf);
    }

    // Load or save optimizers
    if (load) {
        try {
            for (int i = 0; i < 2; i++) {
                auto path = folderPath / OPTIM_FILE_NAMES[i];

                if (!std::filesystem::exists(path)) {
                    RG_LOG("WARNING: No optimizer found at " << path << ", optimizer will be reset");
                    continue;
                }

                { // Check if empty
                    std::ifstream testStream(path, std::istream::ate | std::ios::binary);
                    if (testStream.tellg() == 0) {
                        RG_LOG("WARNING: Saved optimizer is empty, optimizer will be reset");
                        continue;
                    }
                }

                auto& optim = i ? learner->valueOptimizer : learner->policyOptimizer;

                torch::serialize::InputArchive optArchive;
                optArchive.load_from(path.string(), learner->device);
                optim->load(optArchive);
            }

        }
        catch (const std::exception& e) {
            RG_ERR_CLOSE(
                "Failed to load optimizers, exception: " << e.what() << "\n"
                << "Checkpoint may be corrupt."
            );
        }
    }
    else {
        for (int i = 0; i < 2; i++) {
            torch::serialize::OutputArchive optArchive;
            auto& optim = i ? learner->valueOptimizer : learner->policyOptimizer;
            optim->save(optArchive);
            optArchive.save_to((folderPath / OPTIM_FILE_NAMES[i]).string());
        }
    }
}

void RLGPC::PPOLearner::SaveTo(std::filesystem::path folderPath) {
    RG_LOG("PPOLearner(): Saving models to: " << folderPath);
    TorchLoadSaveAll(this, folderPath, false);
}

RLGPC::DiscretePolicy* RLGPC::PPOLearner::LoadAdditionalPolicy(std::filesystem::path folderPath) {
    std::filesystem::path policyPath = folderPath / MODEL_FILE_NAMES[0];
    if (!std::filesystem::exists(policyPath))
        return nullptr;

    RLGPC::DiscretePolicy* newPolicy = new RLGPC::DiscretePolicy(policy->inputAmount, policy->actionAmount, policy->layerSizes, policy->device);
    TorchLoadSaveSeq(newPolicy->seq, policyPath, newPolicy->device, true);
    return newPolicy;
}

void RLGPC::PPOLearner::LoadFrom(std::filesystem::path folderPath) {
    RG_LOG("PPOLearner(): Loading models from: " << folderPath);
    if (!std::filesystem::is_directory(folderPath))
        RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

    TorchLoadSaveAll(this, folderPath, true);

    UpdateLearningRates(config.policyLR, config.criticLR);
}

void RLGPC::PPOLearner::UpdateLearningRates(float policyLR, float criticLR) {
    config.policyLR = policyLR;
    config.criticLR = criticLR;

    for (auto& g : policyOptimizer->param_groups())
        static_cast<torch::optim::AdamOptions&>(g.options()).lr(policyLR);

    for (auto& g : valueOptimizer->param_groups())
        static_cast<torch::optim::AdamOptions&>(g.options()).lr(criticLR);

    RG_LOG("PPOLearner: Updated learning rate to [" << std::scientific << policyLR << ", " << criticLR << "]");
}
