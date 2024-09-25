#include "Learner.h"

#include "../../private/RLGymPPO_CPP/Util/SkillTracker.h"

#include <RLGymPPO_CPP/PPO/PPOLearner.h>
#include <RLGymPPO_CPP/PPO/ExperienceBuffer.h>
#include <RLGymPPO_CPP/Threading/ThreadAgentManager.h>

#include <torch/torch.h>
#include <torch/cuda.h>
#include "../libsrc/json/nlohmann/json.hpp"
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <fstream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>
#include <ctime>

using namespace nlohmann;

namespace RLGPC {

    template <typename T>
    json MakeJSONArray(const std::vector<T>& list) {
        json result = json::array();
        for (const T& v : list) {
            if (std::isnan(v)) {
                continue;
            }
            result.push_back(v);
        }

        return result;
    }

    void DisplayReport(const Report& report) {
        constexpr const char* REPORT_DATA_ORDER[] = {
            "Average Episode Reward",
            "Average Step Reward",
            "Policy Entropy",
            "Value Function Loss",
            "",
            "Mean KL Divergence",
            "SB3 Clip Fraction",
            "Policy Update Magnitude",
            "Value Function Update Magnitude",
            "",
            "Collected Steps/Second",
            "Overall Steps/Second",
            "",
            "Collection Time",
            "-Policy Infer Time",
            "-Env Step Time",
            "Consumption Time",
            "-PPO Learn Time",
            "Collect-Consume Overlap Time",
            "Total Iteration Time",
            "",
            "Cumulative Model Updates",
            "Cumulative Timesteps",
            "",
            "Timesteps Collected"
        };

        for (const char* name : REPORT_DATA_ORDER) {
            if (strlen(name) > 0) {
                int indentLevel = 0;
                while (name[0] == '-') {
                    indentLevel++;
                    name++;
                }

                std::string prefix;
                if (indentLevel > 0) {
                    prefix += std::string((indentLevel - 1) * 3, ' ');
                    prefix += " - ";
                }

                RG_LOG(prefix + report.SingleToString(name, true));
            }
            else {
                RG_LOG("");
            }
        }
    }

    Learner::Learner(EnvCreateFn envCreateFunc, LearnerConfig _config) :
        envCreateFn(envCreateFunc),
        config(std::move(_config))
    {
        pybind11::initialize_interpreter();

#ifndef NDEBUG

#endif

        if (config.timestepsPerSave == 0)
            config.timestepsPerSave = config.timestepsPerIteration;

        if (config.standardizeOBS)
            RG_ERR_CLOSE("LearnerConfig.standardizeOBS has not yet been implemented, sorry");


        if (config.renderMode && !config.renderDuringTraining) {
            config.numThreads = config.numGamesPerThread = 1;

            config.sendMetrics = false;

            config.checkpointSaveFolder.clear();

            config.timestepsPerIteration = INT_MAX;
        }

        if (config.saveFolderAddUnixTimestamp && !config.checkpointSaveFolder.empty())
            config.checkpointSaveFolder += "-" + std::to_string(std::time(0));


        torch::manual_seed(config.randomSeed);

        at::Device device = at::Device(at::kCPU);
        if (
            config.deviceType == LearnerDeviceType::GPU_CUDA ||
            (config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
            ) {

            torch::Tensor t;
            bool deviceTestFailed = false;
            try {
                t = torch::tensor(0);
                t = t.to(at::Device(at::kCUDA));
                t = t.cpu();
            }
            catch (...) {
                deviceTestFailed = true;
            }

            if (!torch::cuda::is_available() || deviceTestFailed)
                RG_ERR_CLOSE(
                    "Learner::Learner(): Can't use CUDA GPU because " <<
                    (torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
                    "Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
                );
            device = at::Device(at::kCUDA);
        }
        else {
            device = at::Device(at::kCPU);
        }

        torch::set_num_interop_threads(1);
        torch::set_num_threads(1);

        if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
            RocketSim::Init("collision_meshes", true);
        }

        {
            auto envCreateResult = envCreateFunc();
            auto obsSet = envCreateResult.gym->Reset();
            obsSize = obsSet[0].size();
            actionAmount = envCreateResult.match->actionParser->GetActionAmount();
            delete envCreateResult.gym;
            delete envCreateResult.match;
        }

        expBuffer = new ExperienceBuffer(config.expBufferSize, config.randomSeed, device);

        ppo = new PPOLearner(obsSize, actionAmount, config.ppo, device);

        agentMgr = new ThreadAgentManager(
            ppo->policy, ppo->policyHalf, expBuffer,
            config.standardizeOBS, config.deterministic, device.is_cpu() && torch::get_num_threads() > 1,
            static_cast<uint64_t>(config.timestepsPerIteration * 1.5f),
            device
        );

        agentMgr->CreateAgents(envCreateFunc, config.numThreads, config.numGamesPerThread);

        if (config.renderMode) {
            renderSender = new RenderSender();
            agentMgr->renderSender = renderSender;
            agentMgr->renderTimeScale = config.renderTimeScale;
            agentMgr->renderDuringTraining = config.renderDuringTraining;
        }
        else {
            renderSender = nullptr;
        }

        if (config.skillTrackerConfig.enabled) {
            if (config.skillTrackerConfig.envCreateFunc == nullptr)
                config.skillTrackerConfig.envCreateFunc = envCreateFunc;

            skillTracker = new SkillTracker(config.skillTrackerConfig, renderSender);
        }
        else {
            skillTracker = nullptr;
        }

        if (!config.checkpointLoadFolder.empty())
            Load();

        if (config.sendMetrics) {
            if (!runID.empty())
                metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
        }
        else {
            metricSender = nullptr;
        }
    }

    void Learner::SaveStats(std::filesystem::path path) {
        constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

        std::ofstream fOut(path, std::ios::out | std::ios::trunc);
        if (!fOut.good())
            RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

        json j = {};
        j["cumulative_timesteps"] = totalTimesteps;
        j["cumulative_model_updates"] = ppo->cumulativeModelUpdates;
        j["epoch"] = totalEpochs;

        if (skillTracker) {
            if (skillTracker->config.perModeRatings) {
                json ratings = {};
                for (auto& pair : skillTracker->curRating.data)
                    ratings[pair.first] = pair.second;
                j["skill_rating"] = ratings;
            }
            else {
                j["skill_rating"] = skillTracker->curRating.data[""];
            }
        }

        auto& rrs = j["reward_running_stats"];
        {
            rrs["mean"] = MakeJSONArray(returnStats.runningMean);
            rrs["var"] = MakeJSONArray(returnStats.runningVariance);
            rrs["shape"] = returnStats.shape;
            rrs["count"] = returnStats.count;
        }

        if (config.sendMetrics)
            j["run_id"] = metricSender->curRunID;

        std::string jStr = j.dump(4);
        fOut << jStr;
    }

    void Learner::LoadStats(std::filesystem::path path) {
        constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

        std::ifstream fIn(path);
        if (!fIn.good())
            RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

        json j = json::parse(fIn);
        totalTimesteps = j["cumulative_timesteps"];
        ppo->cumulativeModelUpdates = j["cumulative_model_updates"];
        totalEpochs = j["epoch"];

        if (skillTracker && j.contains("skill_rating")) {
            skillTracker->curRating = skillTracker->LoadRatingSet(j["skill_rating"]);
        }

        auto& rrs = j["reward_running_stats"];
        {
            returnStats = WelfordRunningStat(rrs["shape"]);
            returnStats.runningMean = rrs["mean"].get<std::vector<double>>();
            returnStats.runningVariance = rrs["var"].get<std::vector<double>>();
            returnStats.count = rrs["count"];
        }

        if (j.contains("run_id"))
            runID = j["run_id"];
    }

    constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

    void Learner::Save() {
        if (config.checkpointSaveFolder.empty())
            RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

        std::filesystem::path saveFolder = config.checkpointSaveFolder / std::to_string(totalTimesteps);
        std::error_code ec;
        std::filesystem::create_directories(saveFolder, ec);
        if (ec)
            RG_ERR_CLOSE("Failed to create directories: " << saveFolder << ", error: " << ec.message());

        SaveStats(saveFolder / STATS_FILE_NAME);
        ppo->SaveTo(saveFolder);

        if (config.checkpointsToKeep != -1) {
            int numCheckpoints = 0;
            int64_t lowestCheckpointTS = std::numeric_limits<int64_t>::max();

            for (const auto& entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
                if (entry.is_directory()) {
                    std::string name = entry.path().filename().string();
                    try {
                        int64_t nameVal = std::stoll(name);
                        lowestCheckpointTS = std::min(nameVal, lowestCheckpointTS);
                        numCheckpoints++;
                    }
                    catch (...) {
                    }
                }
            }

            if (numCheckpoints > config.checkpointsToKeep) {
                std::filesystem::path removePath = config.checkpointLoadFolder / std::to_string(lowestCheckpointTS);
                std::filesystem::remove_all(removePath, ec);
                if (ec)
                    RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", error: " << ec.message());
            }
        }

    }

    void Learner::Load() {
        if (config.checkpointLoadFolder.empty())
            RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

        int64_t highest = -1;
        if (std::filesystem::is_directory(config.checkpointLoadFolder)) {
            for (const auto& entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
                if (entry.is_directory()) {
                    std::string name = entry.path().filename().string();
                    try {
                        int64_t nameVal = std::stoll(name);
                        highest = std::max(nameVal, highest);
                    }
                    catch (...) {
                    }
                }
            }
        }

        if (highest != -1) {
            std::filesystem::path loadFolder = config.checkpointLoadFolder / std::to_string(highest);
            LoadStats(loadFolder / STATS_FILE_NAME);
            ppo->LoadFrom(loadFolder);

            if (config.skillTrackerConfig.loadOldVersionsFromCheckpoints) {

                int64_t targetInterval = config.skillTrackerConfig.timestepsPerVersion;
                int64_t targetTimesteps = static_cast<int64_t>(totalTimesteps);

                int64_t maxAcceptableOverage = targetInterval;

                for (int i = 0; i < config.skillTrackerConfig.maxVersions; ++i) {
                    targetTimesteps -= targetInterval;

                    json bestRating = {};
                    int64_t bestTimesteps = -1;

                    for (const auto& entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
                        if (entry.is_directory()) {
                            std::string name = entry.path().filename().string();
                            try {
                                int64_t nameVal = std::stoll(name);

                                if (nameVal < targetTimesteps + targetInterval) {
                                    if (bestTimesteps == -1 || std::abs(nameVal - targetTimesteps) < std::abs(bestTimesteps - targetTimesteps)) {
                                        std::ifstream fIn(entry.path() / STATS_FILE_NAME);
                                        if (fIn.good()) {
                                            json j = json::parse(fIn);
                                            if (j.contains("skill_rating")) {
                                                bestRating = j["skill_rating"];
                                                bestTimesteps = nameVal;
                                            }
                                        }
                                    }
                                }

                            }
                            catch (...) {
                            }
                        }
                    }

                    if (bestTimesteps != -1 && bestTimesteps >= targetTimesteps - maxAcceptableOverage) {


                        auto oldPolicy = ppo->LoadAdditionalPolicy(config.checkpointLoadFolder / std::to_string(bestTimesteps));

                        if (oldPolicy) {
                            skillTracker->AppendOldPolicy(
                                oldPolicy,
                                skillTracker->LoadRatingSet(bestRating)
                            );
                        }

                    }
                }
            }
        }
    }

    void Learner::Learn() {

#ifdef RG_PARANOID_MODE

#endif


        agentMgr->SetStepCallback(stepCallback);
        agentMgr->StartAgents();

        auto device = ppo->device;


        int64_t tsSinceSave = 0;
        Timer epochTimer;
        while (totalTimesteps < config.timestepLimit || config.timestepLimit == 0) {
            Report report = {};

            agentMgr->SetStepCallback(stepCallback);

            GameTrajectory timesteps = agentMgr->CollectTimesteps(config.timestepsPerIteration);
            double relCollectionTime = epochTimer.Elapsed();
            uint64_t timestepsCollected = timesteps.size;

            totalTimesteps += timestepsCollected;

            if (config.ppo.policyLR == 0 && config.ppo.criticLR == 0) {
#ifdef RG_CUDA_SUPPORT
                if (ppo->device.is_cuda())
                    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
                continue;
            }

            if (!config.collectionDuringLearn)
                agentMgr->disableCollection = true;

            try {
                AddNewExperience(timesteps, report);
            }
            catch (std::exception& e) {
                RG_ERR_CLOSE("Exception during Learner::AddNewExperience(): " << e.what());
            }

            Timer ppoLearnTimer;

            bool blockAgentInferDuringLearn = config.collectionDuringLearn && !device.is_cpu();
            {

                if (config.deterministic) {
                    RG_ERR_CLOSE(
                        "Learner::Learn(): Cannot run PPO learn iteration when on deterministic mode!"
                        "\nDeterministic mode is meant for performing, not training. Only collection should occur."
                    );
                }

                if (blockAgentInferDuringLearn)
                    agentMgr->disableCollection = true;

                try {
                    ppo->Learn(expBuffer, report);
                }
                catch (std::exception& e) {
                    RG_ERR_CLOSE("Exception during PPOLearner::Learn(): " << e.what());
                }

                if (blockAgentInferDuringLearn)
                    agentMgr->disableCollection = false;

                totalEpochs += config.ppo.epochs;
            }

#ifdef RG_CUDA_SUPPORT
            if (ppo->device.is_cuda())
                c10::cuda::CUDACachingAllocator::emptyCache();
#endif

            double ppoLearnTime = ppoLearnTimer.Elapsed();
            double relEpochTime = epochTimer.Elapsed();
            epochTimer.Reset();

            double consumptionTime = relEpochTime - relCollectionTime;

            if (skillTracker) {

                if (skillTracker->config.stepCallback == nullptr)
                    skillTracker->config.stepCallback = stepCallback;

                skillTracker->RunGames(ppo->policy, timestepsCollected);
                for (auto& pair : skillTracker->curRating.data) {
                    std::string metricName = "Skill Rating" + (pair.first.empty() ? "" : " " + pair.first);
                    report[metricName] = pair.second;
                }
            }

            agentMgr->GetMetrics(report);

            if (!config.collectionDuringLearn) {
                agentMgr->disableCollection = false;
            }

            double trueCollectionTime = config.collectionDuringLearn ? agentMgr->lastIterationTime : relCollectionTime;
            if (blockAgentInferDuringLearn)
                trueCollectionTime -= ppoLearnTime;
            trueCollectionTime = std::max(trueCollectionTime, relCollectionTime);

            double trueEpochTime = std::max(relEpochTime, trueCollectionTime);

            {
                report["Total Iteration Time"] = relEpochTime;

                report["Collection Time"] = relCollectionTime;
                report["Consumption Time"] = consumptionTime;
                report["Collect-Consume Overlap Time"] = (trueCollectionTime - relCollectionTime);
            }

            {
                report["Collected Steps/Second"] = static_cast<int64_t>(timestepsCollected / trueCollectionTime);
                report["Overall Steps/Second"] = static_cast<int64_t>(timestepsCollected / trueEpochTime);
                report["Timesteps Collected"] = timestepsCollected;
                report["Cumulative Timesteps"] = totalTimesteps;
            }

            if (iterationCallback) {
                iterationCallback(this, report);
            }

            {
                constexpr const char* DIVIDER = "======================";
                RG_LOG("\n");
                RG_LOG(DIVIDER << DIVIDER);
                RG_LOG("ITERATION COMPLETED:\n");
                DisplayReport(report);
                RG_LOG(DIVIDER << DIVIDER);
                RG_LOG("\n");
            }

            if (config.sendMetrics)
                metricSender->Send(report);

            tsSinceSave += timestepsCollected;
            if (tsSinceSave > config.timestepsPerSave && !config.checkpointSaveFolder.empty()) {
                Save();
                tsSinceSave = 0;
            }

            agentMgr->ResetMetrics();
        }


        agentMgr->StopAgents();
    }

    void Learner::AddNewExperience(GameTrajectory& gameTraj, Report& report) {
        RG_NOGRAD;


        gameTraj.RemoveCapacity();
        auto& trajData = gameTraj.data;

        size_t count = trajData.actions.size(0);

        size_t valPredCount = count + 1;


        torch::Tensor valPredsTensor = torch::zeros({ static_cast<int64_t>(valPredCount) });


        for (size_t i = 0; i < valPredCount; i += ppo->config.miniBatchSize) {
            size_t start = i;
            size_t end = std::min(i + ppo->config.miniBatchSize, valPredCount);
            size_t sliceEnd = (end > count) ? count : end;

            torch::Tensor statesPart = trajData.states.slice(0, start, sliceEnd);

            if (end == valPredCount) {
                auto finalNextState = torch::unsqueeze(trajData.nextStates[count - 1], 0);
                statesPart = torch::cat({ statesPart, finalNextState });
            }

            auto valPredsPart = ppo->valueNet->Forward(statesPart.to(ppo->device, true)).cpu().flatten();
            RG_ASSERT(valPredsPart.size(0) == static_cast<int64_t>(end - start));
            valPredsTensor.slice(0, start, end).copy_(valPredsPart, true);
        }

        FList valPreds = TENSOR_TO_FLIST(valPredsTensor);

#ifdef RG_CUDA_SUPPORT
        if (ppo->device.is_cuda())
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif

        float retStd = (config.standardizeReturns ? returnStats.GetSTD()[0] : 1.0f);

        torch::Tensor advantages, valueTargets;
        FList returns;
        TorchFuncs::ComputeGAE(
            TENSOR_TO_FLIST(trajData.rewards),
            TENSOR_TO_FLIST(trajData.dones),
            TENSOR_TO_FLIST(trajData.truncateds),
            valPreds,
            advantages,
            valueTargets,
            returns,
            config.gaeGamma,
            config.gaeLambda,
            retStd,
            config.rewardClipRange
        );

        float avgRet = 0.0f;
        for (const float& f : returns)
            avgRet += std::abs(f);
        avgRet /= static_cast<float>(returns.size());
        report["Avg Return"] = avgRet / retStd;

        report["Avg Advantage"] = advantages.abs().mean().item<float>();
        report["Avg Val Target"] = valueTargets.abs().mean().item<float>();

        if (config.standardizeReturns) {
            size_t numToIncrement = std::min(static_cast<size_t>(config.maxReturnsPerStatsInc), returns.size());
            returnStats.Increment(returns, numToIncrement);
        }

        ExperienceTensors expTensors{
            std::move(trajData.states),
            std::move(trajData.actions),
            std::move(trajData.logProbs),
            std::move(trajData.rewards),

    #ifdef RG_PARANOID_MODE
            std::move(trajData.debugCounters),
    #endif

            std::move(trajData.nextStates),
            std::move(trajData.dones),
            std::move(trajData.truncateds),
            std::move(valueTargets),
            std::move(advantages)
        };
        expBuffer->SubmitExperience(expTensors);
    }

    void Learner::UpdateLearningRates(float policyLR, float criticLR) {
        ppo->UpdateLearningRates(policyLR, criticLR);
    }

    std::vector<Report> Learner::GetAllGameMetrics() {
        std::vector<Report> reports;

        reports.reserve(agentMgr->agents.size() * 10);

        for (auto agent : agentMgr->agents) {
            std::lock_guard<std::mutex> lock(agent->gameStepMutex);
            for (auto game : agent->gameInsts) {
                if (!game->_metrics.data.empty())
                    reports.push_back(game->_metrics);
            }
        }

        return reports;
    }

    Learner::~Learner() {
        delete ppo;
        delete agentMgr;
        delete expBuffer;
        delete metricSender;
        delete renderSender;
        delete skillTracker;
        pybind11::finalize_interpreter();
    }

}