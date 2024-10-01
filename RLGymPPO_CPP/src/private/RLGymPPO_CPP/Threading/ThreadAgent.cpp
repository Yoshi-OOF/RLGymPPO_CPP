#include "ThreadAgent.h"
#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>
#include <chrono>
#include <cassert>

namespace RLGPC {

    torch::Tensor MakeGamesOBSTensor(const std::vector<GameInst*>& games) {
        assert(!games.empty());
        std::vector<torch::Tensor> obsTensors;
        obsTensors.reserve(games.size());
        for (const auto& game : games)
            obsTensors.push_back(FLIST2_TO_TENSOR(game->curObs));
        try {
            return torch::cat(obsTensors, 0);
        }
        catch (const std::exception& e) {
            RG_ERR_CLOSE("Failed to concat OBS tensors: " << e.what());
            return {};
        }
    }

    ThreadAgent::ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn, int index)
        : _manager(manager), index(index), numGames(numGames), maxCollect(maxCollect), stepsCollected(0) {
        trajectories.resize(numGames);
        gameInsts.reserve(numGames);
        for (int i = 0; i < numGames; ++i) {
            auto envCreateResult = envCreateFn();
            gameInsts.push_back(new GameInst(envCreateResult.gym, envCreateResult.match));
            trajectories[i].resize(envCreateResult.match->playerAmount);
        }
    }

    void ThreadAgent::Start() {
        shouldRun = true;
        thread = std::thread(&ThreadAgent::Run, this);
        thread.detach();
    }

    void ThreadAgent::Stop() {
        shouldRun = false;
        while (isRunning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void ThreadAgent::Run() {
        RG_NOGRAD;
        isRunning = true;
        auto mgr = static_cast<ThreadAgentManager*>(_manager);
        auto& games = gameInsts;
        auto device = mgr->device;
        bool render = (mgr->renderSender != nullptr);
        if (render && mgr->renderDuringTraining && index != 0)
            render = false;
        bool deterministic = mgr->deterministic;
        bool blockConcurrentInfer = mgr->blockConcurrentInfer;
        Timer stepTimer;
        for (auto& game : games)
            game->Start();
        torch::Tensor curObsTensor = MakeGamesOBSTensor(games);
        constexpr bool halfPrec = false;
        auto policy = (halfPrec && mgr->policyHalf) ? mgr->policyHalf : mgr->policy;
        while (shouldRun) {
            if (render)
                stepTimer.Reset();
            while (stepsCollected > maxCollect)
                std::this_thread::yield();
            while (mgr->disableCollection)
                std::this_thread::yield();
            torch::Tensor curObsTensorDevice;
            if (halfPrec) {
                curObsTensorDevice = curObsTensor.to(RG_HALFPERC_TYPE).to(device, true);
            }
            else {
                curObsTensorDevice = curObsTensor.to(device, true);
            }
            Timer policyInferTimer;
            if (blockConcurrentInfer)
                mgr->inferMutex.lock();
            auto actionResults = policy->GetAction(curObsTensorDevice, deterministic);
            if (blockConcurrentInfer)
                mgr->inferMutex.unlock();
            if (halfPrec) {
                actionResults.action = actionResults.action.to(torch::kFloat32);
                actionResults.logProb = actionResults.logProb.to(torch::kFloat32);
            }
            double policyInferTime = policyInferTimer.Elapsed();
            times.policyInferTime += policyInferTime;
            Timer gymStepTimer;
            {
                std::lock_guard<std::mutex> lock(gameStepMutex);
                size_t actionsOffset = 0;
                std::vector<RLGSC::Gym::StepResult> stepResults(numGames);
                for (int i = 0; i < numGames; ++i) {
                    auto& game = games[i];
                    int numPlayers = game->match->playerAmount;
                    auto actionSlice = actionResults.action.slice(0, actionsOffset, actionsOffset + numPlayers);
                    stepResults[i] = game->Step(TENSOR_TO_ILIST(actionSlice));
                    actionsOffset += numPlayers;
                }
                assert(actionsOffset == static_cast<size_t>(actionResults.action.size(0)));
                double envStepTime = gymStepTimer.Elapsed();
                times.envStepTime += envStepTime;
                torch::Tensor nextObsTensor = MakeGamesOBSTensor(games);
                if (!render) {
                    Timer trajAppendTimer;
                    {
                        std::lock_guard<std::mutex> trajLock(trajMutex);
                        for (int i = 0, playerOffset = 0; i < numGames; ++i) {
                            int numPlayers = games[i]->match->playerAmount;
                            auto& stepResult = stepResults[i];
                            float done = static_cast<float>(stepResult.done);
                            float truncated = 0.0f;
                            auto tDone = torch::tensor(done);
                            auto tTruncated = torch::tensor(truncated);
                            for (int j = 0; j < numPlayers; ++j) {
                                trajectories[i][j].AppendSingleStep({
                                    curObsTensor[playerOffset + j],
                                    actionResults.action[playerOffset + j],
                                    actionResults.logProb[playerOffset + j],
                                    torch::tensor(stepResult.reward[j]),
        #ifdef RG_PARANOID_MODE
                                    torch::Tensor(),
        #endif
                                    nextObsTensor[playerOffset + j],
                                    tDone,
                                    tTruncated
                                    });
                            }
                            stepsCollected += numPlayers;
                            playerOffset += numPlayers;
                        }
                    }
                    times.trajAppendTime += trajAppendTimer.Elapsed();
                }
                else {
                    auto renderSender = mgr->renderSender;
                    auto renderGame = games[0];
                    renderSender->Send(renderGame->gym->prevState, renderGame->gym->match->prevActions);
                    using namespace std::chrono;
                    static auto lastRenderTime = high_resolution_clock::now();
                    auto durationSince = high_resolution_clock::now() - lastRenderTime;
                    lastRenderTime = high_resolution_clock::now();
                    double timeTaken = stepTimer.Elapsed();
                    double targetTime = (1.0 / 120.0) * renderGame->gym->tickSkip / mgr->renderTimeScale;
                    double sleepTime = std::max(targetTime - timeTaken, 0.0);
                    std::this_thread::sleep_for(microseconds(static_cast<int64_t>(sleepTime * 1e6)));
                }
                curObsTensor = nextObsTensor;
            }
        }
        isRunning = false;
    }

    ThreadAgent::~ThreadAgent() {
        for (auto& game : gameInsts)
            delete game;
    }

}
