#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>
#include <thread>

namespace RLGPC {

    ThreadAgentManager::ThreadAgentManager(
        DiscretePolicy* policy, DiscretePolicy* policyHalf, ExperienceBuffer* expBuffer,
        bool standardizeOBS, bool deterministic, bool blockConcurrentInfer, uint64_t maxCollect, torch::Device device)
        : policy(policy), policyHalf(policyHalf), expBuffer(expBuffer),
        standardizeOBS(standardizeOBS), deterministic(deterministic),
        blockConcurrentInfer(blockConcurrentInfer), maxCollect(maxCollect), device(device) {}

    void ThreadAgentManager::CreateAgents(EnvCreateFn func, int amount, int gamesPerAgent) {
        for (int i = 0; i < amount; ++i) {
            int numGames = gamesPerAgent;
            if (renderSender && renderDuringTraining && i == 0) {
                numGames = 1;
            }
            auto agent = new ThreadAgent(this, numGames, maxCollect / amount, func, i);
            agents.push_back(agent);
        }
    }

    void ThreadAgentManager::StartAgents() {
        for (auto* agent : agents) {
            agent->Start();
        }
    }

    void ThreadAgentManager::StopAgents() {
        for (auto* agent : agents) {
            agent->Stop();
        }
    }

    void ThreadAgentManager::SetStepCallback(StepCallback callback) {
        for (auto* agent : agents) {
            for (auto* game : agent->gameInsts) {
                game->stepCallback = callback;
            }
        }
    }

    GameTrajectory ThreadAgentManager::CollectTimesteps(uint64_t amount) {
        while (true) {
            uint64_t totalSteps = 0;
            for (auto* agent : agents) {
                totalSteps += agent->stepsCollected.load();
            }
            if (totalSteps >= amount) {
                break;
            }
            std::this_thread::yield();
        }

        GameTrajectory result;
        size_t totalTimesteps = 0;

        try {
            std::vector<GameTrajectory> trajs;
            for (auto* agent : agents) {
                std::lock_guard<std::mutex> lock(agent->trajMutex);
                for (auto& trajSet : agent->trajectories) {
                    for (auto& traj : trajSet) {
                        if (traj.size > 0) {
                            traj.data.truncateds[traj.size - 1] = (traj.data.dones[traj.size - 1].item<float>() == 0);
                            trajs.push_back(std::move(traj));
                            totalTimesteps += traj.size;
                            traj.Clear();
                        }
                    }
                }
                agent->stepsCollected = 0;
            }

            result.MultiAppend(trajs);
        }
        catch (const std::exception& e) {
            RG_ERR_CLOSE("Exception concatenating timesteps: " << e.what());
        }

        if (result.size != totalTimesteps) {
            RG_ERR_CLOSE("ThreadAgentManager::CollectTimesteps(): Timestep concatenation failed (" << result.size << " != " << totalTimesteps << ")");
        }

        lastIterationTime = iterationTimer.Elapsed();
        iterationTimer.Reset();
        return result;
    }

    void ThreadAgentManager::GetMetrics(Report& report) {
        AvgTracker avgStepRew, avgEpRew;
        for (auto* agent : agents) {
            for (auto* game : agent->gameInsts) {
                avgStepRew += game->avgStepRew;
                avgEpRew += game->avgEpRew;
            }
        }

        report["Average Step Reward"] = avgStepRew.Get();
        report["Average Episode Reward"] = avgEpRew.Get();

        ThreadAgent::Times avgTimes;
        for (auto* agent : agents) {
            for (auto itr1 = avgTimes.begin(), itr2 = agent->times.begin(); itr1 != avgTimes.end(); ++itr1, ++itr2) {
                *itr1 += *itr2;
            }
        }

        for (double& time : avgTimes) {
            time /= agents.size();
        }

        report["Env Step Time"] = avgTimes.envStepTime;
        report["Policy Infer Time"] = avgTimes.policyInferTime + avgTimes.trajAppendTime;
    }

    void ThreadAgentManager::ResetMetrics() {
        for (auto* agent : agents) {
            agent->times = {};
            std::lock_guard<std::mutex> lock(agent->gameStepMutex);
            for (auto* game : agent->gameInsts) {
                game->ResetMetrics();
            }
        }
    }

    ThreadAgentManager::~ThreadAgentManager() {
        for (auto* agent : agents) {
            delete agent;
        }
    }

}
