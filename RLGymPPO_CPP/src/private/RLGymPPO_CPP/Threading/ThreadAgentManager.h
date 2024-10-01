#pragma once
#include "ThreadAgent.h"
#include "../PPO/ExperienceBuffer.h"
#include <RLGymPPO_CPP/Util/Report.h>
#include <RLGymPPO_CPP/Util/WelfordRunningStat.h>
#include <RLGymPPO_CPP/Util/Timer.h>
#include <RLGymPPO_CPP/Util/RenderSender.h>
#include <mutex>
#include <vector>

namespace RLGPC {
    class ThreadAgentManager {
    public:
        DiscretePolicy* policy;
        DiscretePolicy* policyHalf;
        std::vector<ThreadAgent*> agents;
        ExperienceBuffer* expBuffer;
        std::mutex expBufferMutex;
        std::mutex inferMutex;
        bool standardizeOBS;
        bool deterministic;
        bool blockConcurrentInfer;
        uint64_t maxCollect;
        torch::Device device;
        RenderSender* renderSender = nullptr;
        bool renderDuringTraining = false;
        float renderTimeScale = 1.0f;
        bool disableCollection = false;
        Timer iterationTimer;
        double lastIterationTime = 0.0;
        WelfordRunningStat obsStats;

        ThreadAgentManager(
            DiscretePolicy* policy, DiscretePolicy* policyHalf, ExperienceBuffer* expBuffer,
            bool standardizeOBS, bool deterministic, bool blockConcurrentInfer, uint64_t maxCollect, torch::Device device);

        void CreateAgents(EnvCreateFn func, int amount, int gamesPerAgent);
        void StartAgents();
        void StopAgents();
        void SetStepCallback(StepCallback callback);
        void GetMetrics(Report& report);
        void ResetMetrics();
        GameTrajectory CollectTimesteps(uint64_t amount);
        ~ThreadAgentManager();

        // Supprimer les constructeurs par copie et par déplacement
        ThreadAgentManager(const ThreadAgentManager&) = delete;
        ThreadAgentManager& operator=(const ThreadAgentManager&) = delete;
        ThreadAgentManager(ThreadAgentManager&&) = delete;
        ThreadAgentManager& operator=(ThreadAgentManager&&) = delete;
    };
}
