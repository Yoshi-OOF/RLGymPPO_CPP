#pragma once
#include "../PPO/DiscretePolicy.h"
#include <RLGymPPO_CPP/Threading/GameInst.h>
#include "GameTrajectory.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>

namespace RLGPC {
    class ThreadAgent {
    public:
        void* _manager;
        std::thread thread;
        int index;
        int numGames;
        std::vector<GameInst*> gameInsts;
        std::atomic<bool> shouldRun{ false };
        std::atomic<bool> isRunning{ false };
        struct Times {
            double envStepTime = 0.0;
            double policyInferTime = 0.0;
            double trajAppendTime = 0.0;
            double* begin() { return &envStepTime; }
            double* end() { return &trajAppendTime + 1; }
        };
        Times times;
        std::vector<std::vector<GameTrajectory>> trajectories;
        std::atomic<uint64_t> stepsCollected{ 0 };
        uint64_t maxCollect;
        std::mutex gameStepMutex;
        std::mutex trajMutex;
        ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn, int index);
        void Start();
        void Stop();
        ~ThreadAgent();
    private:
        void Run();
    };
}
