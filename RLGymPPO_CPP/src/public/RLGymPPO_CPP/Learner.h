#pragma once

#include "Threading/GameInst.h"
#include "Util/WelfordRunningStat.h"
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "LearnerConfig.h"

namespace RLGPC {

    typedef std::function<void(class Learner*, Report&)> IterationCallback;

    // https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
    class RG_IMEXPORT Learner {
    public:
        LearnerConfig config;

        class PPOLearner* ppo;
        class ThreadAgentManager* agentMgr;
        class ExperienceBuffer* expBuffer;
        EnvCreateFn envCreateFn;
        MetricSender* metricSender;
        RenderSender* renderSender;

        struct SkillTracker* skillTracker;

        int obsSize;
        int actionAmount;

        std::string runID;

        uint64_t
            totalTimesteps,
            totalEpochs;

        WelfordRunningStat returnStats;

        Learner(EnvCreateFn envCreateFunc, LearnerConfig config);
        void Learn();
        void AddNewExperience(class GameTrajectory& gameTraj, Report& report);

        void UpdateLearningRates(float policyLR, float criticLR);

        std::vector<Report> GetAllGameMetrics();

        void Save();
        void Load();
        void SaveStats(std::filesystem::path path);
        void LoadStats(std::filesystem::path path);

        IterationCallback iterationCallback = nullptr;
        StepCallback stepCallback = nullptr;

        RG_NO_COPY(Learner);

        ~Learner();
    };
}
