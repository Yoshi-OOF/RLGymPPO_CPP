#pragma once
#include <RLGymPPO_CPP/Lists.h>
#include "../FrameworkTorch.h"

namespace RLGPC {

    struct ExperienceTensors {
        torch::Tensor
            states, actions, logProbs, rewards,

#ifdef RG_PARANOID_MODE
            debugCounters,
#endif

            nextStates, dones, truncated, values, advantages;

        torch::Tensor* begin() { return &states; }
        torch::Tensor* end() { return &advantages + 1; }

        torch::Tensor* begin() const { return const_cast<torch::Tensor*>(&states); }
        torch::Tensor* end() const { return const_cast<torch::Tensor*>(&advantages) + 1; }
    };

    // https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
    class ExperienceBuffer {
    public:

        torch::Device device;
        int seed;

        ExperienceTensors data;

        int64_t curSize;
        int64_t maxSize;

        std::default_random_engine rng;

        ExperienceBuffer(int64_t maxSize, int seed, torch::Device device);

        void SubmitExperience(ExperienceTensors& data);

        struct SampleSet {
            torch::Tensor actions, logProbs, states, values, advantages;
        };
        SampleSet _GetSamples(const int64_t* indices, size_t size) const;

        // Non const car utilise notre générateur aléatoire
        std::vector<SampleSet> GetAllBatchesShuffled(int64_t batchSize);

        void Clear();

        // Combine deux tenseurs en un, en supprimant les données plus anciennes si nécessaire pour atteindre la taille cible
        static torch::Tensor _Concat(torch::Tensor t1, torch::Tensor t2, int64_t size);
    };
}
