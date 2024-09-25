#include "ExperienceBuffer.h"

#include "../Util/TorchFuncs.h"

using namespace torch;

RLGPC::ExperienceBuffer::ExperienceBuffer(int64_t maxSize, int seed, torch::Device device)
    : maxSize(maxSize), seed(seed), device(device), rng(seed), curSize(0) {}

void RLGPC::ExperienceBuffer::SubmitExperience(ExperienceTensors& _data) {
    RG_NOGRAD;

    bool empty = curSize == 0;

#ifdef RG_PARANOID_MODE
    // Conserver une copie du résultat cible de la concaténation pour le suivi
    auto rewardsTarget = _Concat(
        curSize > 0 ? data.rewards.slice(0, 0, curSize) : data.rewards,
        _data.rewards,
        maxSize
    );
#endif

    auto dataItr = data.begin();
    auto _dataItr = _data.begin();

    while (dataItr != data.end() && _dataItr != _data.end()) {
        Tensor& ourTen = *dataItr;
        Tensor& addTen = *_dataItr;

        int64_t addAmount = addTen.size(0);

        if (addAmount > maxSize) {
            addTen = addTen.slice(0, addAmount - maxSize, addAmount);
            addAmount = maxSize;
        }

        int64_t overflow = std::max((curSize + addAmount) - maxSize, int64_t(0));
        int64_t startIdx = curSize - overflow;
        int64_t endIdx = curSize + addAmount - overflow;

        if (empty) {
            // Initialiser le tenseur
            auto sizes = addTen.sizes().vec();
            sizes[0] = maxSize;
            ourTen = torch::empty(sizes);

            // Remplir ourTen avec NAN pour détecter l'utilisation de données non initialisées
            ourTen.fill_(std::numeric_limits<float>::quiet_NaN());

            RG_PARA_ASSERT(ourTen.size(0) == maxSize);
        }
        else {
            // Nous avons déjà des données
            if (overflow > 0) {
                auto fromData = ourTen.slice(0, overflow, curSize).clone();
                auto toView = ourTen.slice(0, 0, curSize - overflow);
                toView.copy_(fromData);

                RG_PARA_ASSERT(ourTen[curSize - overflow - 1].equal(ourTen[curSize - 1]));
            }
        }

        auto ourTenInsertView = ourTen.slice(0, startIdx, endIdx);
        ourTenInsertView.copy_(addTen);
        RG_PARA_ASSERT(ourTen[endIdx - 1].equal(addTen[addTen.size(0) - 1]));

        ++dataItr;
        ++_dataItr;
    }

    curSize = std::min(curSize + _data.begin()->size(0), maxSize);

#ifdef RG_PARANOID_MODE
    // Vérifier que les tenseurs ont la bonne taille
    for (const Tensor& t : data)
        RG_PARA_ASSERT(t.size(0) == maxSize);

    // Vérifier que notre calcul des récompenses correspond à la cible
    RG_PARA_ASSERT(data.rewards.slice(0, 0, curSize).equal(rewardsTarget));

    // Vérifier que les compteurs de débogage augmentent correctement
    auto debugCounters = TENSOR_TO_ILIST(data.debugCounters.slice(0, 0, curSize).cpu());
    for (size_t i = 2; i < debugCounters.size(); i++) {
        if (debugCounters[i] <= debugCounters[i - 1] && debugCounters[i - 1] <= debugCounters[i - 2])
            RG_ERR_CLOSE("Le compteur de débogage a échoué à l'index " << i);
    }
#endif
}

RLGPC::ExperienceBuffer::SampleSet RLGPC::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {
    Tensor tIndices = torch::from_blob(const_cast<int64_t*>(indices), { static_cast<int64_t>(size) }, kLong).clone();

    SampleSet result;
    result.actions = data.actions.index_select(0, tIndices);
    result.logProbs = data.logProbs.index_select(0, tIndices);
    result.states = data.states.index_select(0, tIndices);
    result.values = data.values.index_select(0, tIndices);
    result.advantages = data.advantages.index_select(0, tIndices);
    return result;
}

std::vector<RLGPC::ExperienceBuffer::SampleSet> RLGPC::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize) {
    if (curSize == 0 || batchSize <= 0)
        return {};

    std::vector<int64_t> indices(curSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<SampleSet> result;
    for (int64_t startIdx = 0; startIdx + batchSize <= curSize; startIdx += batchSize) {
        result.push_back(_GetSamples(indices.data() + startIdx, batchSize));
    }

    return result;
}

void RLGPC::ExperienceBuffer::Clear() {
    data = ExperienceTensors();
    curSize = 0;
    rng.seed(seed);
}

Tensor RLGPC::ExperienceBuffer::_Concat(torch::Tensor t1, torch::Tensor t2, int64_t size) {
    Tensor t;
    int64_t len1 = t1.size(0);
    int64_t len2 = t2.size(0);

    if (len2 >= size) {
        // On ne peut utiliser que la fin de t2
        t = t2.slice(0, len2 - size, len2);
    }
    else if (len1 + len2 > size) {
        // Les deux ne rentrent pas, on coupe le début de t1
        t = torch::cat({ t1.slice(0, len1 + len2 - size, len1), t2 }, 0);
    }
    else {
        // Les deux rentrent
        t = torch::cat({ t1, t2 }, 0);
    }

    return t;
}
