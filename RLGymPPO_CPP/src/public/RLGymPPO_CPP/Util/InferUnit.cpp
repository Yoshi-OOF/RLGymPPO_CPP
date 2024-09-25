#include "InferUnit.h"

#include <RLGymPPO_CPP/PPO/DiscretePolicy.h>
#include <RLGymPPO_CPP/PPO/ValueEstimator.h>
#include <RLGymPPO_CPP/FrameworkTorch.h>
#include <torch/csrc/api/include/torch/serialize.h>

using namespace RLGSC;
using namespace RLGPC;

RLGPC::InferUnit::InferUnit(
    OBSBuilder* obsBuilder, ActionParser* actionParser,
    std::filesystem::path modelPath, bool isPolicy, int obsSize, const IList& layerSizes, bool gpu)
    : obsBuilder(obsBuilder), actionParser(actionParser), policy(nullptr), critic(nullptr) {

    RG_LOG("InferUnit():");

    RG_LOG(" > Creating policy/critic...");
    torch::Device device = gpu ? torch::kCUDA : torch::kCPU;

    try {
        if (isPolicy) {
            policy = new DiscretePolicy(obsSize, actionParser->GetActionAmount(), layerSizes, device);
        }
        else {
            critic = new ValueEstimator(obsSize, layerSizes, device);
        }

        RG_LOG(" > > Loading policy/critic...");
        std::ifstream streamIn(modelPath, std::ios::binary);
        if (!streamIn.is_open()) {
            RG_ERR_CLOSE("Can't open model file : " << modelPath);
        }

        if (policy) {
            torch::load(policy->seq, streamIn, device);
        }
        else {
            torch::load(critic->seq, streamIn, device);
        }

    }
    catch (const std::exception& e) {
        RG_ERR_CLOSE(
            "Failed to load model, checkpoint may be corrupt or of different model arch.\n"
            << "Exception : " << e.what()
        );
    }

    RG_LOG(" > Terminé !");
}

RLGPC::InferUnit::~InferUnit() {
    delete policy;
    delete critic;
}

RLGSC::FList RLGPC::InferUnit::GetObs(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction) {
    return obsBuilder->BuildOBS(player, state, prevAction);
}

RLGSC::FList2 RLGPC::InferUnit::GetObs(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions) {
    FList2 obsSet;
    if (state.players.size() != prevActions.size()) {
        RG_ERR_CLOSE("InferUnit::GetObs: The size of state.players and prevActions does not match.");
    }
    for (size_t i = 0; i < state.players.size(); i++) {
        obsSet.push_back(obsBuilder->BuildOBS(state.players[i], state, prevActions[i]));
    }
    return obsSet;
}

#define ASSERT_RIGHT_TYPE(name, otherName) \
if ((name) == nullptr) RG_ERR_CLOSE("InferUnit: Failed to infer " #name " because this inference unit was created to infer " #otherName ".");

ActionSet RLGPC::InferUnit::InferPolicyAll(
    const GameState& state, const ActionSet& prevActions,
    bool deterministic, float temperature
) {

    ASSERT_RIGHT_TYPE(policy, critic);

    FList2 obsSet = GetObs(state, prevActions);

    RG_NOGRAD;
    policy->temperature = temperature;
    torch::Tensor inputTen = FLIST2_TO_TENSOR(obsSet).to(policy->device);
    auto actionResult = policy->GetAction(inputTen, deterministic);
    auto actionParserInput = TENSOR_TO_ILIST(actionResult.action);

    return actionParser->ParseActions(actionParserInput, state);
}

Action RLGPC::InferUnit::InferPolicySingle(
    const PlayerData& player, const GameState& state, const Action& prevAction,
    bool deterministic, float temperature
) {
    ASSERT_RIGHT_TYPE(policy, critic);

    FList obs = GetObs(player, state, prevAction);

    size_t playerIndex = 0;
    for (size_t i = 0; i < state.players.size(); i++) {
        if (state.players[i].carId == player.carId) {
            playerIndex = i;
            break;
        }
    }

    RG_NOGRAD;
    policy->temperature = temperature;
    torch::Tensor inputTen = torch::tensor(obs).to(policy->device);
    auto actionResult = policy->GetAction(inputTen, deterministic);
    IList actionParserInput(state.players.size(), 0);
    actionParserInput[playerIndex] = actionResult.action.item<int>();

    return actionParser->ParseActions(actionParserInput, state)[playerIndex];
}

RLGSC::FList RLGPC::InferUnit::InferPolicySingleDistrib(
    const PlayerData& player, const GameState& state, const Action& prevAction,
    float temperature
) {
    ASSERT_RIGHT_TYPE(policy, critic);

    FList obs = GetObs(player, state, prevAction);

    RG_NOGRAD;
    policy->temperature = temperature;
    torch::Tensor inputTen = torch::tensor(obs).to(policy->device);
    return TENSOR_TO_FLIST(policy->GetActionProbs(inputTen).reshape({ policy->actionAmount }));
}

RLGSC::FList RLGPC::InferUnit::InferCriticAll(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions) {
    ASSERT_RIGHT_TYPE(critic, policy);

    FList2 obsSet = GetObs(state, prevActions);

    RG_NOGRAD;
    torch::Tensor inputTen = FLIST2_TO_TENSOR(obsSet).to(critic->device);
    return TENSOR_TO_FLIST(critic->Forward(inputTen).cpu());
}

float RLGPC::InferUnit::InferCriticSingle(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction) {
    ASSERT_RIGHT_TYPE(critic, policy);

    FList obs = GetObs(player, state, prevAction);

    RG_NOGRAD;
    torch::Tensor inputTen = torch::tensor(obs).to(critic->device);
    return critic->Forward(inputTen).cpu().item<float>();
}
