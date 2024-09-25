#pragma once
#include <RLGymSim_CPP/Utils/RewardFunctions/RewardFunction.h>
#include <cmath> // For std::sqrt and other math functions

using namespace RLGSC;

class SpeedTowardBallReward : public RewardFunction {
public:
    // Constructor
    SpeedTowardBallReward() = default;

    // Do nothing on game reset
    virtual void Reset(const GameState& initialState) override {}

    // Get the reward for a specific player, at the current state
    virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) override {
        // Velocity of our player
        Vec playerVel = player.phys.vel;

        // Difference in position between our player and the ball
        Vec posDiff = state.ball.pos - player.phys.pos;

        // Determine the distance to the ball
        float distToBall = posDiff.Length();

        // Avoid division by zero
        if (distToBall == 0) return 0;

        // Normalize the posDiff vector to get the direction to the ball
        Vec dirToBall = posDiff / distToBall;

        // Use a dot product to determine how much of the player's velocity is toward the ball
        float speedTowardBall = playerVel.Dot(dirToBall);

        // Reward only if the player is moving toward the ball
        if (speedTowardBall > 0) {
            // Return a reward scaled by the maximum car speed
            float reward = speedTowardBall / CommonValues::CAR_MAX_SPEED;
            return reward;
        }
        else {
            // No reward if moving away from the ball
            return 0;
        }
    }
};
