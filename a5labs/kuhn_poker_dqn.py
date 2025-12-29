#!/usr/bin/env python3
"""Kuhn Poker DQN Agent - Simple Example.

This script demonstrates a minimal example of training a DQN agent in the
Kuhn Poker environment using OpenSpiel.
"""

import logging
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch import dqn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    """Simple Kuhn Poker DQN training example"""

    # 1. Create environment
    logging.info("Creating Kuhn Poker environment...")
    env = rl_environment.Environment("kuhn_poker", players=2)

    # 2. Get environment specifications
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    logging.info(f"Info state size: {info_state_size}")
    logging.info(f"Number of actions: {num_actions}")

    # 3. Create DQN agent
    logging.info("Creating DQN agent...")
    agent = dqn.DQN(
        player_id=0,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=[64, 64],  # Small network for faster training
        replay_buffer_capacity=5000,
        batch_size=64,
        learning_rate=0.01,
        epsilon_decay_duration=5000,  # Fast epsilon decay
    )

    # 4. Create random agent (opponent)
    random_opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

    # 5. Training loop
    logging.info("Starting training...")
    num_episodes = 5000

    for episode in range(num_episodes):
        time_step = env.reset()

        # Run episode
        while not time_step.last():
            player_id = time_step.observations["current_player"]

            if player_id == 0:
                # DQN agent's turn
                agent_output = agent.step(time_step, is_evaluation=False)
                action_list = [agent_output.action]
            else:
                # Random agent's turn
                agent_output = random_opponent.step(time_step, is_evaluation=False)
                action_list = [agent_output.action]

            time_step = env.step(action_list)

        # Send final state to agents at episode end
        agent.step(time_step, is_evaluation=False)
        random_opponent.step(time_step, is_evaluation=False)

        # Print progress periodically
        if (episode + 1) % 500 == 0:
            loss = agent.loss
            logging.info(f"Episode {episode + 1}/{num_episodes} - Loss: {loss}")

    logging.info("Training completed!")

    # 6. Evaluation (play against random agent)
    logging.info("Evaluating...")
    num_eval_episodes = 1000
    total_rewards = 0.0

    for _ in range(num_eval_episodes):
        time_step = env.reset()

        while not time_step.last():
            player_id = time_step.observations["current_player"]

            if player_id == 0:
                agent_output = agent.step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
            else:
                agent_output = random_opponent.step(time_step, is_evaluation=True)
                action_list = [agent_output.action]

            time_step = env.step(action_list)

        total_rewards += time_step.rewards[0]  # Player 0's reward
        agent.step(time_step, is_evaluation=True)
        random_opponent.step(time_step, is_evaluation=True)

    avg_reward = total_rewards / num_eval_episodes
    logging.info(f"Average reward (vs random): {avg_reward:.4f}")

    # 7. Save model
    logging.info("Saving model...")
    agent.save("kuhn_poker_dqn_model.pt")
    logging.info("Model saved to 'kuhn_poker_dqn_model.pt'")


if __name__ == "__main__":
    main()
