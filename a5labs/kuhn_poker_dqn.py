#!/usr/bin/env python3
"""Kuhn Poker DQN Agent - Simple Example.

This script demonstrates a minimal example of training a DQN agent in the
Kuhn Poker environment using OpenSpiel.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch import dqn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_episode(
    env: rl_environment.Environment,
    agent: dqn.DQN,
    opponent: random_agent.RandomAgent,
    is_evaluation: bool = False,
) -> float:
    """Run a single episode and return the final reward.

    Args:
        env: RL environment instance
        agent: DQN agent (player 0)
        opponent: Opponent agent (player 1)
        is_evaluation: Whether this is an evaluation episode

    Returns:
        Final reward for player 0
    """
    time_step = env.reset()

    # Run episode until terminal state
    while not time_step.last():
        player_id = time_step.observations["current_player"]

        if player_id == 0:
            # DQN agent's turn
            agent_output = agent.step(time_step, is_evaluation=is_evaluation)
            action_list = [agent_output.action]
        else:
            # Opponent agent's turn
            agent_output = opponent.step(time_step, is_evaluation=is_evaluation)
            action_list = [agent_output.action]

        time_step = env.step(action_list)

    # Send final state to agents at episode end
    agent.step(time_step, is_evaluation=is_evaluation)
    opponent.step(time_step, is_evaluation=is_evaluation)

    # Return player 0's reward
    return time_step.rewards[0]


def main() -> None:
    """Simple Kuhn Poker DQN training example"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DQN agent for Kuhn Poker")
    parser.add_argument(
        "--config",
        type=str,
        default="a5labs/config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation only (skip training)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    logging.info(f"Loaded configuration from: {config_path}")

    # Extract configuration values
    network_config = config["network"]
    training_config = config["training"]
    eval_config = config["evaluation"]
    model_config = config["model"]

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
        hidden_layers_sizes=network_config["hidden_layers_sizes"],
        replay_buffer_capacity=training_config["replay_buffer_capacity"],
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        epsilon_decay_duration=training_config["epsilon_decay_duration"],
    )

    # 4. Create random agent (opponent)
    opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

    # Load model if in evaluation mode
    save_path = model_config["save_path"]
    if args.eval and Path(save_path).exists():
        logging.info(f"Loading model from '{save_path}'...")
        # Add MLP class to safe globals for PyTorch 2.6+ compatibility
        torch.serialization.add_safe_globals([dqn.MLP])
        agent.load(save_path)
        logging.info("Model loaded successfully")

    # Determine number of episodes and mode
    is_evaluation = args.eval
    num_episodes = (
        eval_config["num_episodes"]
        if is_evaluation
        else training_config["num_episodes"]
    )
    progress_interval = training_config.get("progress_print_interval", 500)

    # Single unified loop
    logging.info("Evaluating..." if is_evaluation else "Training...")

    total_rewards = 0.0
    for episode in range(num_episodes):
        reward = run_episode(env, agent, opponent, is_evaluation=is_evaluation)
        total_rewards += reward

        # Print progress periodically (only during training)
        if not is_evaluation and (episode + 1) % progress_interval == 0:
            loss = agent.loss
            logging.info(f"Episode {episode + 1}/{num_episodes} - Loss: {loss}")

    if is_evaluation:
        # Evaluation results
        avg_reward = total_rewards / num_episodes
        logging.info(f"Average reward (vs random): {avg_reward:.4f}")
    else:
        # Training completed - save model
        logging.info("Training completed!")
        logging.info("Saving model...")
        save_path = model_config["save_path"]
        agent.save(save_path)
        logging.info(f"Model saved to '{save_path}'")


if __name__ == "__main__":
    main()
