#!/usr/bin/env python3
"""Kuhn Poker DQN Agent - Simple Example."""

import argparse
import builtins
import logging
import statistics
from pathlib import Path
from typing import Any, Callable, Dict

import torch
import yaml
import os

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch import dqn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# noop profile decorator for line profile
if "profile" not in builtins.__dict__:
    def profile(func: Callable[Any, Any]) -> Any:
        """No-OP profile decorator."""
        def inner(*args: Any, **kwargs: Any) -> Callable[Any, Any]:
            """Identity function."""
            return func(*args, **kwargs)
        return inner


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


def init_agent(
    player_id: int,
    info_state_size: int,
    num_actions: int,
    network_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> dqn.DQN:
    """Initialize DQN agent with configuration.

    Args:
        player_id: Player ID for the agent
        info_state_size: Size of information state representation
        num_actions: Number of possible actions
        network_config: Network architecture configuration
        training_config: Training hyperparameters configuration

    Returns:
        Initialized DQN agent
    """
    return dqn.DQN(
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=network_config["hidden_layers_sizes"],
        replay_buffer_capacity=training_config["replay_buffer_capacity"],
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        epsilon_start=training_config.get("epsilon_start", 1.0),
        epsilon_end=training_config.get("epsilon_end", 0.1),
        epsilon_decay_duration=training_config["epsilon_decay_duration"],
        discount_factor=training_config.get("discount_factor", 0.99),
        min_buffer_size_to_learn=training_config.get("min_buffer_size_to_learn", 1000),
        update_target_network_every=training_config.get(
            "update_target_network_every", 1000
        ),
        learn_every=training_config.get("learn_every", 10),
    )


def load_model(agent: dqn.DQN, save_path: str) -> None:
    """Load DQN model from file if it exists.

    Args:
        agent: DQN agent instance to load weights into
        save_path: Path to the model checkpoint file
    """
    if os.path.exists(save_path):
        logging.info(f"Loading model from '{save_path}'...")
        # Add MLP class to safe globals for PyTorch 2.6+ compatibility
        torch.serialization.add_safe_globals([dqn.MLP])
        agent.load(save_path)
        logging.info("Model loaded successfully")
    else:
        logging.info("No model file found, using untrained model")


@profile
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
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
    return parser.parse_args()


@profile
def main() -> None:
    """Simple Kuhn Poker DQN training example"""
    # Parse command line arguments
    args = parse_args()

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
    agent = init_agent(0, info_state_size, num_actions, network_config, training_config)

    # 4. Create random agent (opponent)
    opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

    # Load model if in evaluation mode
    if args.eval:
        load_model(agent, model_config["save_path"])

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
    wins = losses = draws = 0
    episode_rewards = []  # For evaluation statistics

    for episode in range(num_episodes):
        reward = run_episode(env, agent, opponent, is_evaluation=is_evaluation)
        total_rewards += reward
        episode_rewards.append(reward)

        # Track win/loss/draw
        wins += reward > 0
        losses += reward < 0
        draws += reward == 0

        # Print progress periodically (only during training)
        if not is_evaluation and (episode + 1) % progress_interval == 0:
            loss = agent.loss
            win_rate = wins / (episode + 1) * 100
            avg_reward_window = sum(episode_rewards[-progress_interval:]) / min(
                progress_interval, episode + 1
            )
            replay_buffer_size = len(agent.replay_buffer)
            loss_str = f"{loss:.6f}" if loss is not None else "N/A (buffer too small)"
            logging.info(
                f"Episode {episode + 1}/{num_episodes} - "
                f"Loss: {loss_str}, "
                f"Replay Buffer: {replay_buffer_size}/{training_config['replay_buffer_capacity']}, "
                f"Win Rate: {win_rate:.2f}% ({wins}W/{losses}L/{draws}D), "
                f"Avg Reward (last {progress_interval}): {avg_reward_window:.4f}"
            )

    if is_evaluation:
        # Detailed evaluation results
        avg_reward = total_rewards / num_episodes
        win_rate = wins / num_episodes * 100
        std_dev = statistics.stdev(episode_rewards) if len(episode_rewards) > 1 else 0.0
        max_reward = max(episode_rewards)
        min_reward = min(episode_rewards)

        logging.info("=" * 80)
        logging.info("Evaluation Results")
        logging.info("=" * 80)
        logging.info(f"Total Episodes: {num_episodes}")
        logging.info(f"Wins: {wins} ({wins/num_episodes*100:.2f}%)")
        logging.info(f"Losses: {losses} ({losses/num_episodes*100:.2f}%)")
        logging.info(f"Draws: {draws} ({draws/num_episodes*100:.2f}%)")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Average Reward: {avg_reward:.4f}")
        logging.info(f"Reward Std Dev: {std_dev:.4f}")
        logging.info(f"Max Reward: {max_reward:.4f}")
        logging.info(f"Min Reward: {min_reward:.4f}")
        logging.info("=" * 80)
    else:
        # Training completed - save model
        logging.info("Training completed!")
        logging.info("Saving model...")
        save_path = model_config["save_path"]
        agent.save(save_path)
        logging.info(f"Model saved to '{save_path}'")


if __name__ == "__main__":
    main()
