"""
Training script for 2048 RL agents.
Trains Random and Baseline agents and logs metrics.
"""

import numpy as np
import json
import os
from tqdm import tqdm
import torch

from src.environment import Game2048Env
from src.agents import RandomAgent, BaselineAgent, PartialRewardAgent, FullRewardAgent

# ── Agent registry ─────────────────────────────────────────────────────────
AGENT_REGISTRY = {
    'random':   RandomAgent,
    'baseline': BaselineAgent,
    'partial':  PartialRewardAgent,
    'full':     FullRewardAgent,
}

# Shared DQN constructor kwargs (can be overridden per-agent in TRAIN_CONFIG)
DQN_KWARGS = dict(
    state_size=256, action_size=4, hidden_size=256,
    learning_rate=1e-3, gamma=0.99, buffer_size=20000, batch_size=64,
)


class TrainingLogger:
    """Log training metrics."""
    
    def __init__(self):
        self.episodes = []
        self.episode_scores = []
        self.episode_max_tiles = []
        self.episode_rewards = []
        self.avg_scores = []
        self.avg_max_tiles = []
        self.avg_rewards = []
    
    def log_episode(self, episode: int, score: int, max_tile: int, reward: float):
        """Log single episode result."""
        self.episodes.append(episode)
        self.episode_scores.append(score)
        self.episode_max_tiles.append(max_tile)
        self.episode_rewards.append(reward)
    
    def compute_averages(self, window: int = 100):
        """Compute moving averages."""
        if len(self.episode_scores) >= window:
            self.avg_scores.append(np.mean(self.episode_scores[-window:]))
            self.avg_max_tiles.append(np.mean(self.episode_max_tiles[-window:]))
            self.avg_rewards.append(np.mean(self.episode_rewards[-window:]))
    
    def save(self, path: str):
        """Save logs to JSON file."""
        data = {
            "episodes": self.episodes,
            "episode_scores": self.episode_scores,
            "episode_max_tiles": self.episode_max_tiles,
            "episode_rewards": self.episode_rewards,
            "avg_scores": self.avg_scores,
            "avg_max_tiles": self.avg_max_tiles,
            "avg_rewards": self.avg_rewards
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load logs from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.episodes = data["episodes"]
        self.episode_scores = data["episode_scores"]
        self.episode_max_tiles = data["episode_max_tiles"]
        self.episode_rewards = data["episode_rewards"]
        self.avg_scores = data["avg_scores"]
        self.avg_max_tiles = data["avg_max_tiles"]
        self.avg_rewards = data["avg_rewards"]


def train_agent(agent_type: str = "baseline",
                num_episodes: int = 3000,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.05,
                epsilon_decay: float = 0.9999,
                batch_size: int = 64,
                device: str = 'cpu',
                save_dir: str = "checkpoints",
                **agent_kwargs):
    """
    Train an agent.

    Args:
        agent_type:    Key in AGENT_REGISTRY
        num_episodes:  Number of training episodes
        epsilon_*:     Exploration schedule
        batch_size:    Minibatch size for Q-network updates
        device:        'cpu' or 'cuda'
        save_dir:      Checkpoint output directory
        **agent_kwargs: Extra kwargs forwarded to the agent constructor
                        (e.g. empty_weight, corner_weight, snake_weight …)
    Returns:
        (agent, logger)
    """
    if agent_type.lower() not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent_type '{agent_type}'. Choose from: {list(AGENT_REGISTRY)}")

    os.makedirs(save_dir, exist_ok=True)
    env = Game2048Env()

    # Build agent from registry – RandomAgent has a different constructor
    cls = AGENT_REGISTRY[agent_type.lower()]
    if agent_type.lower() == "random":
        agent = cls(num_actions=4)
    else:
        agent = cls(**DQN_KWARGS, device=device, **agent_kwargs)
    
    # Logger
    logger = TrainingLogger()
    
    # Training loop
    epsilon = epsilon_start
    total_steps = 0  # global step counter for update frequency control
    print(f"\nTraining {agent_type} Agent for {num_episodes} episodes...")
    print(f"Device: {device}\n")

    with tqdm(total=num_episodes, desc=f"{agent_type} Agent") as pbar:
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                # Get legal actions
                legal_actions = env.get_legal_actions()
                
                # Select action
                if legal_actions:  # Only proceed if legal actions available
                    action = agent.select_action(state, epsilon=epsilon, training=True, legal_actions=legal_actions)
                else:  # Should rarely happen as game should be over
                    break
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience and update
                if hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done, info) # pyright: ignore[reportAttributeAccessIssue]
                    total_steps += 1
                    
                    # Update every 4 steps after warmup (standard DQN practice).
                    # Updating every step causes over-fitting to recent transitions
                    # and destabilizes Q-values, especially with sparse rewards.
                    if len(agent.memory) >= 1000 and total_steps % 4 == 0: # pyright: ignore[reportAttributeAccessIssue]
                        agent.update(batch_size)
                    
                    # Decay epsilon per-step for smoother exploration schedule
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)
                
                state = next_state
            
            # Log episode
            stats = env.get_episode_stats()
            logger.log_episode(
                episode,
                stats['score'],
                stats['max_tile'],
                stats['total_reward']
            )
            
            # Decay epsilon per-episode for non-DQN agents (random)
            if not hasattr(agent, 'memory'):
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Compute and display averages every 100 episodes
            if (episode + 1) % 100 == 0:
                logger.compute_averages(window=100)
                avg_score = logger.avg_scores[-1]
                avg_max_tile = logger.avg_max_tiles[-1]
                pbar.set_postfix({
                    'avg_score': f'{avg_score:.1f}',
                    'avg_max_tile': f'{avg_max_tile:.0f}',
                    'epsilon': f'{epsilon:.3f}'
                })
            
            pbar.update(1)
    
    # Save model and logs
    model_path = os.path.join(save_dir, f"{agent_type}_agent.pth")
    log_path = os.path.join(save_dir, f"{agent_type}_logs.json")
    
    if hasattr(agent, 'save'):
        agent.save(model_path)
        print(f"Model saved to {model_path}")
    
    logger.save(log_path)
    print(f"Logs saved to {log_path}")
    
    # Print final stats
    print(f"\n{agent_type.upper()} Agent Final Statistics:")
    print(f"  Final Avg Score (last 100 eps): {logger.avg_scores[-1]:.1f}")
    print(f"  Final Avg Max Tile (last 100 eps): {logger.avg_max_tiles[-1]:.0f}")
    print(f"  Best Score: {max(logger.episode_scores)}")
    print(f"  Best Max Tile: {max(logger.episode_max_tiles)}")
    
    return agent, logger


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Edit this list to choose which agents to train ──────────────────────
    # Uncomment any row to include it. Extra keys are forwarded to the agent
    # constructor, so you can tune reward weights directly here.
    TRAIN_CONFIG = [
        {"agent_type": "random"},
        {"agent_type": "baseline"},
        {"agent_type": "partial",  "empty_weight": 0.1},
        {"agent_type": "full",     "empty_weight": 0.1, "monotonic_weight": 1.0, "smooth_weight": 0.5},
    ]
    # ────────────────────────────────────────────────────────────────────────

    results = {}
    for cfg in TRAIN_CONFIG:
        agent, logger = train_agent(**cfg, device=device)
        results[cfg["agent_type"]] = {"agent": agent, "logger": logger}

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    return results


if __name__ == "__main__":
    results = main()
