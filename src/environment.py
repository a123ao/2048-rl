"""
OpenAI Gym-like Environment wrapper for 2048 game.
"""

import numpy as np
from typing import Optional
from src.game import Game2048


class Game2048Env:
    """
    Environment wrapper for 2048 game compatible with RL algorithms.
    """
    
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            size: Grid size (default 4x4)
            seed: Random seed for reproducibility (default None)
        """
        self.size = size
        self.seed = seed
        self.game = Game2048(size, seed=seed)
        
        # Action and observation spaces
        self.num_actions = 4  # up, right, down, left
        self.observation_size = size * size  # 16 for 4x4 grid
        
        # Track episode statistics
        self.episode_reward = 0
        self.episode_steps = 0
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Args:
            seed: Optional random seed. If provided, reinitializes game with new seed.
        
        Returns:
            Initial observation (state)
        """
        if seed is not None:
            self.seed = seed
            self.game = Game2048(self.size, seed=seed)
        else:
            self.game.reset()
        
        self.episode_reward = 0
        self.episode_steps = 0
        return self.game.get_state()
    
    def step(self, action: int):
        """
        Execute one step in the environment.
        
        Args:
            action: 0=up, 1=right, 2=down, 3=left
        
        Returns:
            (observation, reward, done, info)
        """
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action: {action}")
        
        # Execute move
        moved, score_gained = self.game.move(action)
        
        # Reward is score gained (sparse reward for baseline)
        reward = float(score_gained)
        
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Get new state
        observation = self.game.get_state()
        
        # Check if episode is done
        done = self.game.over
        
        # Additional info
        info = {
            "score": self.game.score,
            "max_tile": self.game.get_max_tile(),
            "empty_cells": self.game.get_empty_count(),
            "moved": moved,
            "game_over": self.game.over,
            "game_won": self.game.won,
            "episode_reward": self.episode_reward
        }
        
        return observation, reward, done, info
    
    def get_legal_actions(self) -> list:
        """
        Get list of legal actions in current state.
        Uses game.can_move() to check if each direction has valid moves.
        
        Returns:
            List of valid action indices
        """
        legal = []
        for action in range(self.num_actions):
            if self._test_move(action):
                legal.append(action)
        
        # If no moves available, return all actions (game will be over)
        return legal if legal else list(range(self.num_actions))
    
    def _test_move(self, action: int) -> bool:
        """
        Test if a move is legal without modifying game state.
        This uses the game's can_move() method.
        
        Args:
            action: 0=up, 1=right, 2=down, 3=left
        
        Returns:
            True if move is legal, False otherwise
        """
        return self.game.can_move(action)
    
    def get_state_display(self) -> str:
        """Get ASCII representation of current game state."""
        return self.game.get_grid_display()
    
    def get_episode_stats(self) -> dict:
        """Get current episode statistics."""
        return {
            "steps": self.episode_steps,
            "total_reward": self.episode_reward,
            "score": self.game.score,
            "max_tile": self.game.get_max_tile(),
            "game_over": self.game.over
        }


if __name__ == "__main__":
    # Test environment
    print("Game2048 Environment\n")
    env = Game2048Env()
    
    state = env.reset()
    print("Initial state shape:", state.shape)
    print("Initial state:", state)
    print("\nInitial board:")
    print(env.get_state_display())
    
    # Run a few steps
    for i in range(5):
        action = np.random.randint(0, 4)
        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        observation, reward, done, info = env.step(action)
        
        print("-" * 3, end="")
        print(f"\nStep {i+1}: Action = {action_names[action]}")
        print(f"Reward: {reward}, Done: {done}")
        print(f"Info: Score={info['score']}, MaxTile={info['max_tile']}, Empty={info['empty_cells']}")
        print(env.get_state_display())
        
        if done:
            print("Episode finished!")
            break

        print()
