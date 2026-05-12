"""
RL Agents for 2048: Random and DQN-based Baseline
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional


class BaseAgent:
    """Abstract base class for agents."""

    def select_action(self, state, epsilon=0.0, training=False, legal_actions: Optional[list] = None) -> int:
        """Select an action based on the current state."""
        ...
    
    def update(self, batch_size):
        """Update the agent's knowledge based on experience."""
        ...

    def save(self, path):
        """Save the agent's model to disk."""
        ...
    
    def load(self, path):
        """Load the agent's model from disk."""
        ...


class RandomAgent(BaseAgent):
    """Agent that takes random actions."""
    
    def __init__(self, num_actions: int = 4):
        self.num_actions = num_actions
    
    def select_action(self, state, epsilon=0.0, training=False, legal_actions: Optional[list] = None):
        """Select a random action (ignoring legal_actions for random agent)."""
        if legal_actions:
            return np.random.choice(legal_actions)
        return np.random.randint(0, self.num_actions)


class DQNNetwork(nn.Module):
    """MLP for DQN with 2 hidden layers."""
    
    def __init__(self, state_size: int = 256, hidden_size: int = 256, action_size: int = 4):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BaselineAgent(BaseAgent):
    """DQN-based agent using score as reward."""
    
    def __init__(self, 
                 state_size: int = 256,
                 action_size: int = 4,
                 hidden_size: int = 256,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 20000,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 seed: Optional[int] = None):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of encoded state (256 = 16 cells × 16 one-hot bins)
            action_size: Number of actions (4)
            hidden_size: Hidden layer size
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Experience replay buffer size
            batch_size: Batch size for training
            device: 'cpu' or 'cuda'
            seed: Random seed for reproducibility (default None)
        """
        # Set random seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, hidden_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, hidden_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.HuberLoss()
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Training step counter
        self.steps_done = 0
        self.update_target_freq = 500  # Update target network every N steps
    
    def select_action(self, state, epsilon: float = 0.0, training: bool = False, legal_actions: Optional[list] = None):
        """
        Select action using epsilon-greedy policy with optional legal action masking.
        
        Args:
            state: Current state (numpy array)
            epsilon: Exploration rate
            training: Whether in training mode
            legal_actions: List of legal action indices (optional). If provided, only these actions are considered.
        
        Returns:
            Action index (0-3)
        """
        if training and random.random() < epsilon:
            # Exploration: random action
            if legal_actions:
                return random.choice(legal_actions)
            return np.random.randint(0, self.action_size)
        
        # Exploitation: greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self._preprocess_state(state)).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Apply legal action mask if provided
            if legal_actions:
                # Create mask for legal actions
                q_values_masked = q_values.clone()
                for action in range(self.action_size):
                    if action not in legal_actions:
                        q_values_masked[0, action] = float('-inf')
                
                # Select best legal action
                best_action = q_values_masked.argmax(dim=1).item()
                
                # Fallback: if all masked (shouldn't happen), pick random legal action
                if best_action not in legal_actions:
                    best_action = legal_actions[0] if legal_actions else 0
                
                return best_action
            
            best_action = q_values.argmax(dim=1).item()
            # Uncomment for debugging:
            # print(f"Debug - Q-values: {q_values[0].cpu().numpy()}, Selected action: {best_action}")
            return best_action
    
    def _preprocess_state(self, state) -> np.ndarray:
        """One-hot encode log2-encoded state (16 cells × 16 possible values = 256 dims).
        
        Tile values are categorical (2,4,8,...), not ordinal, so one-hot gives
        much richer signal than a scalar normalized value.
        """
        s = np.array(state, dtype=np.int32).clip(0, 15)
        encoded = np.zeros((16, 16), dtype=np.float32)
        encoded[np.arange(16), s] = 1.0
        return encoded.flatten()

    def remember(self, state, action, reward, next_state, done, info_dict: dict = None):
        """Store experience in replay buffer with log2-scaled reward."""
        # log2 scaling: 0 for no-merge moves, log2(score) for merges
        scaled_reward = float(np.log2(reward)) if reward > 0 else 0.0
        self.memory.append((self._preprocess_state(state), action, scaled_reward,
                            self._preprocess_state(next_state), done))
    
    def update(self, batch_size=None):
        """
        Update Q-network using experience replay.
        
        Returns:
            Loss value (or None if insufficient samples)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions)
        
        # Double DQN: online network selects action, target network evaluates it
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss and update
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        
        # Set to eval mode for inference
        self.q_network.eval()
        self.target_network.eval()


class PartialRewardAgent(BaselineAgent):
    """
    DQN agent with Partial Reward Shaping.
    
    Reward formula: R = log2(r_score) + alpha * r_empty
    
    This encourages the agent to maintain empty cells for better maneuverability.
    """
    
    def __init__(self, 
                 state_size: int = 256,
                 action_size: int = 4,
                 hidden_size: int = 256,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 20000,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 seed: Optional[int] = None,
                 empty_weight: float = 0.1):
        """
        Initialize Partial Reward Shaping DQN agent.
        
        Args:
            ... (same as BaselineAgent)
            empty_weight: Weight alpha for empty cell reward
        """
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            seed=seed
        )
        self.empty_weight = empty_weight
    
    def remember(self, state, action, reward, next_state, done, info_dict: dict = None):
        """
        Store experience with reward shaping: R = log2(r_score) + alpha * r_empty
        """
        r_score = float(np.log2(reward)) if reward > 0 else 0.0
        r_empty = float(info_dict.get('empty_cells', 0)) if info_dict is not None else 0.0
        shaped_reward = r_score + self.empty_weight * r_empty
        self.memory.append((self._preprocess_state(state), action, shaped_reward,
                            self._preprocess_state(next_state), done))


class FullRewardAgent(BaselineAgent):
    """
    DQN agent with Full Reward Shaping.
    
    Reward formula: R = log2(r_score) + alpha*r_empty + beta*r_corner + gamma*r_monotonic
    
    - r_empty:     number of empty cells (maneuverability)
    - r_corner:    1 if max tile is in a corner (positional strategy)
    - r_monotonic: fraction of rows/cols that are monotone (orderly board)
    """
    
    def __init__(self, 
                 state_size: int = 256,
                 action_size: int = 4,
                 hidden_size: int = 256,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 20000,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 seed: Optional[int] = None,
                 empty_weight: float = 0.1,
                 corner_weight: float = 1.0,
                 monotonic_weight: float = 1.0):
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            seed=seed
        )
        self.empty_weight = empty_weight
        self.corner_weight = corner_weight
        self.monotonic_weight = monotonic_weight
    
    def remember(self, state, action, reward, next_state, done, info_dict: dict = None):
        """
        Store experience with full reward shaping:
        R = log2(r_score) + alpha*r_empty + beta*r_corner + gamma*r_monotonic
        """
        r_score = float(np.log2(reward)) if reward > 0 else 0.0
        r_empty = float(info_dict.get('empty_cells', 0)) if info_dict is not None else 0.0
        
        # next_state is log2-encoded flat array (0=empty, k=tile 2^k)
        grid = np.array(next_state, dtype=np.float32).reshape(4, 4)
        r_corner = self._corner_reward(grid)
        r_monotonic = self._monotonic_reward(grid)
        
        shaped_reward = (r_score
                         + self.empty_weight * r_empty
                         + self.corner_weight * r_corner
                         + self.monotonic_weight * r_monotonic)
        self.memory.append((self._preprocess_state(state), action, float(shaped_reward),
                            self._preprocess_state(next_state), done))
    
    @staticmethod
    def _corner_reward(grid: np.ndarray) -> float:
        """1.0 if the max tile is in one of the four corners, else 0.0."""
        max_val = np.max(grid)
        corners = [grid[0, 0], grid[0, 3], grid[3, 0], grid[3, 3]]
        return 1.0 if max_val in corners else 0.0
    
    @staticmethod
    def _monotonic_reward(grid: np.ndarray) -> float:
        """Fraction of rows and columns that are monotonically ordered (0.0–1.0)."""
        score = 0
        for i in range(4):
            row = grid[i]
            if (all(row[j] <= row[j + 1] for j in range(3)) or
                    all(row[j] >= row[j + 1] for j in range(3))):
                score += 1
            col = grid[:, i]
            if (all(col[j] <= col[j + 1] for j in range(3)) or
                    all(col[j] >= col[j + 1] for j in range(3))):
                score += 1
        return score / 8.0


class BestRewardAgent(BaselineAgent):
    """
    DQN agent with optimized reward shaping.

    Reward formula:
        R = log2(r_score) + alpha * r_empty + beta * r_snake + gamma * r_smooth

    Replaces the binary corner/monotonic rewards with two continuous signals:

    r_snake:
        Dot product of the log2-tile values with a fixed snake-order weight
        matrix.  Large tiles in high-weight positions score higher.  This
        naturally encourages the agent to keep the largest tile in one corner
        and arrange tiles in a decreasing snake path – without rigidly locking
        it to a single corner like r_corner does.

        Weight matrix (normalized to [0, 1]):
            15 14 13 12       ← top row: highest weights
             8  9 10 11
             7  6  5  4
             0  1  2  3       ← bottom row: lowest weights

    r_smooth:
        Negative sum of absolute log2-differences between every pair of
        adjacent tiles (horizontal + vertical).  Penalizes jagged boards where
        large and small tiles sit next to each other.  A smooth gradient makes
        future merges easier.  Normalized by the number of adjacent pairs (24).
    """

    # Snake-order weight matrix, normalized to [0, 1]
    _SNAKE_WEIGHTS: np.ndarray = np.array([
        [15, 14, 13, 12],
        [ 8,  9, 10, 11],
        [ 7,  6,  5,  4],
        [ 0,  1,  2,  3],
    ], dtype=np.float32) / 15.0   # max weight = 15

    def __init__(self,
                 state_size: int = 256,
                 action_size: int = 4,
                 hidden_size: int = 256,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 20000,
                 batch_size: int = 64,
                 device: str = 'cpu',
                 seed: Optional[int] = None,
                 empty_weight: float = 0.1,
                 snake_weight: float = 1.0,
                 smooth_weight: float = 0.5):
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        self.empty_weight  = empty_weight
        self.snake_weight  = snake_weight
        self.smooth_weight = smooth_weight

    def remember(self, state, action, reward, next_state, done, info_dict: dict = None):
        """
        Store experience with optimized reward shaping:
            R = log2(r_score) + alpha*r_empty + beta*r_snake + gamma*r_smooth
        """
        r_score = float(np.log2(reward)) if reward > 0 else 0.0
        r_empty = float(info_dict.get('empty_cells', 0)) if info_dict is not None else 0.0

        # next_state is log2-encoded flat array (0=empty, k=tile 2^k)
        grid = np.array(next_state, dtype=np.float32).reshape(4, 4)
        r_snake  = self._snake_reward(grid)
        r_smooth = self._smooth_reward(grid)

        shaped_reward = (r_score
                         + self.empty_weight  * r_empty
                         + self.snake_weight  * r_snake
                         + self.smooth_weight * r_smooth)
        self.memory.append((self._preprocess_state(state), action, float(shaped_reward),
                            self._preprocess_state(next_state), done))

    @classmethod
    def _snake_reward(cls, grid: np.ndarray) -> float:
        """
        Weighted sum of log2-tile values using the snake weight matrix.
        Higher score means large tiles are sitting in high-value positions.
        Normalized by the theoretical maximum (all cells = tile 2^11 = 2048).
        """
        score = float(np.sum(grid * cls._SNAKE_WEIGHTS))
        max_possible = 11.0 * cls._SNAKE_WEIGHTS.sum()   # 11 * 60 = 660
        return score / max_possible

    @staticmethod
    def _smooth_reward(grid: np.ndarray) -> float:
        """
        Smoothness: negative mean absolute difference between adjacent log2-tiles.
        Range approximately [-1, 0]; closer to 0 means smoother board.
        """
        diff = 0.0
        pairs = 0
        for r in range(4):
            for c in range(4):
                if c + 1 < 4:                          # horizontal neighbour
                    diff += abs(grid[r, c] - grid[r, c + 1])
                    pairs += 1
                if r + 1 < 4:                          # vertical neighbour
                    diff += abs(grid[r, c] - grid[r + 1, c])
                    pairs += 1
        # normalize: max possible diff per pair is 11 (log2 values 0..11)
        return -(diff / pairs) / 11.0

