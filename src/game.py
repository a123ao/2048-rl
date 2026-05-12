"""
2048 Game Logic - Python Implementation
Based on JavaScript game_manager.js and grid.js
"""

import random
from typing import List, Tuple, Optional
import numpy as np

from src.utils import clear_screen

class Tile:
    """Represents a single tile in the game grid."""
    
    def __init__(self, x: int, y: int, value: int = 2):
        self.x = x
        self.y = y
        self.value = value
        self.merged_from: Optional[List['Tile']] = None  # Tracks which tiles merged to create this tile
    
    def __repr__(self):
        return f"Tile({self.x}, {self.y}, {self.value})"


class Grid:
    """Manages the 4x4 game grid."""
    
    def __init__(self, size: int = 4):
        self.size = size
        self.cells = self._empty()
    
    def _empty(self) -> List[List[Optional[Tile]]]:
        """Create an empty grid."""
        return [[None for _ in range(self.size)] for _ in range(self.size)]
    
    def available_cells(self) -> List[Tuple[int, int]]:
        """Get list of available empty cells."""
        cells = []
        for x in range(self.size):
            for y in range(self.size):
                if self.cells[x][y] is None:
                    cells.append((x, y))
        return cells
    
    def cells_available(self) -> bool:
        """Check if any cells are available."""
        return len(self.available_cells()) > 0
    
    def cell_content(self, x: int, y: int) -> Optional[Tile]:
        """Get tile at specific position."""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.cells[x][y]
        return None
    
    def insert_tile(self, tile: Tile):
        """Place tile at its position."""
        self.cells[tile.x][tile.y] = tile
    
    def remove_tile(self, tile: Tile):
        """Remove tile from grid."""
        self.cells[tile.x][tile.y] = None
    
    def move_tile(self, tile: Tile, new_x: int, new_y: int):
        """Move tile to new position."""
        self.cells[tile.x][tile.y] = None
        tile.x = new_x
        tile.y = new_y
        self.cells[new_x][new_y] = tile
    
    def serialize(self) -> List[List]:
        """Convert grid to nested list representation."""
        return [[tile.value if tile else 0 for tile in row] for row in self.cells]


class Game2048:
    """Main 2048 game logic."""
    
    # Direction vectors: 0=up, 1=right, 2=down, 3=left
    DIRECTIONS = {
        0: (0, -1),  # up
        1: (1, 0),   # right
        2: (0, 1),   # down
        3: (-1, 0)   # left
    }

    DIRECTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        self.grid = Grid(size)
        self.score = 0
        self.over = False
        self.won = False
        
        # Initialize with 2 tiles
        self._add_start_tiles()
    
    def _add_start_tiles(self, count: int = 2):
        """Add initial tiles to start the game."""
        for _ in range(count):
            self._add_random_tile()
    
    def _add_random_tile(self):
        """Add a random tile (90% chance 2, 10% chance 4)."""
        available = self.grid.available_cells()
        if available:
            x, y = random.choice(available)
            value = 2 if random.random() < 0.9 else 4
            tile = Tile(x, y, value)
            self.grid.insert_tile(tile)
    
    def _get_vector(self, direction: int) -> Tuple[int, int]:
        """Get direction vector."""
        return self.DIRECTIONS.get(direction, (0, 0))
    
    def _build_traversals(self, vector: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """Build traversal order based on direction."""
        x_traversal = list(range(self.size))
        y_traversal = list(range(self.size))
        
        # Reverse if moving right or down to avoid overwriting
        if vector[0] == 1:  # moving right
            x_traversal = x_traversal[::-1]
        if vector[1] == 1:  # moving down
            y_traversal = y_traversal[::-1]
        
        return x_traversal, y_traversal
    
    def _find_farthest_position(self, x: int, y: int, vector: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Find farthest position tile can move to in given direction."""
        while True:
            next_x = x + vector[0]
            next_y = y + vector[1]
            
            # Check bounds
            if not (0 <= next_x < self.size and 0 <= next_y < self.size):
                break
            
            # Check if cell is available
            if self.grid.cell_content(next_x, next_y) is not None:
                break
            
            x, y = next_x, next_y
        
        return (x, y), (x + vector[0], y + vector[1])
    
    def _prepare_tiles(self):
        """Clear merged state before move."""
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content(x, y)
                if tile:
                    tile.merged_from = None
    
    def _moves_available(self) -> bool:
        """Check if any moves are available."""
        # Check for empty cells
        if self.grid.cells_available():
            return True
        
        # Check for possible merges
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content(x, y)
                if tile:
                    for direction in range(4):
                        vector = self._get_vector(direction)
                        next_x = x + vector[0]
                        next_y = y + vector[1]
                        
                        other = self.grid.cell_content(next_x, next_y)
                        if other and other.value == tile.value:
                            return True
        
        return False
    
    def can_move(self, direction: int) -> bool:
        """
        Check if a move in given direction is legal (would change state).
        Does not modify game state.
        
        Args:
            direction: 0=up, 1=right, 2=down, 3=left
        
        Returns:
            True if move would result in a state change, False otherwise
        """
        if self.over or self.won:
            return False
        
        vector = self._get_vector(direction)
        
        # Check if any tile can move or merge
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content(x, y)
                
                if tile:
                    # Check if tile can move
                    farthest, next_pos = self._find_farthest_position(x, y, vector)
                    
                    # Can move if position changed
                    if farthest != (x, y):
                        return True
                    
                    # Check if tile can merge
                    other = self.grid.cell_content(next_pos[0], next_pos[1])
                    if other and other.value == tile.value and other.merged_from is None:
                        return True
        
        return False
    
    def move(self, direction: int) -> Tuple[bool, int]:
        """
        Execute move in given direction.
        
        Args:
            direction: 0=up, 1=right, 2=down, 3=left
        
        Returns:
            (moved: bool, score_gained: int)
        """
        if self.over or self.won:
            return False, 0
        
        vector = self._get_vector(direction)
        x_traversal, y_traversal = self._build_traversals(vector)
        
        moved = False
        score_gained = 0
        
        self._prepare_tiles()
        
        # Traverse and move tiles
        for x in x_traversal:
            for y in y_traversal:
                tile = self.grid.cell_content(x, y)
                
                if tile:
                    farthest, next_pos = self._find_farthest_position(x, y, vector)
                    
                    # Check for merge
                    other = self.grid.cell_content(next_pos[0], next_pos[1])
                    
                    if other and other.value == tile.value and other.merged_from is None:
                        # Merge tiles
                        merged_value = tile.value * 2
                        merged_tile = Tile(next_pos[0], next_pos[1], merged_value)
                        merged_tile.merged_from = [tile, other]
                        
                        self.grid.remove_tile(tile)
                        self.grid.remove_tile(other)
                        self.grid.insert_tile(merged_tile)
                        
                        # Update score
                        self.score += merged_value
                        score_gained += merged_value
                        
                        # Check for win
                        if merged_value == 2048:
                            self.won = True
                    else:
                        # Move tile
                        if farthest != (x, y):
                            self.grid.move_tile(tile, farthest[0], farthest[1])
                            moved = True
        
        # Check if position changed
        if moved or score_gained > 0:
            self._add_random_tile()
            
            if not self._moves_available():
                self.over = True
            
            return True, score_gained
        else:
            # Move was invalid - check if game is now stuck
            if not self._moves_available():
                self.over = True
        
        return False, 0
    
    def reset(self):
        """Reset game to initial state."""
        self.grid = Grid(self.size)
        self.score = 0
        self.over = False
        self.won = False
        self._add_start_tiles()
    
    def get_state(self) -> np.ndarray:
        """Get current game state as flat array."""
        state = []
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content(x, y)
                # Use log2 of tile value (0 for empty, up to 16 for 2^16 = 65536)
                if tile:
                    state.append(int(np.log2(tile.value)))
                else:
                    state.append(0)
        return np.array(state, dtype=np.float32)
    
    def get_grid_display(self) -> str:
        """Return ASCII representation of grid for display."""
        lines = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                tile = self.grid.cell_content(x, y)
                if tile:
                    row.append(f"{tile.value:>5}")
                else:
                    row.append("    .")
            lines.append(" ".join(row))
        return "\n".join(lines)
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value on the board."""
        max_val = 0
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content(x, y)
                if tile and tile.value > max_val:
                    max_val = tile.value
        return max_val
    
    def get_empty_count(self) -> int:
        """Get number of empty cells."""
        return len(self.grid.available_cells())
    
    def __str__(self) -> str:
        sb = []
        sb.append("="*30)
        sb.append(f"Score: {self.score}, Max Tile: {self.get_max_tile()}, Empty Cells: {self.get_empty_count()}")
        sb.append("\n")
        sb.append(self.get_grid_display())
        sb.append("\n")
        sb.append(f"Game Over: {'Yes' if self.over else 'No'}, Won: {'Yes' if self.won else 'No'}")
        sb.append("="*30)
        return "\n".join(sb)

def play_game():
    """Interactive game loop for manual play."""
    direction_map = {'w': 0, 'd': 1, 's': 2, 'a': 3}
    
    game = Game2048(seed=42)
    
    while True:
        # Clear screen (works on Windows, Mac, and Linux)
        clear_screen()

        print(game)
        
        if game.won:
            print("\n🎉 恭喜！你達到了 2048！")
            print("(輸入 'c' 繼續遊戲，或輸入其他移動)")
        
        if game.over:
            print("\n💔 遊戲結束！沒有可用的移動。")
            print(f"最終分數: {game.score}")
            break
        
        print("\n移動控制:")
        print("  W - 向上移動", end=' ')
        print("  S - 向下移動", end=' ')
        print("  A - 向左移動", end=' ')
        print("  D - 向右移動")
        print("  Q - 退出遊戲", end=' ')
        print("  R - 重新開始")
        print()
        
        # Get user input
        while True:
            user_input = input("請輸入移動方向: ").strip().lower()
            
            if user_input == 'q':
                print("\n謝謝遊玩！最終分數:", game.score)
                return
            
            if user_input == 'r':
                game.reset()
                break
            
            if user_input in direction_map:
                direction = direction_map[user_input]
                
                # Try to move
                moved, score_gained = game.move(direction)
                
                if not moved:
                    print(f"\n⚠️  無法向{game.DIRECTION_NAMES[direction]}移動！")
                    input("按 Enter 繼續...")
                
                break
            else:
                print("❌ 無效的輸入！請重試。")


if __name__ == "__main__":
    play_game()
