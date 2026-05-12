# """
# Utility functions for 2048 RL project.
# """

import json
import os
# import torch
# from environment import Game2048Env
# from agents import BaselineAgent, RandomAgent


def clear_screen():
    """Clear terminal screen (cross-platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')


# def load_agent_and_logs(agent_type: str, checkpoint_dir: str = "checkpoints"):
#     """
#     Load a trained agent and its logs.
    
#     Args:
#         agent_type: "random" or "baseline"
#         checkpoint_dir: Directory containing checkpoints
    
#     Returns:
#         (agent, logger_data)
#     """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     model_path = os.path.join(checkpoint_dir, f"{agent_type}_agent.pth")
#     log_path = os.path.join(checkpoint_dir, f"{agent_type}_logs.json")
    
#     # Load agent
#     if agent_type.lower() == "random":
#         agent = RandomAgent()
#     elif agent_type.lower() == "baseline":
#         agent = BaselineAgent(device=device)
#         if os.path.exists(model_path):
#             agent.load(model_path)
#     else:
#         raise ValueError(f"Unknown agent type: {agent_type}")
    
#     # Load logs
#     logs = None
#     if os.path.exists(log_path):
#         with open(log_path, 'r') as f:
#             logs = json.load(f)
    
#     return agent, logs, device


# def play_game(agent, env: Game2048Env, max_steps: int = 10000, display: bool = True):
#     """
#     Play a game with the given agent.
    
#     Args:
#         agent: The agent to play with
#         env: The game environment
#         max_steps: Maximum steps before terminating
#         display: Whether to display game state
    
#     Returns:
#         Game statistics and move history
#     """
#     state = env.reset()
#     done = False
#     moves = []
#     move_count = 0
    
#     if display:
#         clear_screen()
#         print("\n" + "="*40)
#         print("GAME START")
#         print("="*40)
#         print(env.get_state_display())
#         print(f"Score: {env.game.score}\n")
    
#     while not done and move_count < max_steps:
#         # Get legal actions
#         legal_actions = env.get_legal_actions()
        
#         if not legal_actions:  # No legal moves available
#             break
        
#         # Select action (preferring legal moves)
#         action = agent.select_action(state, epsilon=0.0, training=False, legal_actions=legal_actions)
#         action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        
#         # Store previous state for display
#         prev_score = env.game.score
#         prev_state_display = env.get_state_display()
        
#         # Execute action
#         next_state, reward, done, info = env.step(action)
        
#         move_data = {
#             'move': move_count + 1,
#             'action': action_names[action],
#             'reward': reward,
#             'score_gained': reward,
#             'total_score': info['score'],
#             'max_tile': info['max_tile'],
#             'empty_cells': info['empty_cells'],
#             'game_over': done
#         }
#         moves.append(move_data)
        
#         # Display only every 50 moves or when score gained
#         if display and (reward > 0 or move_count % 50 == 0 or done):
#             clear_screen()
#             print(f"Move {move_count + 1}: {action_names[action]}")
#             if reward > 0:
#                 print(f"  Merged! Gained {int(reward)} points")
#             print(env.get_state_display())
#             print(f"Score: {info['score']}, Max Tile: {info['max_tile']}, Empty: {info['empty_cells']}")
#             print()
        
#         state = next_state
#         move_count += 1
    
#     stats = env.get_episode_stats()
    
#     if display:
#         print("="*40)
#         if done:
#             print("GAME OVER!")
#         else:
#             print("MAX STEPS REACHED")
#         print("="*40)
#         print(f"Final Score: {stats['score']}")
#         print(f"Max Tile: {stats['max_tile']}")
#         print(f"Total Moves: {move_count}")
    
#     return {
#         'score': stats['score'],
#         'max_tile': stats['max_tile'],
#         'moves': move_count,
#         'game_over': done,
#         'moves_history': moves
#     }


# def compare_agents(checkpoint_dir: str = "checkpoints"):
#     """
#     Compare Random and Baseline agents on the same game.
    
#     Args:
#         checkpoint_dir: Directory containing checkpoints
#     """
#     print("\n" + "="*60)
#     print("AGENT COMPARISON: FINAL GAMEPLAY DEMONSTRATION")
#     print("="*60)
    
#     # Load agents
#     random_agent, _, device = load_agent_and_logs("random", checkpoint_dir)
#     baseline_agent, _, _ = load_agent_and_logs("baseline", checkpoint_dir)
    
#     env = Game2048Env()
    
#     # Play with Random Agent
#     print("\n\n>>> RANDOM AGENT <<<")
#     random_stats = play_game(random_agent, env, max_steps=10000, display=True)
    
#     # Play with Baseline Agent
#     print("\n\n>>> BASELINE AGENT <<<")
#     baseline_stats = play_game(baseline_agent, env, max_steps=10000, display=True)
    
#     # Summary
#     print("\n" + "="*60)
#     print("COMPARISON SUMMARY")
#     print("="*60)
#     print(f"{'Metric':<20} {'Random':<20} {'Baseline':<20}")
#     print("-"*60)
#     print(f"{'Score':<20} {random_stats['score']:<20.0f} {baseline_stats['score']:<20.0f}")
#     print(f"{'Max Tile':<20} {random_stats['max_tile']:<20.0f} {baseline_stats['max_tile']:<20.0f}")
#     print(f"{'Moves':<20} {random_stats['moves']:<20} {baseline_stats['moves']:<20}")
#     print("="*60)


# if __name__ == "__main__":
#     compare_agents()
