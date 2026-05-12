"""
Final Demo & Summary Report for 2048 RL Project
"""

from time import sleep
from src.environment import Game2048Env
from src.agents import BaseAgent, RandomAgent, BaselineAgent, PartialRewardAgent, FullRewardAgent
from src.utils import clear_screen

def run_agent(agent: BaseAgent, env: Game2048Env):
    """Run a brief demo of the agent playing the game."""
    state = env.reset()

    print('Model: ', agent.__class__.__name__)
    print(env.game)

    current_step = 1
    while True:
        legal_actions = env.get_legal_actions()
        action = agent.select_action(state, legal_actions=legal_actions)
        observation, reward, done, info = env.step(action)

        clear_screen()

        print('Model: ', agent.__class__.__name__)
        print(env.game)
        print(f"\nStep: {current_step}, Action: {env.game.DIRECTION_NAMES[action]}, Reward: {reward}, Done: {done}")

        if done:
            print("Game over!")
            break

        current_step += 1

        sleep(0.1)

if __name__ == "__main__":
    env = Game2048Env()

    # random_agent = RandomAgent()
    # run_agent(random_agent, env)

    # baseline_agent = BaselineAgent(device='cpu')
    # baseline_agent.load("./checkpoints/baseline_agent.pth")
    # run_agent(baseline_agent, env)

    partial_agent = PartialRewardAgent(device='cpu')
    partial_agent.load("./checkpoints/partial_agent.pth")
    run_agent(partial_agent, env)

    # full_agent = FullRewardAgent(device='cpu')
    # full_agent.load("./checkpoints/full_agent.pth")
    # run_agent(full_agent, env)