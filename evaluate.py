"""
Evaluation script for 2048 RL agents.
Compares Random, Baseline, Partial, and Full reward shaping agents
on the three metrics defined in project.md:
  1. Average Score
  2. Max Tile reached
  3. Training Curve
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import torch

from src.environment import Game2048Env
from src.agents import RandomAgent, BaselineAgent, PartialRewardAgent, FullRewardAgent, BestRewardAgent


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
EVAL_EPISODES  = 200   # greedy evaluation games per agent

AGENTS_CONFIG = [
    {
        "key":   "random",
        "label": "Random",
        "color": "#9E9E9E",
        "marker": "o",
        "has_model": False,
    },
    {
        "key":   "baseline",
        "label": "Baseline\n$R = r_{score}$",
        "color": "#2196F3",
        "marker": "s",
        "has_model": True,
        "cls": BaselineAgent,
        "kwargs": {"state_size": 256, "hidden_size": 256},
    },
    {
        "key":   "partial",
        "label": "Partial\n$R = r_{score} + \\alpha r_{empty}$",
        "color": "#FF9800",
        "marker": "^",
        "has_model": True,
        "cls": PartialRewardAgent,
        "kwargs": {"state_size": 256, "hidden_size": 256},
    },
    {
        "key":   "full",
        "label": "Full\n$R = r_{score} + \\alpha r_{empty}$\n$+ \\beta r_{corner} + \\gamma r_{mono}$",
        "color": "#4CAF50",
        "marker": "D",
        "has_model": True,
        "cls": FullRewardAgent,
        "kwargs": {"state_size": 256, "hidden_size": 256},
    },
    {
        "key":   "best",
        "label": "Best\n$R = r_{score} + \\alpha r_{empty}$\n$+ \\beta r_{snake} + \\gamma r_{smooth}$",
        "color": "#E91E63",
        "marker": "*",
        "has_model": True,
        "cls": BestRewardAgent,
        "kwargs": {"state_size": 256, "hidden_size": 256},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_logs(key: str) -> dict | None:
    """Load training logs JSON for a given agent key."""
    path = os.path.join(CHECKPOINT_DIR, f"{key}_logs.json")
    if not os.path.exists(path):
        print(f"  [WARN] No logs found for '{key}' at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_agent(cfg: dict, device: str):
    """Instantiate and load checkpoint for a DQN agent."""
    agent = cfg["cls"](**cfg["kwargs"], device=device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{cfg['key']}_agent.pth")
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
    else:
        print(f"  [WARN] No checkpoint found for '{cfg['key']}' – using untrained weights")
    agent.q_network.eval()
    return agent


def evaluate_agent(agent, num_episodes: int = EVAL_EPISODES) -> dict:
    """
    Run greedy evaluation (epsilon=0, no training) for num_episodes games.
    Returns dict with scores, max_tiles lists and summary stats.
    """
    env = Game2048Env()
    scores, max_tiles = [], []

    for _ in tqdm(range(num_episodes), desc=f"  Evaluating", leave=False):
        state = env.reset()
        done  = False
        while not done:
            legal = env.get_legal_actions()
            action = agent.select_action(state, epsilon=0.0, training=False,
                                         legal_actions=legal)
            state, _, done, _ = env.step(action)
        stats = env.get_episode_stats()
        scores.append(stats["score"])
        max_tiles.append(stats["max_tile"])

    return {
        "scores":    scores,
        "max_tiles": max_tiles,
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "best_score": int(np.max(scores)),
        "avg_max_tile": float(np.mean(max_tiles)),
        "best_max_tile": int(np.max(max_tiles)),
    }


def evaluate_random(num_episodes: int = EVAL_EPISODES) -> dict:
    """Evaluate the random agent (no model needed)."""
    agent = RandomAgent()
    env   = Game2048Env()
    scores, max_tiles = [], []

    for _ in tqdm(range(num_episodes), desc="  Evaluating", leave=False):
        state = env.reset()
        done  = False
        while not done:
            legal  = env.get_legal_actions()
            action = agent.select_action(state, legal_actions=legal)
            state, _, done, _ = env.step(action)
        stats = env.get_episode_stats()
        scores.append(stats["score"])
        max_tiles.append(stats["max_tile"])

    return {
        "scores":    scores,
        "max_tiles": max_tiles,
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "best_score": int(np.max(scores)),
        "avg_max_tile": float(np.mean(max_tiles)),
        "best_max_tile": int(np.max(max_tiles)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – Training Curves (from logs)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(all_logs: dict, save_dir: str):
    """Plot average score and average max tile over training for all agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves – Average per 100-Episode Window",
                 fontsize=14, fontweight="bold")

    for cfg in AGENTS_CONFIG:
        logs = all_logs.get(cfg["key"])
        if logs is None or not logs.get("avg_scores"):
            continue
        n = len(logs["avg_scores"])
        # x = episode number at end of each 100-ep window
        x = [(i + 1) * 100 for i in range(n)]

        axes[0].plot(x, logs["avg_scores"],
                     label=cfg["label"].split("\n")[0],
                     color=cfg["color"], marker=cfg["marker"],
                     markersize=5, linewidth=2)
        axes[1].plot(x, logs["avg_max_tiles"],
                     label=cfg["label"].split("\n")[0],
                     color=cfg["color"], marker=cfg["marker"],
                     markersize=5, linewidth=2)

    for ax, ylabel, title in zip(
        axes,
        ["Average Score", "Average Max Tile"],
        ["Average Score Over Training", "Average Max Tile Over Training"],
    ):
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1b – Cumulative Reward
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_rewards(all_logs: dict, save_dir: str):
    """
    Plot cumulative reward over training episodes for each agent.

    Two sub-plots:
      Left  – Cumulative sum of per-episode rewards (total reward earned so far)
      Right – Smoothed average reward per episode (100-ep window)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cumulative & Average Reward During Training",
                 fontsize=14, fontweight="bold")

    for cfg in AGENTS_CONFIG:
        logs = all_logs.get(cfg["key"])
        if logs is None:
            continue
        ep_rewards = logs.get("episode_rewards")
        avg_rewards = logs.get("avg_rewards")
        label = cfg["label"].split("\n")[0]

        # Left: cumulative reward
        if ep_rewards:
            cumulative = np.cumsum(ep_rewards)
            episodes = np.arange(1, len(cumulative) + 1)
            axes[0].plot(episodes, cumulative,
                         label=label, color=cfg["color"], linewidth=1.5)

        # Right: smoothed avg reward per 100-ep window
        if avg_rewards:
            n = len(avg_rewards)
            x = [(i + 1) * 100 for i in range(n)]
            axes[1].plot(x, avg_rewards,
                         label=label, color=cfg["color"],
                         marker=cfg["marker"], markersize=5, linewidth=2)

    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Cumulative Reward", fontsize=11)
    axes[0].set_title("Cumulative Reward Over Training", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].set_ylabel("Avg Reward (100-ep window)", fontsize=11)
    axes[1].set_title("Smoothed Average Reward Per Episode", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "cumulative_rewards.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – Evaluation Metrics Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_evaluation_metrics(eval_results: dict, save_dir: str):
    """Bar charts comparing average score, best score, and max tile distribution."""
    labels = [cfg["label"] for cfg in AGENTS_CONFIG if cfg["key"] in eval_results]
    keys   = [cfg["key"]   for cfg in AGENTS_CONFIG if cfg["key"] in eval_results]
    colors = [cfg["color"] for cfg in AGENTS_CONFIG if cfg["key"] in eval_results]

    avg_scores    = [eval_results[k]["avg_score"]    for k in keys]
    std_scores    = [eval_results[k]["std_score"]    for k in keys]
    best_scores   = [eval_results[k]["best_score"]   for k in keys]
    avg_max_tiles = [eval_results[k]["avg_max_tile"] for k in keys]
    best_max_tiles= [eval_results[k]["best_max_tile"]for k in keys]

    x = np.arange(len(keys))
    bar_w = 0.55

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Evaluation Metrics – {EVAL_EPISODES} Greedy Episodes per Agent",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── (1) Average Score with std error bar ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, avg_scores, bar_w, yerr=std_scores, capsize=5,
                   color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.split("\n")[0] for l in labels], fontsize=9)
    ax1.set_ylabel("Score")
    ax1.set_title("Average Score (± std)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, avg_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    # ── (2) Best Score ──
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x, best_scores, bar_w,
                   color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.split("\n")[0] for l in labels], fontsize=9)
    ax2.set_ylabel("Score")
    ax2.set_title("Best Score", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, best_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 f"{val}", ha="center", va="bottom", fontsize=8)

    # ── (3) Average Max Tile ──
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(x, avg_max_tiles, bar_w,
                   color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.split("\n")[0] for l in labels], fontsize=9)
    ax3.set_ylabel("Tile Value")
    ax3.set_title("Average Max Tile", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, avg_max_tiles):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # ── (4) Score Distribution – Box Plot ──
    ax4 = fig.add_subplot(gs[1, :2])
    score_data = [eval_results[k]["scores"] for k in keys]
    short_labels = [l.split("\n")[0] for l in labels]
    bp = ax4.boxplot(score_data, patch_artist=True, labels=short_labels,
                     medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax4.set_ylabel("Score")
    ax4.set_title("Score Distribution", fontsize=11)
    ax4.grid(axis="y", alpha=0.3)

    # ── (5) Max Tile Distribution – stacked bar (% of games reaching each tile) ──
    ax5 = fig.add_subplot(gs[1, 2])
    tile_thresholds = [64, 128, 256, 512, 1024, 2048]
    tile_colors     = ["#CFD8DC", "#90A4AE", "#546E7A", "#FF8F00", "#E65100", "#B71C1C"]
    bottom = np.zeros(len(keys))
    for tile, tc in zip(tile_thresholds, tile_colors):
        pct = np.array([
            100.0 * sum(t >= tile for t in eval_results[k]["max_tiles"]) / len(eval_results[k]["max_tiles"])
            for k in keys
        ])
        bars = ax5.bar(x, pct, bar_w, bottom=bottom, label=f"≥{tile}", color=tc,
                       edgecolor="white", linewidth=0.5)
        # label segments that are large enough
        for bar, p in zip(bars, pct):
            if p > 5:
                ax5.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_y() + bar.get_height() / 2,
                         f"{p:.0f}%", ha="center", va="center", fontsize=7,
                         color="white" if p > 15 else "black")
        bottom += pct
    ax5.set_xticks(x)
    ax5.set_xticklabels(short_labels, fontsize=9)
    ax5.set_ylabel("% of Games")
    ax5.set_title("Max Tile Reached (%)", fontsize=11)
    ax5.legend(fontsize=7, loc="upper right", ncol=2)
    ax5.set_ylim(0, 110)
    ax5.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, "evaluation_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(eval_results: dict):
    """Print a formatted comparison table."""
    header = f"{'Agent':<12} {'Avg Score':>12} {'Std Score':>12} {'Best Score':>12} {'Avg MaxTile':>13} {'Best MaxTile':>13}"
    sep    = "─" * len(header)
    print("\n" + "=" * len(header))
    print("EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    for cfg in AGENTS_CONFIG:
        k = cfg["key"]
        if k not in eval_results:
            continue
        r = eval_results[k]
        name = cfg["label"].split("\n")[0]
        print(f"{name:<12} {r['avg_score']:>12.1f} {r['std_score']:>12.1f} "
              f"{r['best_score']:>12} {r['avg_max_tile']:>13.1f} {r['best_max_tile']:>13}")

    print("=" * len(header))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Evaluation episodes per agent: {EVAL_EPISODES}\n")

    # ── Load training logs ──────────────────────────────────────────────────
    print("Loading training logs...")
    all_logs = {cfg["key"]: load_logs(cfg["key"]) for cfg in AGENTS_CONFIG}

    # ── Plot Training Curves ────────────────────────────────────────────────
    print("\nPlotting training curves...")
    plot_training_curves(all_logs, RESULTS_DIR)

    # ── Plot Cumulative Rewards ─────────────────────────────────────────────
    print("Plotting cumulative rewards...")
    plot_cumulative_rewards(all_logs, RESULTS_DIR)

    # ── Run Greedy Evaluation ───────────────────────────────────────────────
    eval_results = {}

    for cfg in AGENTS_CONFIG:
        key = cfg["key"]
        print(f"\nEvaluating: {cfg['label'].split(chr(10))[0]}")

        if not cfg["has_model"]:
            eval_results[key] = evaluate_random()
        else:
            agent = load_agent(cfg, device)
            eval_results[key] = evaluate_agent(agent)

    # ── Plot Evaluation Metrics ─────────────────────────────────────────────
    print("\nPlotting evaluation metrics...")
    plot_evaluation_metrics(eval_results, RESULTS_DIR)

    # ── Save eval results to JSON ───────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "eval_results.json")
    serializable = {
        k: {kk: vv for kk, vv in v.items()} for k, v in eval_results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {out_path}")

    # ── Print Summary ───────────────────────────────────────────────────────
    print_summary(eval_results)


if __name__ == "__main__":
    main()
