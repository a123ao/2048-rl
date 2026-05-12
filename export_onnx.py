"""
Export trained DQN agents to ONNX for frontend inference.

Usage:
    uv run export_onnx.py
"""

import os
import torch
import onnx
from onnx.external_data_helper import load_external_data_for_model

from src.agents import DQNNetwork

CHECKPOINT_DIR = "checkpoints"
ONNX_DIR = "2048/onnx"

MODELS_CONFIG = [
    "baseline",
    "partial",
    "full",
    "best",
]

os.makedirs(ONNX_DIR, exist_ok=True)
dummy_input = torch.zeros(1, 256, dtype=torch.float32)

for key in MODELS_CONFIG:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{key}_agent.pth")
    onnx_path = os.path.join(ONNX_DIR, f"{key}_agent.onnx")

    if not os.path.exists(ckpt_path):
        print(f"⏭  {key}: checkpoint not found, skipping")
        continue

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = DQNNetwork(state_size=256, hidden_size=256, action_size=4)
    model.load_state_dict(checkpoint["q_network"])
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["q_values"],
        dynamic_axes={"state": {0: "batch_size"}, "q_values": {0: "batch_size"}},
    )

    # Inline any external data so the .onnx is self-contained
    onnx_model = onnx.load(onnx_path)
    try:
        load_external_data_for_model(onnx_model, os.path.dirname(onnx_path))
    except Exception:
        pass
    onnx.save(onnx_model, onnx_path)

    print(f"✅ {onnx_path} exported successfully")

# Clean up stray .data files
for f in os.listdir(ONNX_DIR):
    if f.endswith(".data"):
        os.remove(os.path.join(ONNX_DIR, f))

print("\n✅ All ONNX models exported to onnx/")
print("   Input  : Float32[batch, 256]  – one-hot board (16 cells × 16 log2 values)")
print("   Output : Float32[batch, 4]    – Q-values for [up, down, left, right]")
