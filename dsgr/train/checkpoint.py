import json
import os
from typing import Dict, Any

import torch


def get_trainable_state_dict(model_v):
    trainable_names = {n for n, p in model_v.named_parameters() if p.requires_grad}
    model_state = model_v.state_dict()
    return {k: v.detach().cpu() for k, v in model_state.items() if k in trainable_names}


def load_trainable_state_dict(model_v, state_dict_path):
    if not os.path.exists(state_dict_path):
        return False
    state = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    model_v.load_state_dict(state, strict=False)
    return True


def save_runtime_snapshot(run_dir: str, runtime_config: Dict[str, Any]):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(runtime_config, f, ensure_ascii=False, indent=2)
    return path

