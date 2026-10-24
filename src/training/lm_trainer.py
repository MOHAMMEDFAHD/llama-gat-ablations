from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch

from ..data.synthetic_lm import progression_batch, progression_labels
from ..llama.config import LlamaConfig
from ..llama.model import LlamaForCausalLM
from ..utils.seed import rng_state, set_rng_state, set_seed


def save_checkpoint(path: str | Path, model, optimizer, step: int, extra: Optional[dict] = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "config": model.config.to_dict(), "rng": rng_state(), "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, model=None, optimizer=None, restore_rng: bool = True) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if model is None:
        model = LlamaForCausalLM(LlamaConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if restore_rng:
        set_rng_state(ck["rng"])
    ck["model_obj"] = model
    return ck


def train_progression_lm(config: LlamaConfig, steps: int = 300, batch: int = 32, seq_len: int = 24, lr: float = 3e-3,
                         seed: int = 0, val_every: int = 50, log_path: Optional[str] = None,
                         grad_clip: float = 1.0) -> dict:
    set_seed(seed)
    model = LlamaForCausalLM(config)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * s / steps)).item()))
    g_val = torch.Generator().manual_seed(10_000 + seed)
    val_ids = progression_batch(256, seq_len, config.vocab_size, generator=g_val)
    hist = {"train_loss": [], "val_loss": [], "step": []}
    t0 = time.time()
    for step in range(steps):
        model.train()
        ids = progression_batch(batch, seq_len, config.vocab_size)
        out = model(ids, labels=progression_labels(ids))
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); sched.step()
        hist["train_loss"].append(float(out.loss))
        if (step + 1) % val_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = float(model(val_ids, labels=progression_labels(val_ids)).loss)
            hist["val_loss"].append(vl); hist["step"].append(step + 1)
    hist["seconds"] = time.time() - t0
    hist["num_parameters"] = model.num_parameters()
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(hist, indent=1))
    return {"model": model, "optimizer": opt, "history": hist}
