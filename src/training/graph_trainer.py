"""Full-batch transductive node-classification trainer.

Protocol (fixed before any test evaluation):
  * AdamW, constant LR or cosine/plateau scheduler; gradient clipping.
  * model selection: best validation accuracy (ties -> lower validation loss); test evaluated ONCE
    with the selected weights. Test metrics never influence training or selection.
  * early stopping on validation accuracy with `patience`.
"""
from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..utils.metrics import accuracy, count_parameters, macro_f1, peak_memory_mb
from ..utils.seed import rng_state, set_rng_state


def _fwd(model, data, struct):
    return model(data["x"], data["edge_index"], struct=struct)


def evaluate(model, data, struct, amp_dtype=None) -> dict:
    model.eval()
    ctx = torch.autocast(data["x"].device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    with torch.no_grad(), ctx:
        t0 = time.perf_counter(); logits = _fwd(model, data, struct).float(); t_inf = time.perf_counter() - t0
    out = {"inference_seconds": t_inf}
    for split in ("train", "val", "test"):
        m = data[f"{split}_mask"]
        out[f"{split}_loss"] = float(F.cross_entropy(logits[m], data["y"][m]))
        out[f"{split}_acc"] = accuracy(logits[m], data["y"][m])
        out[f"{split}_macro_f1"] = macro_f1(logits[m], data["y"][m], data["num_classes"])
    return out


def train_node_classifier(model, data: dict, lr: float = 5e-3, weight_decay: float = 5e-4, epochs: int = 300,
                          patience: int = 100, grad_clip: float = 1.0, scheduler: str = "none",
                          amp: Optional[str] = None, checkpoint_path: Optional[str] = None,
                          checkpoint_every: int = 0, resume: bool = False, max_epochs_this_call: Optional[int] = None,
                          verbose: bool = False) -> dict:
    device = data["x"].device
    struct = data.get("struct")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[amp]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=25)
    scaler = torch.amp.GradScaler(enabled=(amp == "fp16" and device.type == "cuda"))
    state = {"epoch": 0, "best_val": -1.0, "best_val_loss": float("inf"), "best_epoch": -1, "bad": 0,
             "best_state": None, "history": []}
    if resume and checkpoint_path and Path(checkpoint_path).exists():
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck.get("scheduler"): sched.load_state_dict(ck["scheduler"])
        state = ck["state"]; set_rng_state(ck["rng"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter(); n_run = 0
    while state["epoch"] < epochs:
        if max_epochs_this_call is not None and n_run >= max_epochs_this_call:
            break
        model.train()
        ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
            logits = _fwd(model, data, struct)
            loss = F.cross_entropy(logits[data["train_mask"]].float(), data["y"][data["train_mask"]])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt); scaler.update()
        ev = evaluate(model, data, struct, amp_dtype)
        if sched is not None:
            sched.step(ev["val_acc"]) if scheduler == "plateau" else sched.step()
        state["history"].append({"epoch": state["epoch"], "train_loss": float(loss), "val_acc": ev["val_acc"], "val_loss": ev["val_loss"]})
        better = ev["val_acc"] > state["best_val"] or (ev["val_acc"] == state["best_val"] and ev["val_loss"] < state["best_val_loss"])
        if better:
            state.update(best_val=ev["val_acc"], best_val_loss=ev["val_loss"], best_epoch=state["epoch"], bad=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad"] += 1
        state["epoch"] += 1; n_run += 1
        if verbose and state["epoch"] % 50 == 0:
            print(state["epoch"], float(loss), ev["val_acc"])
        if checkpoint_path and checkpoint_every and state["epoch"] % checkpoint_every == 0:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict() if sched is not None else None, "state": state, "rng": rng_state()},
                       checkpoint_path)
        if state["bad"] >= patience:
            state["epoch"] = epochs  # stop
    train_seconds = time.perf_counter() - t_start
    finished = state["epoch"] >= epochs
    if not finished:
        return {"finished": False, "state": state}
    model.load_state_dict(state["best_state"])
    final = evaluate(model, data, struct, amp_dtype)
    return {"finished": True, "best_epoch": state["best_epoch"], "epochs_run": len(state["history"]),
            "train_seconds": train_seconds, "seconds_per_epoch": train_seconds / max(n_run, 1),
            "num_parameters": count_parameters(model), **final, **peak_memory_mb(device),
            "history": state["history"]}
