from __future__ import annotations
import time
from typing import Any
import numpy as np
import torch
from sklearn.metrics import f1_score
from . import label_mapping as proxy32



EXPECTED_CLASSES = 32


BMVP_SEEDS = list(range(400, 410))


FOMO_SEEDS = list(range(600, 610))


EPOCHS = 10


BATCH = 512


LR = 1e-3


WEIGHT_DECAY = 1e-2


BMVP_PCA_DIM = 2_048


THRESHOLD = 0.5


def clone_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        result = torch.sigmoid(
            model(torch.tensor(x, dtype=torch.float32).to(device))
        ).cpu().numpy()
    return result.astype(np.float32, copy=False)


def train_bmvp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    trainable: np.ndarray,
    device: torch.device,
    started: float,
) -> tuple[list[dict[str, torch.Tensor]], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    positives = y_train.sum(0)
    pos_weight = np.where(
        trainable,
        np.minimum((len(y_train) - positives) / (positives + 1), 100),
        0,
    )
    xt = torch.tensor(x_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.float32).to(device)
    pw = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    trainable_t = torch.tensor(trainable, dtype=torch.bool).to(device)
    by_seed: list[np.ndarray] = []
    states: list[dict[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    ensemble = np.zeros((len(x_val), EXPECTED_CLASSES), dtype=np.float32)
    for seed in BMVP_SEEDS:
        torch.manual_seed(seed)
        head = proxy32.MLPHead(BMVP_PCA_DIM, EXPECTED_CLASSES).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=pw[trainable_t]
        )
        best_f1 = -1.0
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(EPOCHS):
            head.train()
            for start in range(0, len(xt), BATCH):
                xb = xt[start : start + BATCH]
                yb = yt[start : start + BATCH]
                optimizer.zero_grad()
                logits = head(xb)
                loss = criterion(logits[:, trainable_t], yb[:, trainable_t])
                loss.backward()
                optimizer.step()
            scheduler.step()
            probability = predict(head, x_val, device)
            score = float(
                f1_score(
                    y_val[:, trainable],
                    probability[:, trainable] >= THRESHOLD,
                    average="macro",
                    zero_division=0,
                )
            )
            if score > best_f1:
                best_f1 = score
                best_epoch = epoch + 1
                best_state = clone_cpu_state(head)
        if best_state is None:
            raise RuntimeError(f"BMVP seed {seed} did not select a state")
        head.load_state_dict(best_state)
        probability = predict(head, x_val, device)
        ensemble += probability / len(BMVP_SEEDS)
        by_seed.append(probability)
        states.append(best_state)
        records.append(
            {"seed": seed, "best_epoch": best_epoch, "best_val_macro_f1": best_f1}
        )
        print(
            f"[candidate] BMVP seed={seed} epoch={best_epoch} "
            f"val={best_f1:.6f} elapsed_min={(time.time()-started)/60:.1f}",
            flush=True,
        )
        del head, optimizer, scheduler
    del xt, yt, pw, trainable_t
    torch.cuda.empty_cache()
    return states, np.stack(by_seed), ensemble, records


def train_fomo(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    trainable: np.ndarray,
    device: torch.device,
    started: float,
) -> tuple[list[dict[str, torch.Tensor]], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    positives = y_train.sum(0)
    pos_weight = np.where(
        trainable,
        np.minimum((len(y_train) - positives) / (positives + 1), 100),
        0,
    )
    xt = torch.tensor(x_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.float32).to(device)
    pw = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    trainable_t = torch.tensor(trainable, dtype=torch.bool).to(device)
    by_seed: list[np.ndarray] = []
    states: list[dict[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    ensemble = np.zeros((len(x_val), EXPECTED_CLASSES), dtype=np.float32)
    for seed in FOMO_SEEDS:
        torch.manual_seed(seed)
        head = proxy32.MLPHead(x_train.shape[1], EXPECTED_CLASSES).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=pw[trainable_t]
        )
        for _ in range(EPOCHS):
            head.train()
            for start in range(0, len(xt), BATCH):
                xb = xt[start : start + BATCH]
                yb = yt[start : start + BATCH]
                optimizer.zero_grad()
                loss = criterion(head(xb)[:, trainable_t], yb[:, trainable_t])
                loss.backward()
                optimizer.step()
            scheduler.step()
        probability = predict(head, x_val, device)
        ensemble += probability / len(FOMO_SEEDS)
        by_seed.append(probability)
        states.append(clone_cpu_state(head))
        records.append({"seed": seed, "epochs": EPOCHS})
        print(
            f"[candidate] FOMO seed={seed} epoch={EPOCHS} "
            f"elapsed_min={(time.time()-started)/60:.1f}",
            flush=True,
        )
        del head, optimizer, scheduler
    del xt, yt, pw, trainable_t
    torch.cuda.empty_cache()
    return states, np.stack(by_seed), ensemble, records
