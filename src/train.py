import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.data import get_base, get_mixed_dataset
from src.utils.naming import model_location, pretrained_model_location
from src.utils.load import create_model, load_model
from src.utils.device import (
    get_device, seed_everything, seed_worker, loader_generator, amp_dtype, autocast,
)

from src.config import Config
cfg = Config()


def evaluate(model, test_loader, criterion=None, device=None, dtype=None):
    device = device or get_device()
    # Same loss as training (it used label_smoothing=0.01 here before), so the
    # printed train and validation losses are comparable.
    criterion = criterion or nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.inference_mode(), autocast(device, dtype):
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits.float(), y)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def _loader(ds, batch_size, shuffle, device):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=loader_generator(cfg.seed),
    )


def _train(model, train_ds, test_ds, epochs=cfg.epochs, save_every=1,
           precision="auto", compile_model=False):
    print("Launching training")
    seed_everything(cfg.seed)

    device = get_device()
    dtype = amp_dtype(device, precision)
    print(f"device={device} autocast={dtype}")

    train_loader = _loader(train_ds, cfg.batch_size, True, device)
    # No gradients at evaluation time, so a much bigger batch fits.
    test_loader = _loader(test_ds, cfg.eval_batch_size, False, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # The fused kernel does the whole AdamW update in one CUDA launch.
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
                            fused=(device == "cuda"))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss scaling is only needed for fp16; bf16 has fp32's range.
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # torch.compile needs a GPU of compute capability 7.0+ (Volta or newer),
    # so it is opt-in: it does not work on Pascal cards such as the 1080 Ti.
    step_model = torch.compile(model) if compile_model else model

    for epoch in range(1, epochs + 1):
        step_model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", dynamic_ncols=True)
        running_loss = torch.zeros((), device=device)
        n = 0

        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device, dtype):
                logits = step_model(x)
                loss = criterion(logits.float(), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate on the GPU and only sync for the progress bar every
            # 50 steps; calling loss.item() every step stalls the GPU.
            running_loss += loss.detach() * y.size(0)
            n += y.size(0)
            if step % 50 == 0:
                pbar.set_postfix(loss=running_loss.item() / n, lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        val_loss, val_acc = evaluate(step_model, test_loader, criterion, device, dtype)
        print(f"  train_loss={running_loss.item() / n:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        if save_every is not None and epoch % save_every == 0:
            path = model_location(epoch)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)


def _launch_train(train, test, model=None, epochs=cfg.epochs, save_every=cfg.save_every, **kwargs):
    if model is None:
        print(f"Loading {cfg.model} architecture")
        seed_everything(cfg.seed)  # same initial weights every run
        model = create_model()
        model.to(get_device())

    _train(model, train, test, epochs, save_every=save_every, **kwargs)
    return model


def pretrain(**kwargs):
    base_train, base_test = get_base()

    model = _launch_train(base_train, base_test, save_every=None, **kwargs)
    path = pretrained_model_location()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

    return model


def train(use_parquet=False, imagenet_root=None, **kwargs):
    mix_train, mix_test = get_mixed_dataset(
        use_parquet=use_parquet,
        imagenet_root=imagenet_root,
    )

    model = _launch_train(mix_train, mix_test, model=load_model(), **kwargs)
    return model
