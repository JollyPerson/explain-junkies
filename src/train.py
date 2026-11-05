import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm

from src.data import get_base, get_mixed_dataset
from src.utils.naming import model_location, pretrained_model_location
from src.utils.load import load_model

from src.config import Config
cfg = Config()


def evaluate(model, test_loader, criterion=nn.CrossEntropyLoss(label_smoothing=0.01), device='cuda'):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total



def _train(model, train_loader: DataLoader, test_loader: DataLoader, epochs = cfg.epochs, save_every=1):
    print("Launching training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", dynamic_ncols=True)
        running_loss = 0.0
        n = 0

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            n += y.size(0)
            pbar.set_postfix(loss=running_loss / n, lr=optimizer.param_groups[0]["lr"])

            # skip for tests
            # break

        scheduler.step()
        val_loss, val_acc = evaluate(model, test_loader)
        print(f"  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        if save_every is not None and epoch % save_every == 0:
            torch.save(model.state_dict(), model_location(epoch))


def _launch_train(train, test, model=None, epochs=cfg.epochs, save_every=cfg.save_every):
    print("Begin training task")
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    if model is None:
        print(f"Loading {cfg.model} architecture")
        model = timm.create_model(cfg.model, pretrained=False, num_classes=cfg.num_classes)
        
        print(f"Sending model to {device}")
        model.to(device)
    
    _train(model, train, test, epochs, save_every=save_every)
    return model


def pretrain():
    base_train, base_test = get_base()
    
    model = _launch_train(base_train, base_test, epochs=12, save_every=None)
    torch.save(model.state_dict(), pretrained_model_location())

    return model


def train():
    mix_train, mix_test = get_mixed_dataset()

    model = _launch_train(mix_train, mix_test, model=load_model())
    return model

