import argparse
import yaml
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import time

from dataset import EOSARChangeDataset
from model import build_model, DiceBCELoss, compute_metrics


def get_transforms(image_size):
    train_tf = A.Compose([
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.CenterCrop(image_size, image_size),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    ds = load_dataset(cfg['data']['dataset'])
    train_tf, val_tf = get_transforms(cfg['data']['image_size'])

    train_ds = EOSARChangeDataset(
        ds['train'],
        pre_start    = cfg['data']['train_pre_start'],
        post_start   = cfg['data']['train_post_start'],
        target_start = cfg['data']['train_target_start'],
        n_scenes     = cfg['data']['n_train'],
        transform    = train_tf,
    )
    val_ds = EOSARChangeDataset(
        ds['validation'],
        pre_start    = cfg['data']['val_pre_start'],
        post_start   = cfg['data']['val_post_start'],
        target_start = cfg['data']['val_target_start'],
        n_scenes     = cfg['data']['n_val'],
        transform    = val_tf,
    )

    train_loader = DataLoader(train_ds,
                              batch_size  = cfg['data']['batch_size'],
                              shuffle     = True,
                              num_workers = cfg['data']['num_workers'])
    val_loader   = DataLoader(val_ds,
                              batch_size  = 8,
                              shuffle     = False,
                              num_workers = cfg['data']['num_workers'])

    # Model
    model     = build_model(in_channels=cfg['model']['in_channels'], device=device)
    criterion = DiceBCELoss(pos_weight=cfg['training']['pos_weight'], device=device)
    optimizer = optim.AdamW(model.parameters(),
                            lr           = cfg['training']['learning_rate'],
                            weight_decay = cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg['training']['epochs'], eta_min=1e-6)

    best_f1 = 0.0
    for epoch in range(1, cfg['training']['epochs'] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Ep {epoch} [Train]",
                                leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        agg = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                m = compute_metrics(model(imgs), masks)
                for k in agg: agg[k] += m[k]

        TP, FP, FN = agg['TP'], agg['FP'], agg['FN']
        p   = TP / (TP + FP + 1e-8)
        r   = TP / (TP + FN + 1e-8)
        f1  = 2 * p * r / (p + r + 1e-8)
        iou = TP / (TP + FP + FN + 1e-8)
        scheduler.step()

        tag = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_f1': f1, 'val_iou': iou}, 'best_model.pth')
            tag = " ← best"

        print(f"Ep {epoch:02d} | loss {train_loss:.4f} | IoU {iou:.4f} | "
              f"P {p:.4f} | R {r:.4f} | F1 {f1:.4f} | "
              f"{int(time.time()-t0)}s{tag}")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print("Saved: best_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args.config)
