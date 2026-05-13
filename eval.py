import argparse
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import EOSARChangeDataset
from model import build_model, compute_metrics


def evaluate(data_path, weights_path, split='test'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    ds = load_dataset(data_path)

    # Split index config
    split_cfg = {
        'test': dict(pre_start=77,  post_start=0,
                     target_start=154, n_scenes=77),
        'validation': dict(pre_start=334, post_start=0,
                           target_start=668, n_scenes=334),
    }
    cfg = split_cfg[split]

    transform = A.Compose([A.CenterCrop(256, 256), ToTensorV2()])
    dataset   = EOSARChangeDataset(ds[split], transform=transform, **cfg)
    loader    = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    # Load model
    model = build_model(in_channels=4, device=device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Evaluate
    agg = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    all_preds, all_masks = [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            m = compute_metrics(logits, masks)
            for k in agg: agg[k] += m[k]
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().flatten()
            gt    = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_masks.extend(gt)

    TP, FP, FN, TN = agg['TP'], agg['FP'], agg['FN'], agg['TN']
    p   = TP / (TP + FP + 1e-8)
    r   = TP / (TP + FN + 1e-8)
    f1  = 2 * p * r / (p + r + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    print(f"\n{'='*45}")
    print(f"  {split.upper()} RESULTS")
    print(f"{'='*45}")
    print(f"  IoU       : {iou:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_masks, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                xticklabels=['Pred No-Change', 'Pred Change'],
                yticklabels=['True No-Change', 'True Change'])
    ax.set_title(f'Confusion Matrix — {split}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{split}.png', dpi=150)
    print(f"\nSaved: confusion_matrix_{split}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='doron333/change-detection-dataset')
    parser.add_argument('--weights',   default='best_model.pth')
    parser.add_argument('--split',     default='test')
    args = parser.parse_args()
    evaluate(args.data_path, args.weights, args.split)
