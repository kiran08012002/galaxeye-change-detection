import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss to handle class imbalance."""

    def __init__(self, pos_weight=None, device='cpu'):
        super().__init__()
        pw = torch.tensor([pos_weight]).to(device) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.unsqueeze(1))
        probs    = torch.sigmoid(logits)
        dims     = (0, 2, 3)
        inter    = (probs * targets.unsqueeze(1)).sum(dims)
        total    = (probs + targets.unsqueeze(1)).sum(dims)
        dice     = 1 - (2 * inter + 1) / (total + 1)
        return bce_loss + dice.mean()


def build_model(in_channels=4, device='cpu'):
    """Build UNet with ImageNet-pretrained ResNet34 encoder."""
    model = smp.Unet(
        encoder_name    = 'resnet34',
        encoder_weights = 'imagenet',
        in_channels     = in_channels,
        classes         = 1,
        activation      = None,
    )
    return model.to(device)


def compute_metrics(logits, masks, threshold=0.5):
    """Compute IoU, Precision, Recall, F1 for the change class (label=1)."""
    preds = (torch.sigmoid(logits) > threshold).float()
    masks = masks.unsqueeze(1).float()

    TP = (preds * masks).sum().item()
    FP = (preds * (1 - masks)).sum().item()
    FN = ((1 - preds) * masks).sum().item()
    TN = ((1 - preds) * (1 - masks)).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)

    return {'IoU': iou, 'Precision': precision, 'Recall': recall,
            'F1': f1, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
