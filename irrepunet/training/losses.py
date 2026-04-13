"""Loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation.

    Parameters
    ----------
    smooth : float
        Smoothing factor to avoid division by zero
    softmax : bool
        Whether to apply softmax to predictions
    include_background : bool
        Whether to include background class in the Dice computation
    batch_dice : bool
        If True, compute Dice over the entire batch (sum TP/FP/FN across
        batch before computing ratio). If False, compute Dice per sample
        and average — gives equal weight to each sample regardless of
        lesion volume. Default False (per-sample), matching nnUNet's
        single-stage 3d_fullres behavior.
    """

    def __init__(self, smooth: float = 1e-5, softmax: bool = True,
                 include_background: bool = True, batch_dice: bool = False):
        super().__init__()
        self.smooth = smooth
        self.softmax = softmax
        self.include_background = include_background
        self.batch_dice = batch_dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predictions of shape (B, C, D, H, W)
        target : torch.Tensor
            Target labels of shape (B, D, H, W) with class indices
            or (B, C, D, H, W) one-hot encoded

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        if self.softmax:
            pred = F.softmax(pred, dim=1)

        # Convert target to one-hot if needed
        if target.ndim == pred.ndim - 1:
            target = F.one_hot(target.long(), num_classes=pred.shape[1])
            target = target.permute(0, 4, 1, 2, 3).float()

        if self.batch_dice:
            # Reduce over batch and spatial dims — biases toward large volumes
            dims = (0, 2, 3, 4)
            intersection = (pred * target).sum(dim=dims)
            union = pred.sum(dim=dims) + target.sum(dim=dims)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            # Per-sample dice: reduce over spatial dims only, then average
            dims = (2, 3, 4)
            intersection = (pred * target).sum(dim=dims)  # (B, C)
            union = pred.sum(dim=dims) + target.sum(dim=dims)  # (B, C)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
            dice = dice.mean(dim=0)  # average across batch -> (C,)

        # Average over classes (excluding background if requested)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss.

    Parameters
    ----------
    dice_weight : float
        Weight for Dice loss
    ce_weight : float
        Weight for Cross-Entropy loss
    smooth : float
        Smoothing factor for Dice
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = True,
        batch_dice: bool = False
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth, softmax=True,
                                  include_background=include_background,
                                  batch_dice=batch_dice)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predictions of shape (B, C, D, H, W)
        target : torch.Tensor
            Target labels of shape (B, D, H, W)

        Returns
        -------
        torch.Tensor
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target.long())
        return self.dice_weight * dice + self.ce_weight * ce


class DeepSupervisionLoss(nn.Module):
    """Wrapper for deep supervision loss computation.

    Computes weighted loss across multiple output scales (nnUNet-style).
    Weights decrease by factor of 2 for each coarser scale.

    Parameters
    ----------
    base_loss : nn.Module
        Base loss function (e.g., DiceCELoss)
    n_scales : int
        Number of output scales
    weight_decay : float
        Factor by which weights decrease per scale (default 0.5 = halving)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        n_scales: int,
        weight_decay: float = 0.5
    ):
        super().__init__()
        self.base_loss = base_loss
        self.n_scales = n_scales

        # Compute weights: [1/(2^(n-1)), 1/(2^(n-2)), ..., 1/2, 1]
        # Finest resolution has weight 1
        weights = [weight_decay ** i for i in range(n_scales - 1, -1, -1)]
        total = sum(weights)
        self.weights = [w / total for w in weights]  # Normalize to sum to 1

    def forward(
        self,
        preds: list,
        targets: list
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        preds : list of torch.Tensor
            Predictions at each scale, from coarsest to finest
        targets : list of torch.Tensor
            Target labels at each scale, from coarsest to finest

        Returns
        -------
        torch.Tensor
            Weighted sum of losses
        """
        assert len(preds) == len(targets), \
            f"Mismatched scales: {len(preds)} preds and {len(targets)} targets"

        # Use precomputed weights if scale count matches, otherwise recompute
        if len(preds) == self.n_scales:
            weights = self.weights
        else:
            # Spatial splitting can reduce valid scales; use tail of weight list
            weights = self.weights[-len(preds):]
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]

        total_loss = 0.0
        for pred, target, weight in zip(preds, targets, weights):
            loss = self.base_loss(pred, target)
            total_loss = total_loss + weight * loss

        return total_loss


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    n_classes: int = None
) -> torch.Tensor:
    """Compute Dice score per class.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of shape (B, C, D, H, W) (logits)
    target : torch.Tensor
        Target labels of shape (B, D, H, W)
    smooth : float
        Smoothing factor
    n_classes : int, optional
        Number of classes. If None, inferred from pred shape.

    Returns
    -------
    torch.Tensor
        Dice scores per class of shape (C,)
    """
    if n_classes is None:
        n_classes = pred.shape[1]

    pred_argmax = pred.argmax(dim=1)  # (B, D, H, W)

    dice_scores = []
    for c in range(n_classes):
        pred_c = (pred_argmax == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return torch.stack(dice_scores)
