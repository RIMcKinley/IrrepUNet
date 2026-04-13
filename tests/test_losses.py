"""Tests for irrepunet.training.losses module."""

import pytest
import torch

from irrepunet.training.losses import (
    DiceLoss,
    DiceCELoss,
    DeepSupervisionLoss,
    compute_dice_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perfect_pred(n_classes, spatial=(4, 4, 4), batch=1):
    """Create pred logits and target where pred perfectly matches target."""
    target = torch.zeros(batch, *spatial, dtype=torch.long)
    # Put class 1 in the first quarter
    target[:, : spatial[0] // 2, :, :] = 1

    # Large logits for correct class
    pred = torch.zeros(batch, n_classes, *spatial)
    for c in range(n_classes):
        mask = (target == c).float().unsqueeze(1)
        pred[:, c:c+1] = mask * 10.0 - (1 - mask) * 10.0
    return pred, target


def _make_all_wrong_pred(n_classes, spatial=(4, 4, 4), batch=1):
    """Create pred logits where prediction is maximally wrong."""
    target = torch.zeros(batch, *spatial, dtype=torch.long)
    target[:, : spatial[0] // 2, :, :] = 1

    # Invert: predict class 0 everywhere
    pred = torch.zeros(batch, n_classes, *spatial)
    pred[:, 0] = 10.0
    return pred, target


# ---------------------------------------------------------------------------
# DiceLoss
# ---------------------------------------------------------------------------

class TestDiceLoss:
    def test_perfect_prediction_near_zero(self):
        """Perfect overlap should give loss near 0."""
        loss_fn = DiceLoss(smooth=1e-5)
        pred, target = _make_perfect_pred(2)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_no_overlap_near_one(self):
        """No overlap should give loss near 1."""
        loss_fn = DiceLoss(smooth=1e-5)
        pred, target = _make_all_wrong_pred(2)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.4

    def test_loss_is_scalar(self):
        loss_fn = DiceLoss()
        pred, target = _make_perfect_pred(2)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_output_range(self):
        """Dice loss should be in [0, 1]."""
        loss_fn = DiceLoss()
        pred, target = _make_perfect_pred(2)
        loss = loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_multiclass(self):
        """Should handle 3+ classes."""
        loss_fn = DiceLoss()
        pred, target = _make_perfect_pred(4)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1

    def test_one_hot_target(self):
        """Should accept pre-encoded one-hot targets."""
        loss_fn = DiceLoss()
        pred, target_idx = _make_perfect_pred(2)
        # Convert to one-hot
        target_oh = torch.nn.functional.one_hot(target_idx, 2)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()
        loss = loss_fn(pred, target_oh)
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        loss_fn = DiceLoss()
        pred = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)


# ---------------------------------------------------------------------------
# DiceCELoss
# ---------------------------------------------------------------------------

class TestDiceCELoss:
    def test_perfect_prediction_low_loss(self):
        loss_fn = DiceCELoss(dice_weight=1.0, ce_weight=1.0)
        pred, target = _make_perfect_pred(2)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1

    def test_weights_affect_loss(self):
        pred, target = _make_perfect_pred(2)
        loss_dice_only = DiceCELoss(dice_weight=1.0, ce_weight=0.0)(pred, target)
        loss_ce_only = DiceCELoss(dice_weight=0.0, ce_weight=1.0)(pred, target)
        loss_both = DiceCELoss(dice_weight=1.0, ce_weight=1.0)(pred, target)
        # Combined should be approximately sum of components
        assert abs(loss_both.item() - (loss_dice_only.item() + loss_ce_only.item())) < 0.01

    def test_is_scalar(self):
        loss_fn = DiceCELoss()
        pred, target = _make_perfect_pred(2)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# DeepSupervisionLoss
# ---------------------------------------------------------------------------

class TestDeepSupervisionLoss:
    def test_weights_sum_to_one(self):
        ds_loss = DeepSupervisionLoss(DiceCELoss(), n_scales=4)
        assert abs(sum(ds_loss.weights) - 1.0) < 1e-6

    def test_weight_ordering(self):
        """Weights should increase from coarsest to finest."""
        ds_loss = DeepSupervisionLoss(DiceCELoss(), n_scales=4)
        for i in range(len(ds_loss.weights) - 1):
            assert ds_loss.weights[i] <= ds_loss.weights[i + 1]

    def test_forward_matches_scales(self):
        n_scales = 3
        ds_loss = DeepSupervisionLoss(DiceCELoss(), n_scales=n_scales)

        preds = []
        targets = []
        for i in range(n_scales):
            s = 4 * (2 ** i)
            p, t = _make_perfect_pred(2, spatial=(s, s, s))
            preds.append(p)
            targets.append(t)

        loss = ds_loss(preds, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_mismatched_preds_targets_raises(self):
        ds_loss = DeepSupervisionLoss(DiceCELoss(), n_scales=3)
        preds = [torch.randn(1, 2, 4, 4, 4)]
        targets = [torch.zeros(1, 4, 4, 4, dtype=torch.long),
                   torch.zeros(1, 8, 8, 8, dtype=torch.long)]
        with pytest.raises(AssertionError):
            ds_loss(preds, targets)


# ---------------------------------------------------------------------------
# compute_dice_score
# ---------------------------------------------------------------------------

class TestComputeDiceScore:
    def test_perfect_prediction(self):
        pred, target = _make_perfect_pred(2)
        scores = compute_dice_score(pred, target)
        assert scores.shape == (2,)
        # Both classes should be near 1.0
        assert scores[0].item() > 0.99
        assert scores[1].item() > 0.99

    def test_output_in_range(self):
        pred = torch.randn(1, 2, 4, 4, 4)
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        scores = compute_dice_score(pred, target)
        for s in scores:
            assert 0.0 <= s.item() <= 1.0
