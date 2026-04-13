"""JAX/Flax NNX port of E3nnUNet for equivariant 3D medical image segmentation."""

from irrepunet.models_jax.unet import E3nnUNet, create_model
from irrepunet.models_jax.layers import (
    VoxelConvolution,
    EquivariantGate,
    EquivariantBatchNorm,
    EquivariantLayerNorm,
    EquivariantPool3d,
    ConvolutionBlock,
    Encoder,
    Decoder,
)
from irrepunet.models_jax.radial_basis import soft_one_hot_linspace, soft_unit_step
from irrepunet.models_jax.tp_right import build_tp_right
from irrepunet.models_jax.train import (
    configure_memory_optimizations,
    dice_loss,
    cross_entropy_loss,
    dice_ce_loss,
    deep_supervision_loss,
    create_train_state,
    train_step,
    eval_step,
    create_jitted_train_step,
    create_jitted_train_step_dynamic,
    offload_optimizer_state_to_cpu,
    restore_optimizer_state_to_gpu,
)
from irrepunet.models_jax.weight_transfer import transfer_weights, load_pytorch_checkpoint
from irrepunet.models_jax.inference_layers import (
    distill,
    distill_at_spacing,
    DistilledVoxelConvolution,
    DistilledLinearHead,
)

__all__ = [
    "E3nnUNet",
    "create_model",
    "VoxelConvolution",
    "EquivariantGate",
    "EquivariantBatchNorm",
    "EquivariantLayerNorm",
    "EquivariantPool3d",
    "ConvolutionBlock",
    "Encoder",
    "Decoder",
    "soft_one_hot_linspace",
    "soft_unit_step",
    "build_tp_right",
    "dice_loss",
    "cross_entropy_loss",
    "dice_ce_loss",
    "deep_supervision_loss",
    "configure_memory_optimizations",
    "create_train_state",
    "train_step",
    "eval_step",
    "create_jitted_train_step",
    "create_jitted_train_step_dynamic",
    "offload_optimizer_state_to_cpu",
    "restore_optimizer_state_to_gpu",
    "transfer_weights",
    "load_pytorch_checkpoint",
    "distill",
    "distill_at_spacing",
    "DistilledVoxelConvolution",
    "DistilledLinearHead",
]
