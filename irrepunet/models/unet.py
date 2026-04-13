"""Equivariant UNet model for 3D medical image segmentation."""

import math
import torch
import torch.nn as nn
import numpy as np

from e3nn.o3 import Irreps, Linear

from .layers import Encoder, Decoder, MultipolePool, _pool_factor


class E3nnUNet(nn.Module):
    """Equivariant UNet with physical units for 3D medical image segmentation.

    This model uses e3nn for rotation-equivariant convolutions, making it
    suitable for medical imaging where orientation should not affect predictions.

    Supports dynamic spacing: pass spacing to forward() for multi-resolution
    training. The model will rebuild internal lattices when spacing changes.

    Parameters
    ----------
    n_classes : int
        Number of output classes for segmentation
    in_channels : int
        Number of input channels (default: 1 for single modality)
    diameter : float
        Diameter of the initial convolution kernel in physical units
    num_radial_basis : int
        Number of radial basis functions for convolution
    spacing : tuple of float
        Default physical spacing (voxel size) in each dimension
    normalization : str
        Normalization type: 'batch', 'instance', or 'None'
    n_base_filters : int
        Base number of filters (multiplied at each level)
    n_downsample : int
        Number of downsampling (pooling) operations
    equivariance : str
        Type of equivariance: 'SO3' (rotations only) or 'O3' (rotations + reflections)
    lmax : int
        Maximum spherical harmonics degree
    pool_mode : str
        Pooling mode: 'maxpool3d' or 'average'
    scale : float
        Pooling scale factor in physical units
    dropout_prob : float
        Dropout probability (0 to disable)
    scalar_upsampling : bool
        Use only scalar features in decoder (faster but less expressive)
    cutoff : bool
        Apply cutoff to radial basis functions
    max_features : int
        Maximum number of features (irrep dimension) per level, similar to
        nnUNet's UNet_max_features_3d. Default is 320 to match nnUNet.
        Features are capped when they would exceed this value.
    irrep_ratios : tuple or dict
        Multipliers for each l degree. Can be:
        - tuple: (4, 2, 1) - same ratios for even/odd parity
        - dict: {'e': (4, 2, 1), 'o': (2, 1, 0)} - separate ratios per parity
        Default (4, 2, 1) gives 4*ne scalars, 2*ne vectors, 1*ne tensors.
    fill_to_max : bool
        If True, top up capped levels with extra scalar irreps to reach
        exactly max_features. Default False.
    activation : str
        Scalar activation function for Gate layers. One of:
        'softplus', 'selu', 'relu', 'silu', 'gelu'. Default 'softplus'.
    """

    ACTIVATIONS = {
        'softplus': torch.nn.functional.softplus,
        'selu': torch.selu,
        'relu': torch.relu,
        'silu': torch.nn.functional.silu,
        'gelu': torch.nn.functional.gelu,
    }

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 1,
        diameter: float = 5.0,
        num_radial_basis: int = 5,
        spacing: tuple = (1.0, 1.0, 1.0),
        normalization: str = 'instance',
        n_base_filters: int = 2,
        n_downsample: int = 2,
        equivariance: str = 'SO3',
        lmax: int = 2,
        pool_mode: str = 'maxpool3d',
        scale: float = 2.0,
        dropout_prob: float = 0.0,
        scalar_upsampling: bool = False,
        cutoff: bool = True,
        deep_supervision: bool = False,
        max_features: int = 320,
        irrep_ratios: tuple = (4, 2, 1),
        fill_to_max: bool = False,
        activation: str = 'softplus',
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        kernel_growth: float = 2.0,
        sequential_sc: bool = False,
        sc_mode: str = None,
        fused_gate: bool = True,
        sphere_norm: bool = True,
        backend: str = "e3nn",
        pyramid=False,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.num_classes = n_classes  # For compatibility
        self.n_downsample = n_downsample
        self.conv_op = nn.Conv3d  # For compatibility with some nnUNet utilities
        self.deep_supervision = deep_supervision
        self.fused_gate = fused_gate
        self.sphere_norm = sphere_norm
        self.backend = backend
        self.pyramid = pyramid

        # Store configuration for dynamic spacing rebuild
        self.default_spacing = spacing
        self._current_spacing = None
        self.diameter = diameter
        self.num_radial_basis = num_radial_basis
        self.normalization = normalization
        self.n_base_filters = n_base_filters
        self.equivariance = equivariance
        self.lmax = lmax
        self.pool_mode = pool_mode
        self.scale = scale
        self.dropout_prob = dropout_prob
        self.scalar_upsampling = scalar_upsampling
        self.cutoff = cutoff
        self.in_channels = in_channels
        self.activation_name = activation
        self.pre_norm = activation == 'selu'
        self.max_features = max_features
        self.kernel_trim_threshold = kernel_trim_threshold
        self.kernel_trim_cross_section = kernel_trim_cross_section
        self.kernel_growth = kernel_growth
        # Resolve sc_mode
        if sc_mode is not None:
            self.sc_mode = sc_mode
        elif sequential_sc:
            self.sc_mode = "sc_first"
        else:
            self.sc_mode = "parallel"
        self.sequential_sc = self.sc_mode == "sc_first"  # backward compat
        # Preserve dict format, convert list/tuple to tuple
        if isinstance(irrep_ratios, dict):
            self.irrep_ratios = {k: tuple(v) for k, v in irrep_ratios.items()}
        else:
            self.irrep_ratios = tuple(irrep_ratios)
        self.fill_to_max = fill_to_max

        # Validate inputs
        assert normalization in ['None', 'batch', 'instance', 'layer'], \
            "normalization must be 'batch', 'instance', 'layer', or 'None'"
        assert pool_mode in ['maxpool3d', 'average', 's2d'], \
            "pool_mode must be 'maxpool3d', 'average', or 's2d'"
        assert equivariance in ['SO3', 'O3'], \
            "equivariance must be 'SO3' or 'O3'"

        # Setup equivariance type
        act_fn = self.ACTIVATIONS[activation]
        if equivariance == 'SO3':
            self.activation = [act_fn]
            self.irreps_sh = Irreps.spherical_harmonics(lmax, 1)
            self.ne = n_base_filters  # even parity features
            self.no = 0  # odd parity features
        else:  # O3
            self.activation = [act_fn, torch.tanh]
            self.irreps_sh = Irreps.spherical_harmonics(lmax, -1)
            self.ne = n_base_filters
            self.no = n_base_filters

        # Input irreps (scalar input)
        self.input_irreps = Irreps(f"{in_channels}x0e")

        # Output irreps (scalar classes only for segmentation)
        self.output_irreps = Irreps(f"{n_classes}x0e")

        # Compute scales and diameters at each level
        self.scales = [scale * (2 ** i) for i in range(n_downsample)]
        self.diameters = [diameter * (kernel_growth ** i) for i in range(n_downsample + 1)]

        # Build encoder and decoder with default spacing
        self._build_network(spacing)

        # Output projection
        self.out = Linear(self.decoder.up_blocks[-1].irreps_out, self.output_irreps)

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def _compute_steps_array(self, spacing: tuple) -> list:
        """Compute effective spacing at each level after pooling."""
        steps_array = [spacing]
        for i in range(self.n_downsample):
            output_steps = []
            for step in spacing:
                kernel_dim = _pool_factor(self.scales[i], step)
                output_steps.append(kernel_dim * step)
            steps_array.append(tuple(output_steps))
        return steps_array

    def _build_network(self, spacing: tuple):
        """Build or rebuild encoder and decoder for given spacing."""
        steps_array = self._compute_steps_array(spacing)

        # Build encoder
        self.encoder = Encoder(
            n_downsample=self.n_downsample,
            activation=self.activation,
            irreps_sh=self.irreps_sh,
            ne=self.ne,
            no=self.no,
            normalization=self.normalization,
            irreps_in=self.input_irreps,
            diameters=self.diameters,
            num_radial_basis=self.num_radial_basis,
            steps_array=steps_array,
            down_op=self.pool_mode,
            scales=self.scales,
            dropout_prob=self.dropout_prob,
            cutoff=self.cutoff,
            max_features=self.max_features,
            irrep_ratios=self.irrep_ratios,
            fill_to_max=self.fill_to_max,
            pre_norm=self.pre_norm,
            kernel_trim_threshold=self.kernel_trim_threshold,
            kernel_trim_cross_section=self.kernel_trim_cross_section,
            sc_mode=self.sc_mode,
            fused_gate=self.fused_gate,
            sphere_norm=self.sphere_norm,
            backend=self.backend,
            pyramid=self.pyramid,
        )

        # Build decoder - mirror encoder's per-level feature depths
        # Encoder ne_per_level has n_downsample+1 entries (one per level incl. bottleneck)
        # Decoder needs n_downsample entries, mirroring encoder levels in reverse (excl. bottleneck)
        ne_per_level_decoder = self.encoder.ne_per_level[-2::-1]  # reverse, skip bottleneck
        no_per_level_decoder = self.encoder.no_per_level[-2::-1] if self.no > 0 else None

        # For s2d, pass encoder output irreps directly as decoder targets
        if self.pool_mode == 's2d':
            decoder_irreps_per_level = list(reversed(self.encoder.down_irreps_out[:-1]))
        else:
            decoder_irreps_per_level = None

        self.decoder = Decoder(
            n_blocks=self.n_downsample,
            activation=self.activation,
            irreps_sh=self.irreps_sh,
            ne_per_level=ne_per_level_decoder,
            no_per_level=no_per_level_decoder,
            normalization=self.normalization,
            encoder_irreps=self.encoder.down_irreps_out,
            diameters=self.diameters[::-1][1:],
            num_radial_basis=self.num_radial_basis,
            steps_array=steps_array[::-1][1:],
            scales=self.scales[::-1],
            dropout_prob=self.dropout_prob,
            scalar_upsampling=self.scalar_upsampling,
            cutoff=self.cutoff,
            deep_supervision=self.deep_supervision,
            n_classes=self.n_classes,
            max_features=self.max_features,
            irrep_ratios=self.irrep_ratios,
            fill_to_max=self.fill_to_max,
            pre_norm=self.pre_norm,
            kernel_trim_threshold=self.kernel_trim_threshold,
            kernel_trim_cross_section=self.kernel_trim_cross_section,
            sc_mode=self.sc_mode,
            fused_gate=self.fused_gate,
            irreps_per_level=decoder_irreps_per_level,
            pool_mode=self.pool_mode,
            sphere_norm=self.sphere_norm,
            backend=self.backend,
            pyramid=self.pyramid,
        )

        # Store pooling info for padding computation
        self._encoder_pools = self.encoder.down_pool
        self._current_spacing = spacing

    def _rebuild_for_spacing(self, spacing: tuple, scales: list = None):
        """Update network components for new spacing in-place.

        This updates the convolution lattices and pooling layers for the
        new spacing without replacing parameter tensors, preserving
        optimizer references.

        Parameters
        ----------
        spacing : tuple
            Physical spacing for this batch.
        scales : list, optional
            Override pooling/upsampling scales (e.g., for scale jitter).
            If None, uses self.scales.
        """
        if scales is not None:
            orig_scales = self.scales
            self.scales = scales
            for i, pool in enumerate(self.encoder.down_pool):
                pool.scale = scales[i]
            self.decoder.scales = scales[::-1]

        steps_array = self._compute_steps_array(spacing)

        # Update encoder (steps_array has n_downsample + 1 entries)
        self.encoder.update_spacing(steps_array)

        # Update decoder (uses reversed steps_array without first element)
        self.decoder.update_spacing(steps_array[::-1][1:])

        # Update pooling info for padding computation
        self._encoder_pools = self.encoder.down_pool

        if scales is not None:
            # Restore original scales (kernel sizes already computed)
            self.scales = orig_scales
            for i, pool in enumerate(self.encoder.down_pool):
                pool.scale = orig_scales[i]
            self.decoder.scales = orig_scales[::-1]
            # Force rebuild on next call (jitter changes each time)
            self._current_spacing = None
        else:
            self._current_spacing = spacing

    def _compute_padding(self, image_shape):
        """Compute padding needed for proper downsampling/upsampling."""
        pooling_factor = np.ones(3, dtype='int')
        for pool in self._encoder_pools:
            if isinstance(pool, MultipolePool):
                pooling_factor *= np.array(pool.factors)
            else:
                pooling_factor *= np.array(pool.kernel_size)

        pad = []
        for f, s in zip(pooling_factor, image_shape):
            t = s % f
            p = (f - t) if t != 0 else 0
            pad.append(p)

        return pad

    def forward(self, x, spacing=None, scales=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, depth, height, width)
        spacing : tuple, optional
            Physical spacing for this batch. If provided and different from
            current spacing, the network will be rebuilt for this spacing.
            If None, uses default_spacing.
        scales : list, optional
            Override pooling/upsampling scales for this forward pass
            (e.g., for scale jitter during training). Forces rebuild.

        Returns
        -------
        If deep_supervision=False:
            torch.Tensor of shape (batch, n_classes, depth, height, width)
        If deep_supervision=True:
            List of tensors from coarsest to finest resolution
        """
        # Handle dynamic spacing
        if spacing is None:
            spacing = self.default_spacing
        else:
            # Normalize spacing to tuple of floats
            spacing = tuple(float(s) for s in spacing)

        # Rebuild network if spacing changed or custom scales provided
        if spacing != self._current_spacing or scales is not None:
            self._rebuild_for_spacing(spacing, scales=scales)

        # Compute and apply padding
        pad = self._compute_padding(x.shape[-3:])
        original_shape = x.shape[-3:]
        x = nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        # Encoder
        encoder_features = self.encoder(x)

        # Decoder
        try:
            decoder_out = self.decoder(encoder_features[-1], encoder_features)
        except RuntimeError as e:
            # Add spacing context to the error
            pool_factors = [
                (getattr(p, 'factors', None) or getattr(p, 'kernel_size', None))
                for p in self._encoder_pools
            ]
            raise RuntimeError(
                f"{e}\nspacing={spacing}, pad={pad}, input_shape={original_shape}, "
                f"padded_shape={x.shape[2:]}, pool_factors={pool_factors}"
            ) from e

        if self.deep_supervision and isinstance(decoder_out, tuple):
            x, ds_outputs = decoder_out

            # Main output projection
            dtype = x.dtype
            x = self.out(x.transpose(1, 4)).to(dtype).transpose(1, 4)
            bias = self.bias.reshape(-1, 1, 1, 1)
            x = x + bias

            # Remove padding from main output
            x = x[..., pad[0]:, pad[1]:, pad[2]:]

            # Add bias to deep supervision outputs and remove padding
            # Outputs are from coarsest to finest
            final_outputs = []
            for i, ds_out in enumerate(ds_outputs):
                ds_out = ds_out + bias
                # Compute scaled padding for this resolution
                scale_factor = 2 ** (self.n_downsample - 1 - i)
                scaled_pad = [max(0, p // scale_factor) for p in pad]
                ds_out = ds_out[..., scaled_pad[0]:, scaled_pad[1]:, scaled_pad[2]:]
                final_outputs.append(ds_out)

            # Add main (finest) output last
            final_outputs.append(x)

            return final_outputs
        else:
            x = decoder_out

            # Output projection
            dtype = x.dtype
            x = self.out(x.transpose(1, 4)).to(dtype).transpose(1, 4)

            # Add bias
            bias = self.bias.reshape(-1, 1, 1, 1)
            x = x + bias

            # Remove padding
            x = x[..., pad[0]:, pad[1]:, pad[2]:]

            return x

    def get_model_config(self) -> dict:
        """Return all constructor arguments as a dict for serialization.

        This can be used to reconstruct the model from a checkpoint:
            config = model.get_model_config()
            new_model = E3nnUNet(**config)
        """
        return {
            'model_class': self.__class__.__name__,
            'model_kwargs': {
                'n_classes': self.n_classes,
                'in_channels': self.in_channels,
                'diameter': self.diameter,
                'num_radial_basis': self.num_radial_basis,
                'n_base_filters': self.n_base_filters,
                'n_downsample': self.n_downsample,
                'equivariance': self.equivariance,
                'lmax': self.lmax,
                'normalization': self.normalization,
                'pool_mode': self.pool_mode,
                'scale': self.scale,
                'dropout_prob': self.dropout_prob,
                'scalar_upsampling': self.scalar_upsampling,
                'cutoff': self.cutoff,
                'deep_supervision': self.deep_supervision,
                'max_features': self.max_features,
                'irrep_ratios': self.irrep_ratios,
                'fill_to_max': self.fill_to_max,
                'activation': self.activation_name,
                'kernel_trim_threshold': self.kernel_trim_threshold,
                'kernel_trim_cross_section': self.kernel_trim_cross_section,
                'kernel_growth': self.kernel_growth,
                'sequential_sc': self.sequential_sc,
                'sc_mode': self.sc_mode,
                'fused_gate': self.fused_gate,
                'sphere_norm': self.sphere_norm,
                'backend': self.backend,
                'pyramid': self.pyramid,
            },
        }

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        """Reconstruct a model from a checkpoint file.

        Loads model_config from the checkpoint to create the model,
        then loads spacing-independent weights.

        Parameters
        ----------
        path : str or Path
            Path to checkpoint file (.pt)
        device : str or torch.device
            Device to load onto

        Returns
        -------
        model : E3nnUNet
            Model with loaded weights, in eval mode
        checkpoint : dict
            Full checkpoint dict (contains optimizer state, epoch, etc.)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['model_config']
        # Handle both formats: {model_class, model_kwargs} and flat dict
        if 'model_kwargs' in config:
            kwargs = config['model_kwargs']
        else:
            kwargs = config
        # Old checkpoints lack fused_gate; default to False (legacy Gate) for compat
        kwargs.setdefault('fused_gate', False)
        # Old checkpoints lack sphere_norm; default to False (cuboid normalization) for compat
        kwargs.setdefault('sphere_norm', False)
        # Old checkpoints lack backend; default to e3nn
        kwargs.setdefault('backend', 'e3nn')
        model = cls(**kwargs)
        # Load spacing-independent weights.
        # Filter out spacing-dependent pyramid buffers (_scatter_membership)
        # which are recomputed on update_spacing().  Also handle pyramid_logits
        # size mismatches (K depends on the spacing at save time).
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_sd = model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if '_scatter_membership' in k:
                continue  # spacing-dependent, recomputed
            if k in model_sd and v.shape != model_sd[k].shape:
                # pyramid_logits: copy what fits, leave rest at init
                if 'pyramid_logits' in k:
                    n = min(v.shape[0], model_sd[k].shape[0])
                    model_sd[k][:n] = v[:n]
                    filtered[k] = model_sd[k]
                    continue
                # Other mismatches: skip (will use model default)
                continue
            filtered[k] = v
        model.load_state_dict(filtered, strict=False)
        model = model.to(device)
        model.eval()
        return model, checkpoint

    def _rebuild_for_superres(self, sub_spacing: tuple, orig_spacing: tuple):
        """Update encoder with sub_spacing and decoder with orig_spacing.

        This allows the encoder to process low-resolution input while the
        decoder upsamples to high-resolution output.

        Parameters
        ----------
        sub_spacing : tuple
            Voxel spacing of the subsampled (low-resolution) input
        orig_spacing : tuple
            Voxel spacing of the original (high-resolution) output
        """
        enc_steps = self._compute_steps_array(sub_spacing)
        self.encoder.update_spacing(enc_steps)
        self._encoder_pools = self.encoder.down_pool

        dec_steps = self._compute_steps_array(orig_spacing)
        self.decoder.update_spacing(dec_steps[::-1][1:])

        self._current_spacing = None  # Invalidate cache (mixed spacing state)
        self._superres_sub_spacing = sub_spacing
        self._superres_orig_spacing = orig_spacing

    def forward_superres(self, x, sub_spacing: tuple, orig_spacing: tuple):
        """Forward pass for super-resolution training.

        Encoder processes input at sub_spacing, decoder upsamples to orig_spacing.

        Parameters
        ----------
        x : torch.Tensor
            Low-resolution input (batch, channels, D, H, W)
        sub_spacing : tuple
            Spacing of the low-resolution input
        orig_spacing : tuple
            Spacing of the target high-resolution output

        Returns
        -------
        torch.Tensor
            High-resolution output
        """
        import torch.nn.functional as F

        sub_spacing = tuple(float(s) for s in sub_spacing)
        orig_spacing = tuple(float(s) for s in orig_spacing)

        cached_sub = getattr(self, '_superres_sub_spacing', None)
        cached_orig = getattr(self, '_superres_orig_spacing', None)
        if sub_spacing != cached_sub or orig_spacing != cached_orig:
            self._rebuild_for_superres(sub_spacing, orig_spacing)

        pad = self._compute_padding(x.shape[-3:])
        x = nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        encoder_features = self.encoder(x)
        decoder_out = self.decoder.forward_superres(
            encoder_features[-1], encoder_features
        )

        dtype = decoder_out.dtype
        x = self.out(decoder_out.transpose(1, 4)).to(dtype).transpose(1, 4)
        bias = self.bias.reshape(-1, 1, 1, 1)
        x = x + bias

        # Compute target output size
        factors = np.round(np.array(sub_spacing) / np.array(orig_spacing)).astype(int)
        factors = np.maximum(factors, 1)
        unpadded_shape = np.array(x.shape[-3:])
        for d in range(3):
            if pad[d] > 0:
                unpadded_shape[d] -= pad[d] * factors[d]

        target_size = tuple(int(s) for s in unpadded_shape)

        if x.shape[2:] != target_size:
            x = F.interpolate(
                x, size=target_size,
                mode='trilinear', align_corners=True
            )

        return x


def create_model(
    n_classes: int,
    spacing: tuple = (1.0, 1.0, 1.0),
    **kwargs
) -> E3nnUNet:
    """Factory function to create an E3nnUNet model.

    Parameters
    ----------
    n_classes : int
        Number of segmentation classes
    spacing : tuple
        Voxel spacing in physical units
    **kwargs
        Additional arguments passed to E3nnUNet

    Returns
    -------
    E3nnUNet
        Configured model instance
    """
    return E3nnUNet(n_classes=n_classes, spacing=spacing, **kwargs)
