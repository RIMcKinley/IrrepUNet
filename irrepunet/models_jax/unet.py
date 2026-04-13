"""Equivariant UNet model for 3D medical image segmentation in JAX."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

import cuequivariance as cue

from irrepunet.models_jax.layers import (
    _parse_irreps, _irreps_dim,
    _precompute_sc_layout, _sc_weight_numel,
    _precompute_sc_scatter, _build_sc_weight_matrix_fast,
    Buffer,
    spherical_harmonics_irreps,
    Encoder, Decoder,
    _build_irreps, _features_per_ne, _parse_ratios,
)


class E3nnUNet(nnx.Module):
    """Equivariant UNet with physical units for 3D medical image segmentation.

    JAX/Flax NNX port with dynamic spacing support.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    in_channels : int
        Number of input channels.
    diameter : float
        Diameter of the initial convolution kernel in physical units.
    num_radial_basis : int
        Number of radial basis functions.
    spacing : tuple
        Physical spacing (voxel size) in each dimension.
    normalization : str
        Normalization: 'batch', 'instance', 'layer', or 'None'.
    n_base_filters : int
        Base number of filters.
    n_downsample : int
        Number of downsampling operations.
    equivariance : str
        'SO3' or 'O3'.
    lmax : int
        Maximum spherical harmonics degree.
    pool_mode : str
        'maxpool3d' or 'average'.
    scale : float
        Pooling scale factor in physical units.
    dropout_prob : float
        Dropout probability.
    scalar_upsampling : bool
        Use only scalar features in decoder.
    cutoff : bool
        Apply cutoff to radial basis functions.
    deep_supervision : bool
        Enable deep supervision outputs.
    max_features : int
        Max features per level.
    irrep_ratios : tuple or dict
        Multipliers for each l degree.
    fill_to_max : bool
        Top up capped levels with extra scalars.
    rngs : nnx.Rngs
        Random number generators.
    """

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
        checkpoint=False,
        remat_kernel: bool = False,
        kernel_trim_threshold: float = 1.0,
        kernel_trim_cross_section: float = 0.0,
        sphere_norm: bool = True,
        activation: str = 'softplus',
        *,
        rngs: nnx.Rngs,
    ):
        self.n_classes = n_classes
        self.n_downsample = n_downsample
        self.deep_supervision = deep_supervision
        self.spacing = spacing

        # Store config
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
        self.max_features = max_features
        if isinstance(irrep_ratios, dict):
            self.irrep_ratios = {k: tuple(v) for k, v in irrep_ratios.items()}
        else:
            self.irrep_ratios = tuple(irrep_ratios)
        self.fill_to_max = fill_to_max

        # Activation lookup
        _ACTIVATIONS = {
            'softplus': jax.nn.softplus,
            'selu': jax.nn.selu,
            'relu': jax.nn.relu,
            'silu': jax.nn.silu,
            'gelu': jax.nn.gelu,
        }
        act_fn = _ACTIVATIONS.get(activation, jax.nn.softplus)

        # Setup equivariance
        if equivariance == 'SO3':
            act_list = [act_fn]
            irreps_sh = spherical_harmonics_irreps(lmax, 1)
            ne = n_base_filters
            no = 0
        else:
            act_list = [act_fn, jnp.tanh]
            irreps_sh = spherical_harmonics_irreps(lmax, -1)
            ne = n_base_filters
            no = n_base_filters

        self.ne = ne
        self.no = no

        # Input/output irreps
        input_irreps = cue.Irreps("O3", f"{in_channels}x0e")
        output_irreps = cue.Irreps("O3", f"{n_classes}x0e")

        # Scales and diameters
        scales = [scale * (2 ** i) for i in range(n_downsample)]
        diameters = [diameter * (2 ** i) for i in range(n_downsample + 1)]

        # Compute steps array
        steps_array = self._compute_steps_array(spacing, scales, n_downsample)

        # Build encoder
        self.encoder = Encoder(
            n_downsample=n_downsample,
            activation=act_list,
            irreps_sh=irreps_sh,
            ne=ne,
            no=no,
            normalization=normalization,
            irreps_in=input_irreps,
            diameters=diameters,
            num_radial_basis=num_radial_basis,
            steps_array=steps_array,
            down_op=pool_mode,
            scales=scales,
            dropout_prob=dropout_prob,
            cutoff=cutoff,
            max_features=max_features,
            irrep_ratios=irrep_ratios,
            fill_to_max=fill_to_max,
            checkpoint=checkpoint,
            remat_kernel=remat_kernel,
            kernel_trim_threshold=kernel_trim_threshold,
            kernel_trim_cross_section=kernel_trim_cross_section,
            sphere_norm=sphere_norm,
            rngs=rngs,
        )

        # Build decoder — use UNCAPPED ne so the decoder can apply capping
        # per-level.  Using the capped value from the encoder causes ne to
        # reach 0 when n_downsample is large (e.g. 6) because repeated
        # halving of the small capped value exhausts it before all decoder
        # blocks are built.
        ne_decoder = ne * (2 ** (n_downsample - 1))
        no_decoder = no * (2 ** (n_downsample - 1)) if no > 0 else 0

        self.decoder = Decoder(
            n_blocks=n_downsample,
            activation=act_list,
            irreps_sh=irreps_sh,
            ne=ne_decoder,
            no=no_decoder,
            normalization=normalization,
            encoder_irreps=self.encoder.down_irreps_out,
            diameters=diameters[::-1][1:],
            num_radial_basis=num_radial_basis,
            steps_array=steps_array[::-1][1:],
            scales=scales[::-1],
            dropout_prob=dropout_prob,
            scalar_upsampling=scalar_upsampling,
            cutoff=cutoff,
            deep_supervision=deep_supervision,
            n_classes=n_classes,
            max_features=max_features,
            irrep_ratios=irrep_ratios,
            fill_to_max=fill_to_max,
            checkpoint=checkpoint,
            remat_kernel=remat_kernel,
            kernel_trim_threshold=kernel_trim_threshold,
            kernel_trim_cross_section=kernel_trim_cross_section,
            sphere_norm=sphere_norm,
            rngs=rngs,
        )

        # Output projection (equivariant linear: last decoder irreps -> n_classes scalars)
        last_irreps = self.decoder.up_blocks[-1].irreps_out
        self._out_layout = _precompute_sc_layout(last_irreps, output_irreps)
        out_numel = _sc_weight_numel(self._out_layout)
        self.out_weight = nnx.Param(jax.random.normal(rngs.params(), (out_numel,)))
        self._out_in_dim = _irreps_dim(last_irreps)
        self._out_out_dim = _irreps_dim(output_irreps)
        _out_scatter = _precompute_sc_scatter(
            self._out_layout, self._out_out_dim, self._out_in_dim
        )
        if _out_scatter is not None:
            self._out_sc_src = Buffer(_out_scatter[0])
            self._out_sc_dst = Buffer(_out_scatter[1])
            self._out_sc_alpha = Buffer(_out_scatter[2])
            self._has_out_sc = True
        else:
            self._has_out_sc = False

        # Learnable bias
        self.bias = nnx.Param(jnp.zeros(n_classes))

        # Store pool kernel sizes for padding computation
        self._pool_kernel_sizes = [p.kernel_size for p in self.encoder.down_pool]

    @staticmethod
    def _compute_steps_array(spacing, scales, n_downsample, override_pool_kernels=None):
        """Compute effective spacing at each level after pooling.

        Pool factors are computed from the ORIGINAL spacing (not accumulated),
        matching PyTorch's convention where each level's scale represents the
        target effective spacing from the input resolution.

        Parameters
        ----------
        spacing : tuple
            Physical spacing at level 0.
        scales : list of float
            Pooling scales per level (target effective spacing).
        n_downsample : int
            Number of pooling levels.
        override_pool_kernels : dict or None
            Optional dict mapping level index -> tuple of int (pool kernel per dim).
            If provided for a level, uses the override instead of computing from
            scale/step.
        """
        from irrepunet.models_jax.layers import _pool_factor

        steps_array = [spacing]
        for i in range(n_downsample):
            if override_pool_kernels and i in override_pool_kernels:
                override_k = override_pool_kernels[i]
                prev = steps_array[-1]
                output_steps = []
                for d, step in enumerate(prev):
                    k = override_k[d]
                    output_steps.append(k * step if k > 1 else step)
            else:
                output_steps = []
                for step in spacing:  # use ORIGINAL spacing
                    kernel_dim = _pool_factor(scales[i], step)
                    output_steps.append(kernel_dim * step)
            steps_array.append(tuple(output_steps))
        return steps_array

    def _compute_padding(self, image_shape):
        """Compute padding needed for proper downsampling/upsampling."""
        pooling_factor = np.ones(3, dtype='int')
        for ks in self._pool_kernel_sizes:
            pooling_factor *= np.array(ks)

        pad = []
        for f, s in zip(pooling_factor, image_shape):
            t = s % f
            p = (f - t) if t != 0 else 0
            pad.append(p)
        return pad

    def update_spacing(self, spacing: tuple, override_pool_kernels=None):
        """Update network for new spacing without replacing learnable parameters.

        Rebuilds all spacing-dependent buffers (lattice, SH, radial basis) and
        plain attributes (padding, kernel_size, scale_factors). Learnable
        weights, CG coefficients, optimizer state, norms, and gates are
        unchanged.

        For JIT'd training, call this *before* the JIT boundary (before
        nnx.split). The changed graphdef triggers JIT retracing. Each unique
        spacing is cached separately.

        Parameters
        ----------
        spacing : tuple
            New physical spacing (voxel size) per dimension.
        override_pool_kernels : dict or None
            Optional dict mapping level index -> tuple of int (pool kernel per dim).
            If provided for a level, uses the override instead of floor(scale/step).
            Used by targeted pooling to consolidate L2+ JIT families.
        """
        spacing = tuple(float(s) for s in spacing)
        # Normalize override to a comparable form (None or frozen dict of tuples)
        ovr_key = None
        if override_pool_kernels:
            ovr_key = tuple(sorted(
                (k, tuple(v)) for k, v in override_pool_kernels.items()
            ))
        cur_ovr = getattr(self, '_override_pool_key', None)
        if spacing == self.spacing and ovr_key == cur_ovr:
            return

        scales = [self.scale * (2 ** i) for i in range(self.n_downsample)]
        steps_array = self._compute_steps_array(
            spacing, scales, self.n_downsample,
            override_pool_kernels=override_pool_kernels,
        )

        cache = {}  # ephemeral within-call dedup (discarded after update)
        self.encoder.update_spacing(
            steps_array, _cache=cache,
            override_pool_kernels=override_pool_kernels,
        )

        # Compute decoder upsample overrides from the actual encoder pool kernels.
        # Decoder block i upsamples from level (n-i) to (n-i-1), reversing pool at
        # encoder level (n-1-i). With targeted pooling, the upsample factor must
        # match the actual pool kernel, not the standard floor(scale/step).
        override_upsample = None
        if override_pool_kernels:
            override_upsample = {}
            for enc_level, kernel in override_pool_kernels.items():
                # Encoder level enc_level maps to decoder block (n-1-enc_level)
                dec_idx = self.n_downsample - 1 - enc_level
                override_upsample[dec_idx] = kernel
        self.decoder.update_spacing(
            steps_array[::-1][1:], _cache=cache,
            override_upsample_factors=override_upsample,
        )

        self.spacing = spacing
        self._override_pool_key = ovr_key
        self._pool_kernel_sizes = [p.kernel_size for p in self.encoder.down_pool]

    def __call__(self, x, *, spacing=None, deterministic=True, use_running_average=False):
        """Forward pass.

        Parameters
        ----------
        x : jax.Array
            Shape (batch, channels, D, H, W).
        spacing : tuple or None
            Optional new spacing. If provided, calls update_spacing() before
            the forward pass. For JIT'd usage, call update_spacing() before
            the JIT boundary instead.
        deterministic : bool
            If True, disable dropout.
        use_running_average : bool
            If True, use running BN stats.

        Returns
        -------
        jax.Array or list of jax.Array
            Segmentation logits.
        """
        if spacing is not None:
            self.update_spacing(spacing)
        # Compute and apply padding
        pad = self._compute_padding(x.shape[-3:])
        original_shape = x.shape[-3:]
        # Pad: (D_before, D_after, H_before, H_after, W_before, W_after)
        x = jnp.pad(x, (
            (0, 0), (0, 0),  # batch, channels
            (pad[0], 0), (pad[1], 0), (pad[2], 0),  # spatial
        ))

        # Encoder
        encoder_features = self.encoder(x, deterministic=deterministic, use_running_average=use_running_average)

        # Decoder
        decoder_out = self.decoder(
            encoder_features[-1], encoder_features,
            deterministic=deterministic, use_running_average=use_running_average,
        )

        if self.deep_supervision and isinstance(decoder_out, tuple):
            x, ds_outputs = decoder_out

            # Main output projection
            w_mat = _build_sc_weight_matrix_fast(
                self.out_weight[...],
                self._out_sc_src[...], self._out_sc_dst[...], self._out_sc_alpha[...],
                self._out_out_dim, self._out_in_dim,
            )
            x = jnp.moveaxis(x, 1, -1)  # (B, D, H, W, C)
            x = x @ w_mat.astype(x.dtype).T
            x = jnp.moveaxis(x, -1, 1)  # (B, n_classes, D, H, W)

            bias = self.bias[...].astype(x.dtype).reshape(-1, 1, 1, 1)
            x = x + bias

            # Remove padding
            x = x[..., pad[0]:, pad[1]:, pad[2]:]

            final_outputs = []
            for i, ds_out in enumerate(ds_outputs):
                ds_out = ds_out + bias
                scale_factor = 2 ** (self.n_downsample - 1 - i)
                scaled_pad = [max(0, p // scale_factor) for p in pad]
                ds_out = ds_out[..., scaled_pad[0]:, scaled_pad[1]:, scaled_pad[2]:]
                final_outputs.append(ds_out)
            final_outputs.append(x)
            return final_outputs
        else:
            x = decoder_out

            # Output projection
            w_mat = _build_sc_weight_matrix_fast(
                self.out_weight[...],
                self._out_sc_src[...], self._out_sc_dst[...], self._out_sc_alpha[...],
                self._out_out_dim, self._out_in_dim,
            )
            x = jnp.moveaxis(x, 1, -1)
            x = x @ w_mat.astype(x.dtype).T
            x = jnp.moveaxis(x, -1, 1)

            bias = self.bias[...].astype(x.dtype).reshape(-1, 1, 1, 1)
            x = x + bias

            # Remove padding
            x = x[..., pad[0]:, pad[1]:, pad[2]:]

            return x


def create_model(n_classes: int, spacing: tuple = (1.0, 1.0, 1.0), *, rngs: nnx.Rngs, **kwargs) -> E3nnUNet:
    """Factory function to create a JAX E3nnUNet model."""
    return E3nnUNet(n_classes=n_classes, spacing=spacing, rngs=rngs, **kwargs)
