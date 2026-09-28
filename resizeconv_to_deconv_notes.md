<!-- Jude: Created -->
# ResizeConv to Deconv Notes

## Conversation Summary

This note captures the explanation of how `ResizeConvolutionToDeconvolution` converts a Resize + Conv pattern into ConvTranspose in:

- [build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py](build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py)

## Target Class

- Class: `ResizeConvolutionToDeconvolution(Transformation)`
- Purpose: Replace nearest-neighbor upsample + same-padded convolution with an equivalent deconvolution (`ConvTranspose`) using weight convolution.

## High-Level Conversion Flow

1. Scan graph nodes and find `Resize` nodes.
2. Check whether Resize output is consumed by a single `Conv` node.
3. Validate supported constraints (mode/group/dims/dilation/scale assumptions).
4. Read Conv weights (either initializer or QONNX `Quant`-produced weights).
5. Build deconvolution weights using `_weight_convolution`.
6. Resolve padding (including `auto_pad` conversion to explicit `pads`).
7. Handle quantization behavior:
   - Re-quantize to maintain bit width, or
   - Recompute smallest safe bit width.
8. Create a new `ConvTranspose` node.
9. Preserve bias input and related initializers.
10. Insert `ConvTranspose`, remove old `Resize` and `Conv` nodes.

## Key Details from Implementation

### 1) Pattern detection

The transformation iterates through graph nodes, selecting `Resize` nodes and collecting consumers.

- Skips if no consumers.
- Skips if multiple consumers include a `Conv` (currently unsupported).
- Only proceeds when the first consumer is `Conv`.

### 2) Supported-only checks

The conversion is intentionally conservative and warns/skips when unsupported:

- Resize mode must be `nearest`.
- Conv `group` must be 1.
- 2D NCHW tensors only (`len(idim) == len(odim) == 4`).
- Same-padded conv assumption (`ifm_dim_h == ofm_dim_h` and `ifm_dim_w == ofm_dim_w`).
- Dilation must be `[1, 1]`.
- Resize scale only on spatial dims (N and C scales must be 1).
- Spatial scale must be square (`scale_h == scale_w`).

### 3) Weight loading and quant input support

Weights are loaded from one of two paths:

- Direct Conv initializer (`W_conv` with shape `(OC, IC, KH, KW)`), or
- QONNX `Quant` producer inputs:
  - Weight tensor,
  - Scale,
  - Zero-point,
  - Bit width,
  - Plus signed/narrow/rounding attributes.

If weight producer is not `Quant`, conversion is skipped.

### 4) Weight convolution algorithm

`_weight_convolution(cnv_weights, scale)` constructs deconv weights by summing rotated Conv kernels over all sub-pixel offsets.

- Input shape: `(OC, IC, KH, KW)`
- Output shape: `(IC, OC, KH + scale - 1, KW + scale - 1)`
- Kernel is rotated by 180 degrees (`np.rot90(..., 2, [0, 1])`) before accumulation.

This is based on the referenced Colbert et al. approach in the docstring.

### 5) Quantization behavior

If Conv weights came through `Quant`:

- `maintain_bit_width=True`:
  - Re-quantize transformed weights with original quant params.
  - Warn if clipping mismatch appears.

- `maintain_bit_width=False`:
  - Convert to integer domain using scale/zero-point.
  - Apply configured rounding mode.
  - Find smallest representable datatype and update bit width if needed.
  - Keep signedness consistent.

### 6) Node rewrite

The new `ConvTranspose` is created with:

- Inputs: original Resize input + transformed weight (+ bias if present)
- Outputs: original Conv outputs
- `kernel_shape`: `[KH + scale - 1, KW + scale - 1]`
- `strides`: `[scale, scale]`
- `pads`: resolved explicit pads
- `group`: original group
- `dilations`: original dilation

Then:

- New deconv weights are written as initializer.
- Quant initializers (`scale`, `zero-point`, `bitwidth`) are updated if needed.
- Bias initializer is preserved when required.
- Old `Resize` and `Conv` nodes are removed.

## Practical Interpretation

This transformation mathematically folds:

- nearest-neighbor spatial upsampling
- followed by same-padded convolution

into a single transposed convolution that should produce equivalent behavior under supported conditions.

## Current Limitations

The implementation currently does **not** support:

- Bilinear or bicubic Resize
- Grouped convolutions (`group > 1`)
- Non-2D convolution cases
- Dilation other than `[1,1]`
- Non-square spatial upsampling
- Resize with multiple Conv-related consumers
- Non-QONNX-Quant producer types for non-initializer weights

## ConvTranspose Subset Characterization

This section describes the subset of ConvTranspose parameter space reachable by converting
Resize + Conv with `ResizeConvolutionToDeconvolution`.

### 1) Structural constraints (reachable architecture subset)

A ConvTranspose is reachable only if it corresponds to a source pattern that satisfies all
implemented checks:

- Resize mode is `nearest`
- 2D NCHW tensors only
- Conv group is 1
- Conv dilation is `[1, 1]`
- Resize scales satisfy `(s_n, s_c, s_h, s_w) = (1, 1, s, s)`
- Resize output has a single supported Conv consumer path
- Conv side satisfies same-padded behavior in the checked case

Therefore, generated ConvTranspose nodes are constrained to:

- `stride = [s, s]`
- `group = 1`
- `dilation = [1, 1]`
- `kernel_shape = [k_h + s - 1, k_w + s - 1]`
- `pads` inherited from Conv (explicitly represented)

### 2) Deterministic parameter mapping

Given Conv kernel size `(k_h, k_w)` and resize factor `s`:

- Transposed-convolution stride is fixed by scale: `[s, s]`
- Transposed-convolution kernel size is fixed: `[k_h + s - 1, k_w + s - 1]`
- Other operator attributes (`group`, `dilation`, `pads`) are fixed or inherited as above

This means many ConvTranspose attributes are not free variables in this conversion.

### 3) Weight-space subset (main restriction)

For each output-input channel pair, reachable deconvolution weights are generated by overlap-adding
shifted 180-degree-rotated Conv kernels:

$$
W_{tconv} = \sum_{i=0}^{s-1}\sum_{j=0}^{s-1}\operatorname{shift}_{i,j}(\operatorname{rot180}(W_{conv}))
$$

Equivalent interpretation:

- `W_tconv` is the full convolution of `rot180(W_conv)` with an `s x s` all-ones kernel
- Reachable weights form a linear image of Conv weights, not the full unconstrained ConvTranspose space

### 4) Degrees of freedom

For one channel pair:

- Conv parameter count: `k_h * k_w`
- ConvTranspose tensor size after conversion: `(k_h + s - 1) * (k_w + s - 1)`
- Reachable DOF is at most `k_h * k_w`

So for `s > 1`, the generated ConvTranspose weights occupy a strict lower-dimensional subset of all
possible ConvTranspose weights.

Example (`k_h = k_w = 3`):

- `s = 2`: unconstrained ConvTranspose has `4 x 4 = 16` params, reachable subset has at most 9 DOF
- `s = 3`: unconstrained ConvTranspose has `5 x 5 = 25` params, reachable subset has at most 9 DOF

### 5) Practical conclusion

The conversion produces a strict subset of ConvTranspose parameter space:

- Architecture-level subset (attribute constraints)
- Weight-level subset (linear structure induced by weight convolution)

So not every ConvTranspose can be represented as a converted Resize + Conv, even when output shapes match.

## Worked Parameter Space (4 Configurations)

Below is a simple valid ResizeConv parameter space (all configurations satisfy the current conversion checks),
and the resulting generated ConvTranspose parameter space.

Assumptions shared across all 4 cases:

- Resize mode: nearest
- Scales: `(s_n, s_c, s_h, s_w) = (1, 1, s, s)`
- Conv group: 1
- Conv dilation: `[1, 1]`
- Same-padded behavior on Conv side

### Input ResizeConv space

| Config | Conv kernel `(k_h, k_w)` | Resize scale `s` | Conv stride `(h,w)` | Conv pads `(top,left,bottom,right)` |
|---|---:|---:|---:|---|
| C1 | (3, 3) | 2 | (1, 1) | (1, 1, 1, 1) |
| C2 | (5, 5) | 2 | (1, 1) | (2, 2, 2, 2) |
| C3 | (3, 3) | 3 | (1, 1) | (1, 1, 1, 1) |
| C4 | (1, 1) | 4 | (1, 1) | (0, 0, 0, 0) |

### Generated ConvTranspose space for the 4 configurations

Mapping rules used:

- `stride_tconv = (s, s)`
- `kernel_tconv = (k_h + s - 1, k_w + s - 1)`
- `group_tconv = 1`
- `dilation_tconv = (1, 1)`
- `pads_tconv = pads_conv` (after explicit normalization)

| Config | Generated `kernel_shape` | Generated `strides` | Generated `pads` | `group` | `dilation` |
|---|---:|---:|---|---:|---:|
| C1 | (4, 4) | (2, 2) | (1, 1, 1, 1) | 1 | (1, 1) |
| C2 | (6, 6) | (2, 2) | (2, 2, 2, 2) | 1 | (1, 1) |
| C3 | (5, 5) | (3, 3) | (1, 1, 1, 1) | 1 | (1, 1) |
| C4 | (4, 4) | (4, 4) | (0, 0, 0, 0) | 1 | (1, 1) |

### Weight-space size comparison (per output-input channel pair)

`Reachable DOF` is upper-bounded by Conv kernel parameter count `k_h * k_w`, while unconstrained
ConvTranspose has `(k_h + s - 1) * (k_w + s - 1)` entries.

| Config | Conv params `k_h * k_w` | Unconstrained TCONV entries | Reachable DOF upper bound |
|---|---:|---:|---:|
| C1 | 9 | 16 | <= 9 |
| C2 | 25 | 36 | <= 25 |
| C3 | 9 | 25 | <= 9 |
| C4 | 1 | 16 | <= 1 |

This table is the concrete 4-point example of the strict subset claim: generated ConvTranspose weights
live in a constrained linear subspace induced by the conversion in
[build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py](build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py).

## Conversation Context Captured

- Workspace repository: [Xilinx/finn-examples](https://github.com/Xilinx/finn-examples)
- Current branch: `feature/espcn`
- Default branch: `main`
- Related repositories mentioned:
  - [Xilinx/finn](https://github.com/Xilinx/finn) (branch `feature/deconv`, PR: [[WIP] Optimized deconvolution HLS implementation](https://github.com/Xilinx/finn/pull/1407))
  - [fastmachinelearning/qonnx](https://github.com/fastmachinelearning/qonnx)
- Active file discussed:
  - [build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py](build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py)

## Related Local Files

- [resizeconv_to_deconv_notes.md](resizeconv_to_deconv_notes.md)
- [build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py](build/finn/deps/qonnx/src/qonnx/transformation/resize_conv_to_deconv.py)

## Glossary

- ResizeConv: Resize op followed by Conv op.
- Deconv: Common shorthand here for `ConvTranspose`.
- IFM/OFM: Input/output feature maps.
- QONNX Quant: QONNX quantization operator used as weight producer.
