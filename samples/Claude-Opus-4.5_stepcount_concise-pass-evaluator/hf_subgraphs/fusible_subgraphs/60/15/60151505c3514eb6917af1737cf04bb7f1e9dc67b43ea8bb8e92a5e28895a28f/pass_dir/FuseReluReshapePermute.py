import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern to match:
    - relu(in_1, inplace=True)
    - in_0.reshape(B, 256, -1) -> tmp_1
    - relu_out.reshape(B, 256, -1) -> tmp_2
    - tmp_2.permute(0, 2, 1) -> tmp_3
    - return (tmp_3, tmp_1)
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0.reshape(in_0.shape[0], 256, -1)
    tmp_2 = tmp_0.reshape(in_0.shape[0], 256, -1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return (tmp_3, tmp_1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_relu_transpose_kernel(
    in_ptr,           # Input pointer for relu+permute
    out_ptr,          # Output pointer (transposed)
    B,                # Batch size
    C,                # Channels (256)
    spatial,          # Spatial dimension (H*W)
    in_stride_b,      # Strides
    in_stride_c,
    in_stride_s,
    out_stride_b,
    out_stride_s,
    out_stride_c,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel that applies ReLU and transposes from [B, C, spatial] to [B, spatial, C]
    Each program handles a tile of [BLOCK_S, BLOCK_C]
    """
    pid_b = tl.program_id(0)  # batch index
    pid_s = tl.program_id(1)  # spatial tile
    pid_c = tl.program_id(2)  # channel tile
    
    # Compute offsets
    off_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    off_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Mask for bounds
    mask_s = off_s < spatial
    mask_c = off_c < C
    mask = mask_s[:, None] & mask_c[None, :]
    
    # Input: [B, C, spatial] layout
    # We read from position [b, c, s]
    in_idx = pid_b * in_stride_b + off_c[None, :] * in_stride_c + off_s[:, None] * in_stride_s
    
    # Load and apply relu
    x = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    
    # Output: [B, spatial, C] layout
    # We write to position [b, s, c]
    out_idx = pid_b * out_stride_b + off_s[:, None] * out_stride_s + off_c[None, :] * out_stride_c
    
    tl.store(out_ptr + out_idx, out, mask=mask)


@torch.fx.wrap
def fused_relu_reshape_permute(in_0, in_1):
    """
    Optimized implementation:
    1. Reshape in_0 to [B, 256, spatial] (view operation)
    2. Apply relu to in_1 and write directly to transposed output [B, spatial, 256]
    """
    B = in_0.shape[0]
    C = in_0.shape[1]  # 256
    H = in_0.shape[2]
    W = in_0.shape[3]
    spatial = H * W
    
    # tmp_1: Simple reshape (view operation - no data copy)
    tmp_1 = in_0.reshape(B, C, spatial)
    
    # tmp_3: Fused relu + reshape + permute
    # Output shape: [B, spatial, C]
    out_permuted = torch.empty((B, spatial, C), dtype=in_1.dtype, device=in_1.device)
    
    # Input is [B, C, H, W], we treat it as [B, C, spatial]
    in_flat = in_1.reshape(B, C, spatial)
    
    # Compute strides
    in_stride_b = C * spatial
    in_stride_c = spatial
    in_stride_s = 1
    out_stride_b = spatial * C
    out_stride_s = C
    out_stride_c = 1
    
    # Block sizes
    BLOCK_S = 32
    BLOCK_C = 32
    
    # Grid
    grid = (B, triton.cdiv(spatial, BLOCK_S), triton.cdiv(C, BLOCK_C))
    
    fused_relu_transpose_kernel[grid](
        in_flat,
        out_permuted,
        B,
        C,
        spatial,
        in_stride_b,
        in_stride_c,
        in_stride_s,
        out_stride_b,
        out_stride_s,
        out_stride_c,
        BLOCK_S=BLOCK_S,
        BLOCK_C=BLOCK_C,
    )
    
    return (out_permuted, tmp_1)


def replacement_func():
    return fused_relu_reshape_permute