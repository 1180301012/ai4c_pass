import torch
import triton
import triton.language as tl
import math


# ============================================================
# Triton Matmul Kernel (with bias fusion)
# Computes: C = A @ B_weight^T + bias
# A: [M, K], B_weight: [N, K], bias: [N], C: [M, N]
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BN': 32, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 32, 'BK': 32, 'GM': 8}, num_stages=5, num_warps=2),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 32, 'GM': 8}, num_stages=5, num_warps=2),
        triton.Config({'BM': 32, 'BN': 32, 'BK': 32, 'GM': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 32, 'BN': 128, 'BK': 32, 'GM': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BN': 32, 'BK': 32, 'GM': 4}, num_stages=4, num_warps=4),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=4),
        triton.Config({'BM': 64, 'BN': 32, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=4),
    ],
    key=['M_DIM', 'N_DIM', 'K_DIM'],
)
@triton.jit
def linear_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr, bias_ptr,
    # Dimensions
    M_DIM, N_DIM, K_DIM,
    # Strides for A: [M, K]
    stride_am, stride_ak,
    # Strides for B_weight: [N, K] - we read it transposed
    stride_bn_row, stride_bk_col,
    # Strides for C: [M, N]
    stride_cm, stride_cn,
    # Strides for bias: [N]
    stride_bias,
    # Block sizes
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
    # Output dtype for proper store
    DTYPE_OUT: tl.constexpr,
):
    """
    Matmul kernel: C = A @ B_weight^T + bias
    A is [M_DIM, K_DIM], B_weight is [N_DIM, K_DIM], bias is [N_DIM], C is [M_DIM, N_DIM]
    B_weight is stored as [N, K] and accessed transposed.
    """
    # Program ID and grid mapping (group-based for L2 cache optimization)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_DIM, BM)
    num_pid_n = tl.cdiv(N_DIM, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    group_size_m = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for A and B_weight tiles
    offs_am = pid_m * BM + tl.arange(0, BM)
    offs_bn = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # Accumulator in float32 for precision
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, tl.cdiv(K_DIM, BK)):
        k_offs = k_start * BK + offs_k
        
        # Load A tile: [BM, BK]
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + k_offs[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M_DIM) & (k_offs[None, :] < K_DIM)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        
        # Load B_weight tile: we need B_weight[n, k] values arranged as [BK, BN]
        # B_weight[n, k] = b_ptr + n * stride_bn_row + k * stride_bk_col
        # Tile: offs_bn[None, :] for n, k_offs[:, None] for k -> [BK, BN]
        b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn_row + k_offs[:, None] * stride_bk_col)
        b_mask = (offs_bn[None, :] < N_DIM) & (k_offs[:, None] < K_DIM)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        
        # Accumulate dot product
        accumulator += tl.dot(a, b, allow_tf32=False)

    # Add bias
    offs_bias = pid_n * BN + tl.arange(0, BN)
    bias_mask = offs_bias < N_DIM
    bias = tl.load(bias_ptr + offs_bias * stride_bias, mask=bias_mask, other=0.0).to(tl.float32)
    accumulator += bias[None, :]

    # Store result with proper dtype conversion
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M_DIM) & (offs_cn[None, :] < N_DIM)
    
    tl.store(c_ptrs, accumulator.to(DTYPE_OUT), mask=c_mask)


# ============================================================
# Triton Bilinear Interpolation Kernel
# Reads from [B, SEQ_LEN, OUT_FEATURES] (linear output)
# Computes permute(0,2,1) + reshape + bilinear interpolation
# Output: [B, OUT_FEATURES, H_OUT, W_OUT]
# Uses 1D grid for maximum flexibility
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 1, 'BLOCK_C': 16, 'BLOCK_OY': 8, 'BLOCK_OX': 8}, num_warps=4),
        triton.Config({'BLOCK_B': 1, 'BLOCK_C': 32, 'BLOCK_OY': 4, 'BLOCK_OX': 4}, num_warps=4),
        triton.Config({'BLOCK_B': 1, 'BLOCK_C': 8, 'BLOCK_OY': 16, 'BLOCK_OX': 16}, num_warps=8),
        triton.Config({'BLOCK_B': 1, 'BLOCK_C': 64, 'BLOCK_OY': 2, 'BLOCK_OX': 2}, num_warps=4),
        triton.Config({'BLOCK_B': 1, 'BLOCK_C': 128, 'BLOCK_OY': 1, 'BLOCK_OX': 1}, num_warps=4),
        triton.Config({'BLOCK_B': 2, 'BLOCK_C': 16, 'BLOCK_OY': 4, 'BLOCK_OX': 4}, num_warps=4),
        triton.Config({'BLOCK_B': 4, 'BLOCK_C': 8, 'BLOCK_OY': 4, 'BLOCK_OX': 4}, num_warps=4),
    ],
    key=['BATCH', 'CHANNELS', 'H_OUT', 'W_OUT'],
)
@triton.jit
def interpolate_kernel(
    # Pointers
    input_ptr, output_ptr,
    # Dimensions
    BATCH, SEQ_LEN, CHANNELS,
    H_IN, W_IN, H_OUT, W_OUT,
    # Strides for input: [B, SEQ_LEN, CHANNELS]
    stride_ib, stride_is, stride_ic,
    # Strides for output: [B, CHANNELS, H_OUT, W_OUT]
    stride_ob, stride_oc, stride_oh, stride_ow,
    # Block sizes
    BLOCK_B: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_OY: tl.constexpr, BLOCK_OX: tl.constexpr,
    # Output dtype
    DTYPE_OUT: tl.constexpr,
):
    """
    Bilinear interpolation kernel with permute+reshape fusion.
    Uses 1D grid, decomposes program ID into 4D tile indices.
    Grid dimensions are computed internally from constexpr block sizes.
    """
    # Compute grid dimensions using constexpr block sizes
    NUM_B = (BATCH + BLOCK_B - 1) // BLOCK_B
    NUM_C = (CHANNELS + BLOCK_C - 1) // BLOCK_C
    NUM_H = (H_OUT + BLOCK_OY - 1) // BLOCK_OY
    NUM_W = (W_OUT + BLOCK_OX - 1) // BLOCK_OX
    
    # 1D program ID -> decompose into 4D
    pid = tl.program_id(0)
    
    pid_w = pid % NUM_W
    pid_h = (pid // NUM_W) % NUM_H
    pid_c = (pid // (NUM_W * NUM_H)) % NUM_C
    pid_b = pid // (NUM_W * NUM_H * NUM_C)

    # Compute tile offsets
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_oy = pid_h * BLOCK_OY + tl.arange(0, BLOCK_OY)
    offs_ox = pid_w * BLOCK_OX + tl.arange(0, BLOCK_OX)

    # Masks
    mask_b = offs_b < BATCH
    mask_c = offs_c < CHANNELS
    mask_oy = offs_oy < H_OUT
    mask_ox = offs_ox < W_OUT
    mask_all = mask_b[:, None, None, None] & mask_c[None, :, None, None] & mask_oy[None, None, :, None] & mask_ox[None, None, None, :]

    # Compute interpolation source coordinates
    # align_corners=False: src = (dst + 0.5) * (src_size / dst_size) - 0.5
    scale_y = H_IN / H_OUT
    scale_x = W_IN / W_OUT

    # For each output (oy, ox), compute src_y and src_x
    src_y = (offs_oy[None, None, :, None] + 0.5) * scale_y - 0.5  # broadcast to [1,1,BLOCK_OY,1]
    src_x = (offs_ox[None, None, None, :] + 0.5) * scale_x - 0.5  # broadcast to [1,1,1,BLOCK_OX]

    # Compute floor indices (before clamping, for weight calculation)
    iy0_f = tl.floor(src_y)
    ix0_f = tl.floor(src_x)
    
    # Compute interpolation weights using unclamped floor values
    wy1 = src_y - iy0_f  # weight for iy1
    wy0 = 1.0 - wy1      # weight for iy0
    wx1 = src_x - ix0_f  # weight for ix1
    wx0 = 1.0 - wx1      # weight for ix0

    # Convert to int and clamp
    iy0 = tl.maximum(iy0_f.to(tl.int32), 0)
    iy1 = tl.minimum(iy0 + 1, H_IN - 1)
    ix0 = tl.maximum(ix0_f.to(tl.int32), 0)
    ix1 = tl.minimum(ix0 + 1, W_IN - 1)

    # Compute spatial indices: spatial_idx = iy * W_IN + ix
    spatial_00 = iy0 * W_IN + ix0
    spatial_01 = iy0 * W_IN + ix1
    spatial_10 = iy1 * W_IN + ix0
    spatial_11 = iy1 * W_IN + ix1

    # Read input values from linear output [B, SEQ_LEN, CHANNELS]
    def load_input(spatial_idx):
        ptrs = input_ptr + (
            offs_b[:, None, None, None] * stride_ib +
            spatial_idx * stride_is +
            offs_c[None, :, None, None] * stride_ic
        )
        load_mask = mask_b[:, None, None, None] & mask_c[None, :, None, None]
        return tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)

    v00 = load_input(spatial_00)
    v01 = load_input(spatial_01)
    v10 = load_input(spatial_10)
    v11 = load_input(spatial_11)

    # Bilinear interpolation in float32
    result = (
        wy0 * wx0 * v00 +
        wy0 * wx1 * v01 +
        wy1 * wx0 * v10 +
        wy1 * wx1 * v11
    )

    # Store output: [B, CHANNELS, H_OUT, W_OUT]
    out_ptrs = output_ptr + (
        offs_b[:, None, None, None] * stride_ob +
        offs_c[None, :, None, None] * stride_oc +
        offs_oy[None, None, :, None] * stride_oh +
        offs_ox[None, None, None, :] * stride_ow
    )
    tl.store(out_ptrs, result.to(DTYPE_OUT), mask=mask_all)


# ============================================================
# Dtype mapping helper
# ============================================================
_DTYPE_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


# ============================================================
# Wrapper functions (decorated with @torch.fx.wrap)
# ============================================================

@torch.fx.wrap
def triton_linear(bias, weight, input_tensor):
    """Compute linear = input_tensor @ weight.T + bias using Triton kernel."""
    B = input_tensor.shape[0]
    SEQ = input_tensor.shape[1]
    K = input_tensor.shape[2]
    M = weight.shape[0]
    
    # Allocate output [B, SEQ, M]
    output = torch.empty((B, SEQ, M), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Strides
    stride_am = input_tensor.stride(1)
    stride_ak = input_tensor.stride(2)
    stride_bn_row = weight.stride(0)
    stride_bk_col = weight.stride(1)
    stride_cm = output.stride(1)
    stride_cn = output.stride(2)
    stride_bias = bias.stride(0) if bias.dim() > 0 else 1
    
    M_DIM = B * SEQ
    N_DIM = M
    K_DIM = K
    
    DTYPE_OUT = _DTYPE_MAP[input_tensor.dtype]
    
    grid = lambda META: (
        triton.cdiv(M_DIM, META['BM']) * triton.cdiv(N_DIM, META['BN']),
    )
    
    linear_kernel[grid](
        a_ptr=input_tensor, b_ptr=weight, c_ptr=output, bias_ptr=bias,
        M_DIM=M_DIM, N_DIM=N_DIM, K_DIM=K_DIM,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bn_row=stride_bn_row, stride_bk_col=stride_bk_col,
        stride_cm=stride_cm, stride_cn=stride_cn,
        stride_bias=stride_bias,
        DTYPE_OUT=DTYPE_OUT,
    )
    
    return output


@torch.fx.wrap
def triton_interpolate(linear_output):
    """Compute permute(0,2,1) + reshape + bilinear interpolation."""
    B = linear_output.shape[0]
    SEQ = linear_output.shape[1]  # 256 = 16*16
    CHANNELS = linear_output.shape[2]  # 768
    
    H_IN = 16
    W_IN = 16
    H_OUT = 128
    W_OUT = 128
    
    # Allocate output [B, CHANNELS, H_OUT, W_OUT]
    output = torch.empty((B, CHANNELS, H_OUT, W_OUT), dtype=linear_output.dtype, device=linear_output.device)
    
    # Strides
    stride_ib = linear_output.stride(0)
    stride_is = linear_output.stride(1)
    stride_ic = linear_output.stride(2)
    stride_ob = output.stride(0)
    stride_oc = output.stride(1)
    stride_oh = output.stride(2)
    stride_ow = output.stride(3)
    
    DTYPE_OUT = _DTYPE_MAP[linear_output.dtype]
    
    # Launch kernel with 1D grid
    grid = lambda META: (
        triton.cdiv(B, META['BLOCK_B']) * 
        triton.cdiv(CHANNELS, META['BLOCK_C']) * 
        triton.cdiv(H_OUT, META['BLOCK_OY']) * 
        triton.cdiv(W_OUT, META['BLOCK_OX']),
    )
    
    # Compute grid decomposition info for the kernel
    # These are needed to decompose the 1D program ID into 4D indices
    
    interpolate_kernel[grid](
        input_ptr=linear_output, output_ptr=output,
        BATCH=B, SEQ_LEN=SEQ, CHANNELS=CHANNELS,
        H_IN=H_IN, W_IN=W_IN, H_OUT=H_OUT, W_OUT=W_OUT,
        stride_ib=stride_ib, stride_is=stride_is, stride_ic=stride_ic,
        stride_ob=stride_ob, stride_oc=stride_oc, stride_oh=stride_oh, stride_ow=stride_ow,
        DTYPE_OUT=DTYPE_OUT,
    )
    
    return output


@torch.fx.wrap
def fused_linear_permute_reshape_interpolate(in_0, in_1, in_2):
    """Fused implementation of linear + permute(0,2,1) + reshape + bilinear interpolate."""
    # Step 1: Compute linear (matmul + bias)
    linear_output = triton_linear(in_0, in_1, in_2)
    
    # Step 2: Compute permute+reshape+interpolate
    result = triton_interpolate(linear_output)
    
    return result