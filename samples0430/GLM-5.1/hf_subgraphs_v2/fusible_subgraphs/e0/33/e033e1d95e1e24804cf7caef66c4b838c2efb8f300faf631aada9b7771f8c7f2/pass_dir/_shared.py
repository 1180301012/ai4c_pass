import torch
import triton
import triton.language as tl


# =============================================================================
# Triton Kernel: mean(dim=-2, keepdim=True)
# Input: [N, M, K], Output: [N, 1, K] (reduce over M)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256}),
        triton.Config({'BLOCK_M': 512}),
        triton.Config({'BLOCK_M': 1024}),
        triton.Config({'BLOCK_M': 2048}),
        triton.Config({'BLOCK_M': 4096}),
    ],
    key=['M'],
)
@triton.jit
def mean_dim_neg2_keepdim_kernel(
    input_ptr, output_ptr,
    N, M, K,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    n_idx = pid // K
    k_idx = pid % K

    # Accumulate in float32 for precision
    acc = 0.0

    # Base offset for this (n, k) slice in input [N, M, K]
    # input[n, m, k] = input_ptr + n * M * K + m * K + k
    base_offset = n_idx * M * K + k_idx

    # Loop over M in chunks of BLOCK_M
    for m_start in range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        mask = m_offsets < M
        # Load values at input[n, m, k] for varying m
        values = tl.load(input_ptr + base_offset + m_offsets * K, mask=mask, other=0.0)
        acc = acc + tl.sum(values.to(tl.float32))

    # Divide by M
    result = acc / M

    # Store output at [n, 0, k] in keepdim shape [N, 1, K]
    # output[n, 0, k] = output_ptr + n * K + k
    output_offset = n_idx * K + k_idx
    tl.store(output_ptr + output_offset, result)


@torch.fx.wrap
def mean_neg2_keepdim_impl(x):
    N, M, K = x.shape
    output = torch.empty((N, 1, K), dtype=x.dtype, device=x.device)
    grid = (N * K,)
    mean_dim_neg2_keepdim_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        N=N, M=M, K=K,
    )
    return output


# =============================================================================
# Triton Kernel: conv2d 1x1 (batched matmul + bias)
# Weight: [C_OUT, C_IN, 1, 1] -> accessed as [C_OUT, C_IN]
# Input: [N, C_IN, H, W] -> accessed as [N, C_IN, HW]
# Output: [N, C_OUT, H, W] -> accessed as [N, C_OUT, HW]
# =============================================================================

@triton.autotune(
    configs=[
        # Small BLOCK_M for better parallelism on small batch sizes
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        # Medium BLOCK_M for balanced performance
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        # Large BLOCK_M for maximum tl.dot efficiency on large batch sizes
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
    ],
    key=['C_OUT', 'C_IN', 'HW'],
)
@triton.jit
def conv1x1_matmul_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N_BATCH, C_OUT, C_IN, HW,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)  # batch dimension
    pid_m = tl.program_id(0)  # output channel tile
    pid_n = tl.program_id(1)  # spatial position tile

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator [BLOCK_M, BLOCK_N] in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (input channels) dimension
    for k_start in range(0, C_IN, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)

        # Load weight tile: [BLOCK_M, BLOCK_K]
        # weight[c_out, c_in] = weight_ptr + c_out * C_IN + c_in
        w_ptrs = weight_ptr + m_off[:, None] * C_IN + k_off[None, :]
        w_mask = (m_off[:, None] < C_OUT) & (k_off[None, :] < C_IN)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Load input tile: [BLOCK_K, BLOCK_N] for this batch element
        # input[b, c_in, hw] = input_ptr + b * C_IN * HW + c_in * HW + hw
        i_ptrs = input_ptr + pid_b * C_IN * HW + k_off[:, None] * HW + n_off[None, :]
        i_mask = (k_off[:, None] < C_IN) & (n_off[None, :] < HW)
        i = tl.load(i_ptrs, mask=i_mask, other=0.0)

        # Matrix multiply using tl.dot (Tensor Core accelerated)
        acc += tl.dot(w, i, allow_tf32=True)

    # Add bias: [BLOCK_M] broadcasted to [BLOCK_M, BLOCK_N]
    b_ptrs = bias_ptr + m_off
    b_mask = m_off < C_OUT
    bias = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
    acc += bias[:, None]

    # Store output: [N_BATCH, C_OUT, HW] -> written as [N_BATCH, C_OUT, H, W]
    # output[b, c_out, hw] = output_ptr + b * C_OUT * HW + c_out * HW + hw
    o_ptrs = output_ptr + pid_b * C_OUT * HW + m_off[:, None] * HW + n_off[None, :]
    o_mask = (m_off[:, None] < C_OUT) & (n_off[None, :] < HW)
    tl.store(o_ptrs, acc, mask=o_mask)


@torch.fx.wrap
def conv1x1_impl(bias, weight, input):
    N_BATCH = input.shape[0]
    C_IN = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    HW = H * W
    C_OUT = weight.shape[0]
    # Allocate output as 4D tensor [N, C_OUT, H, W] to match conv2d output format
    output = torch.empty((N_BATCH, C_OUT, H, W), dtype=input.dtype, device=input.device)
    
    num_m_blocks = (C_OUT + 63) // 64  # will be adjusted by autotune
    # Grid dimensions depend on autotuned BLOCK sizes, but we use a fixed grid
    # and let the kernel handle masking for edge tiles
    # We need to calculate grid based on the selected config's BLOCK sizes
    # Triton handles this via the autotune mechanism
    
    # For now, use a conservative grid that covers all tiles
    # The actual BLOCK sizes will be determined by autotuning
    grid = lambda META: (
        (C_OUT + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (HW + META['BLOCK_N'] - 1) // META['BLOCK_N'],
        N_BATCH,
    )
    
    conv1x1_matmul_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N_BATCH=N_BATCH, C_OUT=C_OUT, C_IN=C_IN, HW=HW,
    )
    return output


# =============================================================================
# Dispatch wrapper (shared across all passes for replacement_func_limit)
# =============================================================================

@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "mean_neg2_keepdim":
        return mean_neg2_keepdim_impl(args[0])
    elif route == "conv1x1":
        return conv1x1_impl(args[0], args[1], args[2])
    else:
        raise ValueError(f"Unknown route: {route}")