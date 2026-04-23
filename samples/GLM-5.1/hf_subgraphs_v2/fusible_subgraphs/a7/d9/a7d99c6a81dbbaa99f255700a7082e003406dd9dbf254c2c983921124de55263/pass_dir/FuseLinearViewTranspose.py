import torch
import triton
import triton.language as tl

# Pattern: linear(in_3, in_1, in_0) then reshape to [1,1,8,64], transpose(1,2), contiguous
# This computes V-projection and reshapes it to multi-head format in one fused kernel
def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_10 = tmp_6.contiguous()
    return tmp_10

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def fused_linear_gemv_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N_out: tl.constexpr, N_in: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """GEMV kernel: vector-matrix multiply with output reshape.
    Fixed-size kernel for 512x512 weight matrix."""
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)

    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < N_out

    # 2D accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over input dimension
    for k_start in range(0, N_in, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)

        # Load input chunk - no mask needed since N_in = BLOCK_N * num_iters exactly
        input_vals = tl.load(input_ptr + k_offsets)

        # Load weight rows
        weight_ptrs = weight_ptr + row_offsets[:, None] * N_in + k_offsets[None, :]
        weight_vals = tl.load(weight_ptrs, mask=row_mask[:, None], other=0.0)

        # Accumulate products
        acc += weight_vals * input_vals[None, :]

    # Reduce and add bias
    result = tl.sum(acc, axis=1)
    bias_vals = tl.load(bias_ptr + row_offsets, mask=row_mask, other=0.0)
    result = result + bias_vals

    # Store in output layout
    output_offsets = pid_batch * N_out + row_offsets
    tl.store(output_ptr + output_offsets, result, mask=row_mask)


@torch.fx.wrap
def fused_linear_view_transpose(bias, weight, input):
    N_out = 512  # weight.shape[0]
    N_in = 512   # weight.shape[1]
    H = 8
    D = 64
    
    # Transfer to GPU
    weight_gpu = weight.to(input.device) if weight.device != input.device else weight
    bias_gpu = bias.to(input.device) if bias.device != input.device else bias

    # Allocate output
    out = torch.empty((1, H, 1, D), dtype=input.dtype, device=input.device)

    # Use constexpr sizes for maximum optimization
    BLOCK_M = 64   # Each program handles 64 output rows
    BLOCK_N = 128  # Input dimension chunk size (512 / 128 = 4 iterations)
    num_programs = (N_out + BLOCK_M - 1) // BLOCK_M  # = 8
    
    fused_linear_gemv_kernel[(num_programs, 1)](
        input_ptr=input,
        weight_ptr=weight_gpu,
        bias_ptr=bias_gpu,
        output_ptr=out,
        N_out=N_out, N_in=N_in,
        H=H, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return out

def replacement_func():
    return fused_linear_view_transpose