import torch
import triton
import triton.language as tl


def pattern(bias, weight, conv_input, cat_input):
    conv = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv], dim=0)
    summed = stacked.sum(dim=0)
    result = torch.cat([summed, cat_input], 1)
    return (result,)


def replacement_args(bias, weight, conv_input, cat_input):
    return (bias, weight, conv_input, cat_input)


# ---- Conv kernel: 1x1 conv2d as matmul with row-based grid ----
# Grid: (num_m_conv, N*H) where each program handles BLOCK_M output channels
# and BLOCK_N=W spatial positions within one (n,h) row
# This gives contiguous W-dimension access for both input and output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=8),
    ],
    key=['C_out', 'C_in', 'N', 'H', 'W'],
)
@triton.jit
def conv_1x1_kernel(
    conv_input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C_in, C_out, H, W,
    s_in_n, s_in_c, s_in_h, s_in_w,
    s_w_co, s_w_ci,
    s_out_n, s_out_c, s_out_h, s_out_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nh = tl.program_id(1)

    # Output channel offsets
    c_start = pid_m * BLOCK_M
    c_off = c_start + tl.arange(0, BLOCK_M)
    mask_c = c_off < C_out

    # Decode (n, h) from pid_nh
    n_idx = pid_nh // H
    h_idx = pid_nh % H
    mask_n = n_idx < N

    # W offsets (contiguous dimension)
    w_idx = tl.arange(0, BLOCK_N)
    mask_w = w_idx < W

    # Accumulator in float32 for accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matmul loop over C_in
    for k_start in range(0, C_in, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_off < C_in

        # Load weight [BLOCK_M, BLOCK_K] - c_in dimension is contiguous (stride s_w_ci=1)
        w_ptrs = weight_ptr + c_off[:, None] * s_w_co + k_off[None, :] * s_w_ci
        w_vals = tl.load(w_ptrs, mask=mask_c[:, None] & mask_k[None, :], other=0.0)

        # Load input [BLOCK_K, BLOCK_N] - W dimension is contiguous (stride s_in_w=1)
        # input[n, c_in, h, w] = n*s_in_n + c_in*s_in_c + h*s_in_h + w*s_in_w
        in_ptrs = conv_input_ptr + n_idx * s_in_n + k_off[:, None] * s_in_c + h_idx * s_in_h + w_idx[None, :] * s_in_w
        in_vals = tl.load(in_ptrs, mask=mask_k[:, None] & mask_w[None, :] & mask_n, other=0.0)

        acc += tl.dot(w_vals, in_vals)

    # Add bias for each output channel
    b_ptrs = bias_ptr + c_off
    b_vals = tl.load(b_ptrs, mask=mask_c, other=0.0)
    acc += b_vals[:, None]

    # Store output - W dimension is contiguous (stride s_out_w=1)
    out_ptrs = output_ptr + n_idx * s_out_n + c_off[:, None] * s_out_c + h_idx * s_out_h + w_idx[None, :] * s_out_w
    tl.store(out_ptrs, acc, mask=mask_c[:, None] & mask_w[None, :] & mask_n)


# ---- Cat kernel: copy cat_input channels to output channels [C_out, C_out+C_cat) ----
@triton.jit
def cat_copy_kernel(
    cat_input_ptr, output_ptr,
    N, C_cat, H, W, C_out,
    s_cat_n, s_cat_c, s_cat_h, s_cat_w,
    s_out_n, s_out_c, s_out_h, s_out_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nh = tl.program_id(1)

    # Cat channel offsets
    cat_c_start = pid_m * BLOCK_M
    cat_c_off = cat_c_start + tl.arange(0, BLOCK_M)
    mask_cat_c = cat_c_off < C_cat

    # Output channel index = C_out + cat_channel_index
    out_c_off = C_out + cat_c_off

    # Decode (n, h) from pid_nh
    n_idx = pid_nh // H
    h_idx = pid_nh % H
    mask_n = n_idx < N

    # W offsets (contiguous dimension)
    w_idx = tl.arange(0, BLOCK_N)
    mask_w = w_idx < W

    # Load from cat_input - W dimension is contiguous
    cat_ptrs = cat_input_ptr + n_idx * s_cat_n + cat_c_off[:, None] * s_cat_c + h_idx * s_cat_h + w_idx[None, :] * s_cat_w
    cat_vals = tl.load(cat_ptrs, mask=mask_cat_c[:, None] & mask_w[None, :] & mask_n, other=0.0)

    # Store to output at channels [C_out, C_out+C_cat) - W dimension is contiguous
    out_ptrs = output_ptr + n_idx * s_out_n + out_c_off[:, None] * s_out_c + h_idx * s_out_h + w_idx[None, :] * s_out_w
    tl.store(out_ptrs, cat_vals, mask=mask_cat_c[:, None] & mask_w[None, :] & mask_n)


def conv_grid_fn(args):
    BLOCK_M = args['BLOCK_M']
    C_out = args['C_out']
    N = args['N']
    H = args['H']
    num_m = (C_out + BLOCK_M - 1) // BLOCK_M
    return (num_m, N * H)


def cat_grid_fn(BLOCK_M, BLOCK_N, C_cat, N, H):
    num_m = (C_cat + BLOCK_M - 1) // BLOCK_M
    return (num_m, N * H)


@torch.fx.wrap
def fused_conv2d_stack_sum_cat(bias, weight, conv_input, cat_input):
    N = conv_input.shape[0]
    C_in = conv_input.shape[1]
    H = conv_input.shape[2]
    W = conv_input.shape[3]
    C_out = weight.shape[0]
    C_cat = cat_input.shape[1]
    C_total = C_out + C_cat

    # Allocate output tensor
    output = torch.empty((N, C_total, H, W), dtype=conv_input.dtype, device=conv_input.device)

    # Get strides
    s_in = conv_input.stride()
    s_w = weight.stride()
    s_cat = cat_input.stride()
    s_out = output.stride()

    # Block sizes
    BLOCK_N = W  # Process full W dimension per program for contiguous access
    BLOCK_K = 32  # Will be overridden by autotune for conv kernel

    # Conv kernel grid
    total_nh = N * H
    num_m_conv = (C_out + 32 - 1) // 32  # Minimum BLOCK_M for grid calc, autotune will adjust
    conv_grid = (num_m_conv, total_nh)  # Autotune will override grid

    # Launch conv kernel with autotune grid function
    conv_1x1_kernel[conv_grid_fn](
        conv_input_ptr=conv_input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N, C_in=C_in, C_out=C_out, H=H, W=W,
        s_in_n=s_in[0], s_in_c=s_in[1], s_in_h=s_in[2], s_in_w=s_in[3],
        s_w_co=s_w[0], s_w_ci=s_w[1],
        s_out_n=s_out[0], s_out_c=s_out[1], s_out_h=s_out[2], s_out_w=s_out[3],
        BLOCK_N=BLOCK_N,
    )

    # Cat kernel - use large blocks for memory throughput
    BLOCK_M_CAT = 128
    cat_grid = cat_grid_fn(BLOCK_M_CAT, BLOCK_N, C_cat, N, H)
    cat_copy_kernel[cat_grid](
        cat_input_ptr=cat_input,
        output_ptr=output,
        N=N, C_cat=C_cat, H=H, W=W, C_out=C_out,
        s_cat_n=s_cat[0], s_cat_c=s_cat[1], s_cat_h=s_cat[2], s_cat_w=s_cat[3],
        s_out_n=s_out[0], s_out_c=s_out[1], s_out_h=s_out[2], s_out_w=s_out[3],
        BLOCK_M=BLOCK_M_CAT, BLOCK_N=BLOCK_N,
    )

    return output


def replacement_func():
    return fused_conv2d_stack_sum_cat