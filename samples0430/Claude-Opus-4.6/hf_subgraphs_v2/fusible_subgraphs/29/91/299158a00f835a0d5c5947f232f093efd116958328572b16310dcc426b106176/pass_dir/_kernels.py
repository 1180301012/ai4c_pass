import torch
import triton
import triton.language as tl


@triton.jit
def linear_2out_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, K,
    stride_x_b,
    stride_w_n,
    stride_out_b,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Load input row
    x = tl.load(x_ptr + pid_b * stride_x_b + k_offsets, mask=k_mask, other=0.0).to(tl.float32)

    # Load weight row 0 and compute dot product
    w0 = tl.load(w_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
    dot0 = tl.sum(x * w0, axis=0)

    # Load weight row 1 and compute dot product
    w1 = tl.load(w_ptr + stride_w_n + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
    dot1 = tl.sum(x * w1, axis=0)

    # Add bias
    b0 = tl.load(b_ptr).to(tl.float32)
    b1 = tl.load(b_ptr + 1).to(tl.float32)

    result0 = dot0 + b0
    result1 = dot1 + b1

    tl.store(out_ptr + pid_b * stride_out_b, result0)
    tl.store(out_ptr + pid_b * stride_out_b + 1, result1)


@triton.jit
def mean_dim1_kernel(
    in_ptr, out_ptr,
    S, C,
    stride_in_b, stride_in_s,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    # Base pointer for this batch element + C offset
    base_ptr = in_ptr + pid_b * stride_in_b + c_offsets

    # Accumulate over S dimension with coalesced 1D loads
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for s in range(0, S):
        vals = tl.load(base_ptr + s * stride_in_s, mask=c_mask, other=0.0)
        acc += vals.to(tl.float32)

    # Compute mean
    result = acc / S

    # Store output
    out_base = out_ptr + pid_b * C + c_offsets
    tl.store(out_base, result, mask=c_mask)


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "linear":
        in_0, in_1, in_2 = args[0], args[1], args[2]
        B = in_2.shape[0]
        K = in_2.shape[1]
        N = in_1.shape[0]
        linear_out = torch.empty((B, N), dtype=in_2.dtype, device=in_2.device)
        linear_2out_kernel[(B,)](
            in_2, in_1, in_0, linear_out,
            B, K,
            in_2.stride(0),
            in_1.stride(0),
            linear_out.stride(0),
            BLOCK_K=512,
            num_warps=4,
        )
        return linear_out
    else:
        in_3 = args[0]
        B = in_3.shape[0]
        S = in_3.shape[1]
        C = in_3.shape[2]
        mean_out = torch.empty((B, C), dtype=in_3.dtype, device=in_3.device)
        mean_dim1_kernel[(B, (C + 511) // 512)](
            in_3, mean_out,
            S, C,
            in_3.stride(0), in_3.stride(1),
            BLOCK_C=512,
            num_warps=4,
            num_stages=3,
        )
        return mean_out