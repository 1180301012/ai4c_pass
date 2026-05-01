import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_bmm_kernel(
    softmax_in_ptr, in_2_ptr, out_ptr,
    B, N, P, R,
    stride_softmax_in_batch, stride_softmax_in_n, stride_softmax_in_p,
    stride_in_2_batch, stride_in_2_p, stride_in_2_r,
    stride_out_batch, stride_out_n, stride_out_r,
    BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr
):
    pid = tl.program_id(0)
    batch = pid // (N * R)
    n = (pid % (N * R)) // R
    r = pid % R

    # Load softmax input for current batch and query position
    softmax_in = tl.load(
        softmax_in_ptr + batch * stride_softmax_in_batch + n * stride_softmax_in_n + tl.arange(0, P) * stride_softmax_in_p,
        mask=tl.arange(0, P) < P,
        other=0.0
    )

    # Compute softmax in float32 for precision
    softmax_in_fp32 = tl.cast(softmax_in, tl.float32)
    max_val = tl.max(softmax_in_fp32, axis=0)
    exp_val = tl.exp(softmax_in_fp32 - max_val)
    sum_val = tl.sum(exp_val, axis=0)
    softmax_out = exp_val / sum_val

    # Load value input for current batch and key position
    in_2_vals = tl.load(
        in_2_ptr + batch * stride_in_2_batch + tl.arange(0, P) * stride_in_2_p + r * stride_in_2_r,
        mask=tl.arange(0, P) < P,
        other=0.0
    )

    # Compute dot product
    out_val = tl.dot(softmax_out, in_2_vals)

    # Store output
    tl.store(
        out_ptr + batch * stride_out_batch + n * stride_out_n + r * stride_out_r,
        out_val
    )

@torch.fx.wrap
def fused_softmax_bmm(softmax_in, in_2):
    B, N, P = softmax_in.shape
    _, _, R = in_2.shape
    out = torch.empty((B, N, R), dtype=softmax_in.dtype, device=softmax_in.device)

    # Get strides
    _, stride_n, stride_p = softmax_in.stride()
    _, stride_p_in2, stride_r = in_2.stride()
    _, stride_n_out, stride_r_out = out.stride()

    # Launch configuration
    BLOCK_N = 64
    BLOCK_P = 64
    BLOCK_R = 64
    grid = (B * N * R,)

    fused_softmax_bmm_kernel[grid](
        softmax_in, in_2, out,
        B, N, P, R,
        softmax_in.stride(0), stride_n, stride_p,
        in_2.stride(0), stride_p_in2, stride_r,
        out.stride(0), stride_n_out, stride_r_out,
        BLOCK_N, BLOCK_P, BLOCK_R
    )
    return out

def pattern(softmax_in, in_2):
    tmp_1 = torch.nn.functional.softmax(softmax_in, dim=-1)
    return torch.bmm(tmp_1, in_2)

def replacement_args(softmax_in, in_2):
    return (softmax_in, in_2)

def replacement_func():
    return fused_softmax_bmm