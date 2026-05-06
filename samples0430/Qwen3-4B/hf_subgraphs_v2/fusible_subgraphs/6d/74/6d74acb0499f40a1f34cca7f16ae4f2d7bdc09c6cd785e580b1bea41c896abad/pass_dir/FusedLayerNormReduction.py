import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def layer_norm_kernel(
    x_ptr: tl.pointer[tl.float32],
    out_ptr: tl.pointer[tl.float32],
    n_batches: tl.int32,
    n_seq: tl.int32,
    n_features: tl.int32,
    scale_ptr: tl.pointer[tl.float32],
    bias_ptr: tl.pointer[tl.float32],
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_features
    x_vals = tl.load(x_ptr + block_start, mask=mask, other=0.0)
    sum_x = tl.zeros(tl.float32)
    sum_sq = tl.zeros(tl.float32)
    for i in range(BLOCK_SIZE):
        if mask[i]:
            x_val = x_vals[i]
            sum_x += x_val
            sum_sq += x_val * x_val
    mean = sum_x / n_features
    var = (sum_sq - sum_x * sum_x / n_features) / n_features
    std = tl.sqrt(var + eps)
    normalized = (x_vals - mean) / std
    out_val = tl.load(scale_ptr) * normalized + tl.load(bias_ptr)
    tl.store(out_ptr + block_start, out_val, mask=mask)

@torch.fx.wrap
def kernel_wrapper(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-07,
):
    x = x.float()
    n_batches = x.shape[0]
    n_seq = x.shape[1]
    n_features = x.shape[2]
    out = torch.empty_like(x, dtype=torch.float32)
    layer_norm_kernel[(n_batches, n_seq)](
        x_ptr=x,
        out_ptr=out,
        n_batches=n_batches,
        n_seq=n_seq,
        n_features=n_features,
        scale_ptr=scale,
        bias_ptr=bias,
        eps=eps,
        BLOCK_SIZE=128,
    )
    return out
def replacement_func():
    return kernel_wrapper