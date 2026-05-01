import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def linear_kernel(x_ptr, w_ptr, bias_ptr, out_ptr, n_rows, n_cols, n_k, BLOCK_K: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    if i >= n_rows or j >= n_cols:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, n_k, BLOCK_K):
        k_end = min(k + BLOCK_K, n_k)
        x_val = tl.load(x_ptr + i * n_k + k, mask=k_end > k, other=0.0)
        w_val = tl.load(w_ptr + j * n_k + k, mask=k_end > k, other=0.0)
        acc += x_val * w_val
    acc += tl.load(bias_ptr + j)
    tl.store(out_ptr + i * n_cols + j, acc)

@torch.fx.wrap
def linear_wrapper(in_2, in_1, in_0):
    batch_size = in_2.shape[0]
    seq_len = in_2.shape[1]
    in_features = in_2.shape[2]
    out_features = in_1.shape[0]
    out = torch.empty(batch_size, seq_len, out_features, device=in_2.device, dtype=in_2.dtype)
    num_rows = batch_size * seq_len
    num_cols = out_features
    linear_kernel[(num_rows, num_cols)](
        x_ptr=in_2,
        w_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=out,
        n_rows=num_rows,
        n_cols=num_cols,
        n_k=in_features,
        BLOCK_K=64
    )
    return out

def replacement_func():
    return linear_wrapper