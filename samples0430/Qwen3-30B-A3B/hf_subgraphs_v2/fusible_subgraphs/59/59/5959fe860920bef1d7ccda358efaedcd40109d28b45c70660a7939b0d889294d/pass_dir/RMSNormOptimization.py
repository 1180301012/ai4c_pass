import torch
import triton
import triton.language as tl

def pattern(in_2):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    return tmp_16

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    y_ptr,
    eps,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // seq_len
    seq = pid % seq_len
    start_idx = (batch * seq_len + seq) * hidden_size
    x = tl.load(x_ptr + start_idx, shape=(hidden_size,))
    sum_sq = 0.0
    for i in range(hidden_size):
        sum_sq += x[i] * x[i]
    norm = tl.sqrt(sum_sq / hidden_size + eps)
    for i in range(hidden_size):
        y = x[i] / norm
        tl.store(y_ptr + start_idx + i, y)

@torch.fx.wrap
def kernel_wrapper(in_2):
    batch_size, seq_len, hidden_size = in_2.shape
    input_fp32 = in_2.to(torch.float32)
    output = torch.empty_like(input_fp32, dtype=torch.bfloat16)
    BLOCK_SIZE = 128
    grid_size = batch_size * seq_len
    rmsnorm_kernel[(grid_size,)](
        input_fp32,
        output,
        1e-06,
        batch_size,
        seq_len,
        hidden_size,
        BLOCK_SIZE
    )
    return output

def replacement_func():
    return kernel_wrapper