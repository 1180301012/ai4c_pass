import torch
import triton
import triton.language as tl

# Pattern matching function
@torch.fx.wrap

def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_2, tmp_13

# Argument extraction function

def replacement_args(in_0, in_1, in_2):
    return (in_0 * in_2, in_1)

# Triton kernel for layer norm
@triton.jit
def layer_norm_kernel(
    x_ptr, gamma_ptr, out_ptr,
    batch, seq, hidden,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one sequence row (over the 'seq' dimension)
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # Compute the start index for this sequence row
    x_row = x_ptr + (batch_idx * seq * hidden) + (seq_idx * hidden)
    out_row = out_ptr + (batch_idx * seq * hidden) + (seq_idx * hidden)

    # Compute sum of squares for the row
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, hidden, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden
        x = tl.load(x_row + offset, mask=mask, other=0.0)
        x_sq = x * x
        sum_sq += tl.sum(x_sq, axis=0)

    # Compute mean and standard deviation
    mean = sum_sq / hidden
    var = mean + epsilon
    inv_std = 1.0 / tl.sqrt(var)

    # Compute normalized output for the row
    for i in range(0, hidden, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden
        x = tl.load(x_row + offset, mask=mask, other=0.0)
        x_norm = x * inv_std
        gamma = tl.load(gamma_ptr + offset, mask=mask, other=0.0)
        out = x_norm * (1.0 + gamma)
        tl.store(out_row + offset, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def layer_norm_kernel_wrapper(x, gamma):
    batch, seq, hidden = x.shape
    grid = (seq, batch)  # Blocks per sequence and batch
    out = torch.empty_like(x, dtype=torch.float32)

    # Launch kernel with 256 threads per block (block size 256)
    layer_norm_kernel[grid](
        x_ptr=x,
        gamma_ptr=gamma,
        out_ptr=out,
        batch=batch,
        seq=seq,
        hidden=hidden,
        epsilon=1e-6,
        BLOCK_SIZE=256
    )
    return out.to(dtype=x.dtype)

# Replacement function

def replacement_func():
    return layer_norm_kernel_wrapper