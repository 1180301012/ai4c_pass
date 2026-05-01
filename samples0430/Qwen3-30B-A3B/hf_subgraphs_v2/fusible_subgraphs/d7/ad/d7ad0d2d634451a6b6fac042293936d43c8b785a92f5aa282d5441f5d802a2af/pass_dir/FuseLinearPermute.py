import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    permuted = linear.permute(0, 2, 1)
    return permuted

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_linear_permute_kernel(in_ptr,
                               weight_ptr,
                               bias_ptr,
                               out_ptr,
                               batch_size,
                               seq_len,
                               in_features,
                               out_features,
                               BLOCK_SIZE: tl.constexpr):
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    out_start = out_idx * BLOCK_SIZE
    seq_start = seq_idx * BLOCK_SIZE

    bias = tl.load(bias_ptr + out_start)
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, in_features, BLOCK_SIZE):
        in_block = tl.load(
            in_ptr + batch_idx * seq_len * in_features + seq_start * in_features + k,
            mask=(k + BLOCK_SIZE <= in_features),
            other=0.0
        )
        in_block = tl.reshape(in_block, (BLOCK_SIZE, BLOCK_SIZE))

        weight_block = tl.load(
            weight_ptr + out_start * in_features + k,
            mask=(k + BLOCK_SIZE <= in_features),
            other=0.0
        )
        weight_block = tl.reshape(weight_block, (BLOCK_SIZE, BLOCK_SIZE))

        acc += tl.dot(in_block, weight_block)

    acc += bias
    out_block = out_ptr + batch_idx * out_features * seq_len + out_start * seq_len + seq_start
    tl.store(
        out_block,
        acc,
        mask=(out_start + tl.arange(0, BLOCK_SIZE) < out_features)[:, None] &
              (seq_start + tl.arange(0, BLOCK_SIZE) < seq_len)[None, :]
    )

@torch.fx.wrap
def fused_linear_permute_wrapper(in_0, in_1, in_2):
    batch_size, seq_len, in_features = in_2.shape
    out_features = in_1.shape[0]
    out = torch.empty((batch_size, out_features, seq_len), dtype=in_2.dtype, device=in_2.device)
    BLOCK_SIZE = 32
    grid = (batch_size, (out_features + BLOCK_SIZE - 1) // BLOCK_SIZE, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    fused_linear_permute_kernel[grid](
        in_2, in_1, in_0, out,
        batch_size,
        seq_len,
        in_features,
        out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return fused_linear_permute_wrapper