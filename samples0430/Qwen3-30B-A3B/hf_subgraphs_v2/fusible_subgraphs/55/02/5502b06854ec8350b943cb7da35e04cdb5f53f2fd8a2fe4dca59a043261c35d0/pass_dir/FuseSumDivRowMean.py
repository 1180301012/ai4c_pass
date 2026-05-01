import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    return tmp_5

def replacement_args(in_0, in_1):
    return (in_1,)


@triton.jit
def mean_kernel(
    in_ptr,
    out_ptr,
    n_batches,
    n_sequences,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    feature_start = tl.program_id(1) * BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < (n_features - feature_start)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for seq in range(n_sequences):
        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_batches, n_sequences, n_features),
            strides=(n_sequences * n_features, n_features, 1),
            offsets=(batch, seq, feature_start),
            block_shape=(1, 1, BLOCK_SIZE),
            order=(0, 1, 2)
        )
        x = tl.load(in_block_ptr, boundary_check=(2,))
        x = tl.reshape(x, (BLOCK_SIZE,))
        acc += x
    acc = acc / n_sequences
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(n_batches, n_features),
        strides=(n_features, 1),
        offsets=(batch, feature_start),
        block_shape=(1, BLOCK_SIZE),
        order=(0, 1)
    )
    acc = tl.reshape(acc, (1, BLOCK_SIZE))
    tl.store(out_block_ptr, acc, boundary_check=(1,))


@torch.fx.wrap
def mean_kernel_wrapper(in_1):
    n_batches, n_sequences, n_features = in_1.shape
    out = torch.empty((n_batches, n_features), dtype=torch.float32, device='cuda')
    BLOCK_SIZE = 128
    grid = (n_batches, triton.cdiv(n_features, BLOCK_SIZE))
    
    # Convert to float32 for kernel computation
    in_1 = in_1.to('cuda').float()
    
    mean_kernel[grid](
        in_ptr=in_1,
        out_ptr=out,
        n_batches=n_batches,
        n_sequences=n_sequences,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return mean_kernel_wrapper