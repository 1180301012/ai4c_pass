import torch
import triton
import triton.language as tl


def pattern(x):
    x_cat = torch.cat([x], 1)
    x_norm = torch.nn.functional.normalize(x_cat, p=2, dim=1)
    return x_norm

def replacement_args(x):
    return (x,)

@triton.jit
def normalize_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_features,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.float32 = 1e-12
):
    row_id = tl.program_id(0)
    row_start = row_id * n_features
    sum_sq = tl.zeros([1], dtype=tl.float32)
    
    # Reduction: sum of squares over features (dim=1)
    for i in range(0, n_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        x_sq = x * x
        sum_sq += tl.sum(x_sq)
    
    # Compute norm (L2) and handle division by zero with eps
    norm = 1.0 / tl.sqrt(sum_sq + eps)
    
    # Element-wise division (output = input / norm)
    feat_id = tl.thread_id(0)
    if feat_id < n_features:
        x = tl.load(x_ptr + row_start + feat_id)
        out = x * norm
        tl.store(y_ptr + row_start + feat_id, out)

@torch.fx.wrap
def normalize_func(x):
    n_rows = x.shape[0]
    n_features = x.shape[1]
    BLOCK_SIZE = 256  # Optimized for 768 feature dimension (768/256=3 blocks)
    num_blocks = n_rows
    
    out = torch.empty_like(x)
    normalize_kernel[(num_blocks,)](
        x_ptr=x,
        y_ptr=out,
        n_rows=n_rows,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return normalize_func