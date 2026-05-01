import torch
import triton
import triton.language as tl

def pattern(tensor):
    return torch.nn.functional.pad(tensor, (0, 0, 0, 1), 'constant', None)

def replacement_args(tensor):
    return (tensor,)

@triton.jit
def pad_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    start = 0
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_features
    if row_id < n_rows:
        data = tl.load(in_ptr + row_id * n_features + offsets, mask=mask)
    else:
        data = 0.0
    tl.store(out_ptr + row_id * n_features + offsets, data, mask=mask)

@torch.fx.wrap
def pad_wrapper(tensor):
    batch = tensor.shape[0]
    n_rows_in = tensor.shape[1]
    n_rows_out = n_rows_in + 1
    n_features = tensor.shape[2]
    out = torch.empty((batch, n_rows_out, n_features), dtype=tensor.dtype, device=tensor.device)
    BLOCK_SIZE = 256
    num_blocks = n_rows_out
    pad_kernel[(num_blocks,)](
        in_ptr=tensor,
        out_ptr=out,
        n_rows=n_rows_in,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return pad_wrapper