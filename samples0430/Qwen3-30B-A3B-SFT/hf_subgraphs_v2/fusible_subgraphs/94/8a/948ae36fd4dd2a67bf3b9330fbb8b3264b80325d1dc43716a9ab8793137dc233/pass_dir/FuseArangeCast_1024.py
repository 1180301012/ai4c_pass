import torch
import triton
import triton.language as tl
from pass_dir.shared_cast_kernel import bool_cast_kernel


# .to(device=torch.device('cuda',0), dtype=torch.bool) -- try variant
def pattern(in_0):
    return in_0.to(device=torch.device('cuda', 0), dtype=torch.bool)


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def _dispatch(in_0, route):
    n_elements = in_0.numel()
    bool_out = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    bool_cast_kernel[grid](
        in_0,
        bool_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return bool_out


def replacement_func():
    return _dispatch