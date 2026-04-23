import torch
import triton
import triton.language as tl

# Pattern: view in_4 as [1,1,-1,64], then transpose(1,2), then contiguous
# This reshapes key_states from [1,1,512] to [1,8,1,64]
def pattern(in_4):
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    return tmp_9

def replacement_args(in_4):
    return (in_4,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
    ],
    key=['n_elements', 'H', 'D'],
)
@triton.jit
def reshape_view_transpose_kernel(
    input_ptr, output_ptr,
    n_elements, H: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, vals, mask=mask)


@torch.fx.wrap
def reshape_key_states(in_4):
    # in_4: [1, 1, 512]
    # output: [1, 8, 1, 64]
    H = 8
    D = 64
    n_elements = in_4.numel()

    out = torch.empty((1, H, 1, D), dtype=in_4.dtype, device=in_4.device)

    def grid(META):
        return ((n_elements + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)

    reshape_view_transpose_kernel[grid](
        input_ptr=in_4,
        output_ptr=out,
        n_elements=n_elements,
        H=H, D=D,
    )

    return out

def replacement_func():
    return reshape_key_states