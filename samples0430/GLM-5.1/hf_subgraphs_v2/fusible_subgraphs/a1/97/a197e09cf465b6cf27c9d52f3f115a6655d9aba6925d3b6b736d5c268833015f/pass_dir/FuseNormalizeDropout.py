import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = in_0 / tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['row_size'],
)
@triton.jit
def normalize_sum_kernel(
    input_ptr,
    output_ptr,
    num_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    row_offset = row_idx * row_size
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    # Load the entire row, upcast to float32 for precision
    values = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute sum in float32
    row_sum = tl.sum(values, axis=0)

    # Divide each element by sum
    normalized = values / row_sum

    # Store (Triton handles downcast to output dtype automatically)
    tl.store(output_ptr + row_offset + offsets, normalized, mask=mask)


@torch.fx.wrap
def normalize_sum(input_tensor):
    original_shape = input_tensor.shape
    num_rows = input_tensor.numel() // original_shape[-1]
    row_size = original_shape[-1]

    # Flatten to 2D for kernel processing
    input_2d = input_tensor.reshape(num_rows, row_size)
    output_2d = torch.empty_like(input_2d)

    grid = (num_rows,)

    normalize_sum_kernel[grid](
        input_ptr=input_2d,
        output_ptr=output_2d,
        num_rows=num_rows,
        row_size=row_size,
    )

    return output_2d.reshape(original_shape)


def replacement_func():
    return normalize_sum