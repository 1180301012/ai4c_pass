import torch
import triton
import triton.language as tl


@triton.jit
def optimized_transpose_view_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one output element
    # Total output: B * 32 * 15 * 15 = B * 7200 elements
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_out = B * 32 * 15 * 15
    mask = idx < total_out
    
    # Output indices: [B, 32, 15, 15]
    b = idx // (32 * 15 * 15)
    remaining = idx % (32 * 15 * 15)
    c = remaining // (15 * 15)
    remaining = remaining % (15 * 15)
    h = remaining // 15
    w = remaining % 15
    
    # Original input: [B, 1, 225, 32]
    # After transpose(2, 3): [B, 1, 32, 225]
    # After view(4, 32, 15, 15): [B, 32, 15, 15]
    # For output[b, c, h, w], data comes from:
    # tmp_30 = tmp_29.view(4, 32, 15, 15)
    # tmp_29 = tmp_28.transpose(2, 3) which is [B, 1, 32, 225]
    # tmp_28 is the original input [B, 1, 225, 32]
    # tmp_30[b, c, h, w] = tmp_29[b, 1, c, h*15 + w]
    # tmp_29[b, 1, c, h*15 + w] = tmp_28[b, 1, h*15 + w, c]
    
    spatial_idx = h * 15 + w  # h*15 + w in [0, 225)
    
    # Load from tmp_28[b, 1, spatial_idx, c]
    in_offset = b * 1 * 225 * 32 + 0 * 225 * 32 + spatial_idx * 32 + c
    val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    # Store to tmp_30[b, c, h, w]
    out_offset = b * 32 * 15 * 15 + c * 15 * 15 + h * 15 + w
    tl.store(out_ptr + out_offset, val, mask=mask)


@torch.fx.wrap
def optimized_transpose_view(input_tensor):
    B = 4
    output = torch.empty(B, 32, 15, 15, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 256
    total_elements = B * 32 * 15 * 15
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_transpose_view_kernel[(num_programs,)](
        in_ptr=input_tensor,
        out_ptr=output,
        B=B,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(x):
    t = x.transpose(2, 3)
    v = t.view(4, 32, 15, 15)
    return v


def replacement_args(x):
    return (x,)


def replacement_func():
    return optimized_transpose_view