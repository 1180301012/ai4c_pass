import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_add_relu_kernel(
    in_0_ptr, in_1_ptr, in_3_ptr,
    out_ptr,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = B * C * H * W
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: relu(in_0 + in_1 + in_3)
    temp = in_0 + in_1 + in_3
    relu_result = tl.where(temp > 0, temp, 0.0)
    
    # Store
    tl.store(out_ptr + offsets, relu_result, mask=mask)


@torch.fx.wrap
def optimized_computation(in_0, in_1, in_2, in_3):
    """Fused kernel for (in_0 + in_1 + in_3) -> relu"""
    B, C, H, W = in_0.shape
    
    relu_out = torch.empty_like(in_0)
    
    BLOCK_SIZE = 1024
    num_elements = B * C * H * W
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_add_relu_kernel[(num_programs,)](
        in_0, in_1, in_3,
        relu_out,
        B, C, H, W,
        BLOCK_SIZE,
    )
    
    # Now do the chunk operations using PyTorch (they're memory bound anyway)
    in_2_chunks = in_2.chunk(2, dim=1)
    relu_chunks = relu_out.chunk(2, dim=1)
    
    return in_2_chunks[0], relu_chunks[0], in_2_chunks[1], relu_chunks[1]


def pattern(in_0, in_1, in_2, in_3):
    """Match the pattern: in_0 + in_1, then + in_3, then relu"""
    # Element-wise additions
    t1 = in_0 + in_1
    t2 = t1 + in_3
    # ReLU activation
    t3 = torch.nn.functional.relu(t2, inplace=False)
    # Chunk operations
    c1 = in_2.chunk(2, dim=1)
    c2 = t3.chunk(2, dim=1)
    return c1[0], c2[0], c1[1], c2[1]


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return optimized_computation