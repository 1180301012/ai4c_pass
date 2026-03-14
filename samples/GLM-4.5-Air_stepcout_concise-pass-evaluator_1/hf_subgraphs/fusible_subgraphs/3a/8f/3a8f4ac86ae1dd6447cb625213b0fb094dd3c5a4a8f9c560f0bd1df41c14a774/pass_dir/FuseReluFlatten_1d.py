import torch
import triton
import triton.language as tl

def pattern(x):
    # ReLU -> Dropout -> flatten pattern (matching the actual graph)
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes BLOCK_SIZE consecutive elements in the flattened output
    # Output shape is [B, C], so total elements = B * C
    flat_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = flat_idx < (B * C)
    
    # Convert flat index to [B, C] coordinates
    b = flat_idx // C
    c = flat_idx % C
    
    # Compute input indices: [B, C, 1, 1]
    input_idx = b * C + c  # Flattened input index for [B, C, 1, 1]
    
    # Load input element
    x_val = tl.load(x_ptr + input_idx, mask=mask, other=0.0)
    
    # Apply ReLU
    out_val = tl.maximum(x_val, 0.0)
    
    # Store result
    tl.store(out_ptr + flat_idx, out_val, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    # Input shape is [B, C, 1, 1]
    B, C, H, W = x.shape
    assert H == 1 and W == 1, "This optimization only supports H=1, W=1"
    
    # Output shape is [B, C]
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    
    N = B * C  # Total number of elements after flatten
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        B=B,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_relu_flatten