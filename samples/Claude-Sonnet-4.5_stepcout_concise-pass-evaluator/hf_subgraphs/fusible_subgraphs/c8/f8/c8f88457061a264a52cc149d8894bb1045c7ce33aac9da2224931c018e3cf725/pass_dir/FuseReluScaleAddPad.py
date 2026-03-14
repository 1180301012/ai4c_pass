import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern: ReLU -> Scale -> Add -> Pad
    """
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_scale_add_kernel(
    in_ptr,
    out_ptr,
    scale_val,
    bias_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: ReLU -> Scale -> Add
    x = tl.maximum(x, 0.0)
    x = x * scale_val
    x = x + bias_val
    
    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_relu_scale_add_pad(in_0, in_1, in_2):
    """
    Fused implementation of ReLU -> Scale -> Add -> Pad
    """
    # Get scalar values
    bias_val = in_0.item()
    scale_val = in_1.item()
    
    # Get input shape
    N, C, H, W = in_2.shape
    
    # Output shape after padding (0, 1, 0, 1) - right and bottom padding
    H_out = H + 1
    W_out = W + 1
    
    # Allocate output with padding
    out = torch.zeros((N, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)
    
    # Compute fused ReLU + Scale + Add for the non-padded region
    n_elements = N * C * H * W
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Apply kernel on flattened input, store in flattened output (excluding padding)
    fused_relu_scale_add_kernel[grid](
        in_2,
        out[:, :, :H, :W],  # Only write to non-padded region
        scale_val,
        bias_val,
        n_elements,
    )
    
    return out

def replacement_func():
    return fused_relu_scale_add_pad