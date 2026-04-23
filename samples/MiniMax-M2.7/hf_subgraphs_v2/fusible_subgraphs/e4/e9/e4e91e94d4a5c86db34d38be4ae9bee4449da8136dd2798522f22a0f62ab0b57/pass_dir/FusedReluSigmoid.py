import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=1, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Sigmoid kernel for better performance.
    ReLU: y = max(0, x)
    Sigmoid: z = 1 / (1 + exp(-y))
    """
    # Get program ID for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    relu_out = tl.where(x > 0, x, 0.0)
    
    # Apply Sigmoid: 1 / (1 + exp(-x))
    # Use stable sigmoid: 1 / (1 + exp(-x)) where x is already relu output (non-negative)
    sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Fused ReLU + Sigmoid operation."""
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Default, will be autotuned
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel
    grid = (num_programs,)
    fused_relu_sigmoid_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0):
    """
    Match the pattern: relu followed by sigmoid.
    Pattern function name must match model's parameter name for proper tracing.
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_relu_sigmoid