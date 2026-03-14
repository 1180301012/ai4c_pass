import torch
import triton
import triton.language as tl


@triton.jit
def silu_maxpool2d_kernel(
    input_ptr,
    output_pooled_ptr,
    output_silu_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU activation + MaxPool2d kernel.
    
    SiLU(x) = x * sigmoid(x)
    MaxPool2d with kernel_size=5, stride=1, padding=2
    """
    # Each program processes one element
    pid = tl.program_id(0)
    elem_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = elem_idx < N * C * H * W
    
    # Compute indices
    n = elem_idx // (C * H * W)
    remainder = elem_idx % (C * H * W)
    c = remainder // (H * W)
    hw = remainder % (H * W)
    h = hw // W
    w = hw % W
    
    # Load input
    x = tl.load(input_ptr + elem_idx, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid = tl.where(x >= 0, 1 / (1 + tl.exp(-x)), tl.exp(x) / (1 + tl.exp(x)))
    silu_output = x * sigmoid
    
    # Store SiLU output
    tl.store(output_silu_ptr + elem_idx, silu_output, mask=mask)
    
    # MaxPool2d with kernel_size=5, stride=1, padding=2
    # For each output position, we need to find max in a 5x5 window
    # With padding=2, the input is virtually padded
    # Output size = Input size (with stride=1 and padding=2 for kernel_size=5)
    
    # We need to compute max pool for each output position
    # Since each thread computes one element, we need to compute max over 5x5 window
    max_val = tl.float32(-1e38)  # Initialize with very small value
    
    # Loop over 5x5 window
    for kh in range(-2, 3):
        for kw in range(-2, 3):
            # Virtual padding: clamp indices
            h_in = h + kh
            w_in = w + kw
            
            # Check bounds (virtual padding means we don't go out of bounds)
            valid = (h_in >= 0) and (h_in < H) and (w_in >= 0) and (w_in < W)
            
            if valid:
                # Compute flat index for input position
                in_idx = n * C * H * W + c * H * W + h_in * W + w_in
                val = tl.load(input_ptr + in_idx, mask=True, other=tl.float32(-1e38))
                # Apply SiLU to input before max
                sigmoid = tl.where(val >= 0, 1 / (1 + tl.exp(-val)), tl.exp(val) / (1 + tl.exp(val)))
                val_silu = val * sigmoid
                max_val = tl.max(max_val, val_silu)
    
    # Store pooled output
    tl.store(output_pooled_ptr + elem_idx, max_val, mask=mask)


@torch.fx.wrap
def fused_silu_maxpool2d(x):
    """
    Fused SiLU + MaxPool2d kernel.
    
    This kernel performs SiLU activation and MaxPool2d in a single kernel,
    avoiding memory traffic between two separate operations.
    """
    N, C, H, W = x.shape
    BLOCK_SIZE = 16
    
    # Allocate output tensors
    output_silu = torch.empty_like(x)
    output_pooled = torch.empty((N, C, H, W), dtype=torch.float32, device=x.device)
    
    # Calculate grid
    n_elements = N * C * H * W
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    silu_maxpool2d_kernel[(num_programs,)](
        x,
        output_pooled,
        output_silu,
        N, C, H, W,
        BLOCK_SIZE,
    )
    
    return output_pooled, output_silu


def pattern(in_0):
    """
    Match the pattern: silu(in_0) followed by max_pool2d
    Both outputs must be returned.
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return (tmp_1, tmp_0)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_silu_maxpool2d