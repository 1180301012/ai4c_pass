import torch
import triton
import triton.language as tl

# Pattern matching function (matches first graph's specific view dimensions)
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(4, 1, 192)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return tmp_5

# Argument extraction function

def replacement_args(in_0, in_1, in_2):
    # We'll capture the Conv2D output as input to the kernel
    # In actual implementation, this would be the conv2d tensor
    # For the framework, we return the necessary inputs to the kernel
    return (in_2, in_1, in_0)

# Triton kernel for fused softmax
@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    B,
    H,
    W,
    BLOCK_SIZE: tl.constexpr
):
    # Each block handles one batch item
    b = tl.program_id(0)
    # Calculate N = H*W
    N = H * W
    # Thread index for the N dimension
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    # Load the input data for this batch
    start_idx = b * N
    x = tl.load(x_ptr + start_idx + idx, mask=mask, other=0.0)
    
    # Compute max for softmax
    max_val = tl.max(x, axis=0)
    # Subtract max and compute exponential
    x = tl.exp(x - max_val)
    # Compute sum of exponentials
    sum_exp = tl.sum(x, axis=0)
    # Apply softmax
    softmax = x / sum_exp
    
    # Store the result in the output tensor
    # Output shape: [B, 1, H, W, 1]
    # For each element, store at position (b, 0, h, w, 0)
    # We'll store as (B, N) and let the wrapper handle reshape
    tl.store(out_ptr + start_idx + idx, softmax, mask=mask)


# Kernel wrapper (handles tensor allocation and layout)
@torch.fx.wrap
def fused_softmax(x):
    B, _, H, W = x.shape
    N = H * W
    
    # Create output tensor with shape [B, N]
    out = torch.empty((B, N), dtype=x.dtype, device=x.device)
    
    # Calculate grid and block dimensions
    grid = (B,)
    BLOCK_SIZE = 128
    
    # Launch kernel
    fused_softmax_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        B=B,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to final output shape [B, 1, H, W, 1]
    # out: [B, N] -> [B, 1, N, 1] -> [B, 1, H, W, 1]
    return out.view(B, 1, H, W, 1)

# Replacement function

def replacement_func():
    return fused_softmax