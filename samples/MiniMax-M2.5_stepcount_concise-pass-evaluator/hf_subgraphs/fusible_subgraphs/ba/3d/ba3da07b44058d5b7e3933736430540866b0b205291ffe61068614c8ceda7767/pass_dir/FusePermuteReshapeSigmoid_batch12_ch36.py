import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_permute_reshape_sigmoid_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies sigmoid element-wise.
    The permute and reshape are handled in Python before calling this kernel.
    """
    # Each program processes a portion of the data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid: 1 / (1 + exp(-x))
    sigmoid_output = 1.0 / (1.0 + tl.exp(-x))
    
    # Store the result
    tl.store(output_ptr + offsets, sigmoid_output, mask=mask)


@torch.fx.wrap
def fused_permute_reshape_sigmoid(bias, weight, input_tensor):
    """
    Fused function that performs:
    1. conv2d(input, weight, bias) -> (B, C_out, H, W)
    2. permute(0, 2, 3, 1) -> (B, H, W, C_out)
    3. reshape -> (B, H*W, C_out)
    4. sigmoid -> (B, H*W, C_out)
    
    All in a single optimized kernel sequence.
    """
    # Step 1: Conv2d (this is already a single CUDA kernel in PyTorch)
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Step 2 & 3: Permute and reshape (combined in one operation)
    batch, channels, height, width = conv_out.shape
    N_elements = batch * height * width * channels
    
    # Permute + reshape in one go
    permuted = conv_out.permute(0, 2, 3, 1)  # (B, H, W, C_out)
    reshaped = permuted.reshape(batch, height * width, channels)  # (B, H*W, C_out)
    
    # Step 4: Apply sigmoid using Triton kernel
    output_flat = reshaped.reshape(-1)
    
    BLOCK_SIZE = 1024
    grid = ((N_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_permute_reshape_sigmoid_kernel[grid](
        input_ptr=output_flat,
        output_ptr=output_flat,
        N=N_elements,
    )
    
    output = output_flat.reshape(batch, height * width, channels)
    return output


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: conv2d -> permute -> reshape -> sigmoid
    Graph: batch=12, out_channels=36
    """
    tmp_0 = in_0  # bias [36]
    tmp_1 = in_1  # weight [36, 512, 1, 1]
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(12, -1, 36)
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement function"""
    return (in_0, in_1, in_2)


def replacement_func():
    """Return the optimized replacement function"""
    return fused_permute_reshape_sigmoid