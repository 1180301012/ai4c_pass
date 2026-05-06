import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Matches the 1x1 conv + 2x2 avg pool pattern exactly
    # Returns only the final pooled tensor (no cleanup)
    conv_out = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    pooled_out = torch.nn.functional.avg_pool2d(conv_out, 2, 2, 0, False, True, None)
    return pooled_out
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    in_1_shape: tl.tensor,
    in_0_shape: tl.tensor,
    out_shape: tl.tensor,
    BLOCK_SIZE: tl.constexpr,
):
    # Get local block indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Process each block of output
    for i in range(BLOCK_SIZE):
        offset = block_start + i
        # Load 1x1 convolution result (channel-wise)
        conv_val = tl.load(in_1_ptr + offset, mask=tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE)
        # Calculate 2x2 average pooling (simplified for this pattern)
        pooled_val = tl.sum(conv_val) / 4.0
        tl.store(out_ptr + offset, pooled_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    # Determine output shape
    B, C_in, H, W = in_1.shape
    C_out, _, _, _ = in_0.shape
    out_shape = (B, C_out, H//2, W//2)
    out = torch.empty(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Configure kernel grid
    num_programs = (out_shape[2] * out_shape[3] + BLOCK_SIZE - 1) // BLOCK_SIZE
    optimized_kernel[(num_programs,)](    in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        in_1_shape=in_1.shape,
        in_0_shape=in_0.shape,
        out_shape=out.shape,
        BLOCK_SIZE=256,
    )
    return out
def replacement_func():
    return kernel_wrapper