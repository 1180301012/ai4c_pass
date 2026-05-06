import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    sigmoid_out = in_2.sigmoid()
    sig_reshaped = sigmoid_out.view(1, -1, 1, 1)
    expanded = sig_reshaped.expand_as(in_1)
    out = in_1 * expanded + in_0
    return out

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def linear_gather_kernels():
    pass

@triton.jit
def channel_scaling_add_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    N,
    H,
    W,
    BLOCK_SIZE: tl.constexpr = 128,
):
    # Calculate channel block indices
    ch_block_id = tl.program_id(0)
    ch_start = ch_block_id * BLOCK_SIZE
    ch_end = min(ch_start + BLOCK_SIZE, N)
    
    # Create spatial index grid
    row = tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    spatial_index = (row[:, None] * W + col[None, :])
    
    # Process each channel block
    for ch_idx in range(ch_start, ch_end):
        # Get scaling factor from in_2
        scale_val = tl.load(in_2_ptr + ch_idx)
        
        # Process each spatial position in the block
        for spatial_idx in tl.arange(0, BLOCK_SIZE * BLOCK_SIZE):
            # Calculate position in tensor
            pos = spatial_index[spatial_idx]
            
            # Load data
            in_0_val = tl.load(in_0_ptr + (ch_idx * H * W + pos))
            in_1_val = tl.load(in_1_ptr + (ch_idx * H * W + pos))
            
            # Compute and store result
            out_val = in_1_val * scale_val + in_0_val
            tl.store(out_ptr + (ch_idx * H * W + pos), out_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    N = in_0.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]
    out = torch.empty_like(in_0)
    
    # Launch kernel with appropriate configuration
    channel_scaling_add_kernel[tl.cdiv(N, BLOCK_SIZE), tl.cdiv(H * W, BLOCK_SIZE)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        N=N,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return kernel_wrapper