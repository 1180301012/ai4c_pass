import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def make_kernel():
    @triton.jit
    def optimized_conv2d_mean_kernel(
        in_1_ptr,
        in_0_ptr,
        out_conv_ptr,
        out_mean_ptr,
        in_1_shape,
        in_0_shape,
        out_conv_shape,
        out_mean_shape,
        BLOCK_SIZE: tl.constexpr,
        STRIDE: tl.constexpr,
        PAD: tl.constexpr,
        DILATION: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
    ):
        # Calculate grid position
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE, dtype=tl.int32)
        
        # Calculate valid range
        valid_mask = offsets < out_conv_shape[2]
        
        # Load inputs
        in_1 = tl.load(in_1_ptr + offsets, mask=valid_mask, other=0.0)
        in_0 = tl.load(in_0_ptr + offsets, mask=valid_mask, other=0.0)
        
        # Compute convolution
        conv = tl.dot(in_1, in_0)  # Simplified kernel for this example
        
        # Compute mean for this block (this will be aggregated later)
        block_mean = tl.sum(conv, axis=0) / tl.float32(BLOCK_SIZE)
        
        # Store results
        tl.store(out_conv_ptr + offsets, conv, mask=valid_mask)
        tl.store(out_mean_ptr + offsets, block_mean, mask=valid_mask)
    
    @torch.fx.wrap
    def kernel_wrapper(in_0, in_1):
        B, C_in, H_in, W_in = in_1.shape
        C_out = in_0.shape[0]
        
        # Allocate output
        out_conv = torch.empty((B, C_out, H_in, W_in), dtype=in_1.dtype, device=in_1.device)
        out_mean = torch.empty((B, C_out, 1, 1), dtype=in_1.dtype, device=in_1.device)
        
        # Create kernel with appropriate parameters
        optimized_kernel = make_kernel()
        
        # Launch kernel
        grid = (tl.cdiv(out_conv.shape[2], BLOCK_SIZE),)
        optimized_kernel[
            grid
        ](
            in_1_ptr=in_1,
            in_0_ptr=in_0,
            out_conv_ptr=out_conv,
            out_mean_ptr=out_mean,
            in_1_shape=(B, C_in, H_in, W_in),
            in_0_shape=(C_out, C_in, 1, 1),
            out_conv_shape=(B, C_out, H_in, W_in),
            out_mean_shape=(B, C_out, 1, 1),
            BLOCK_SIZE=128,
            STRIDE=1,
            PAD=1,
            DILATION=1,
            OUT_CHANNELS=C_out
        )
        
        return (out_conv, out_mean)
    
    return kernel_wrapper

def replacement_func():
    return make_kernel()