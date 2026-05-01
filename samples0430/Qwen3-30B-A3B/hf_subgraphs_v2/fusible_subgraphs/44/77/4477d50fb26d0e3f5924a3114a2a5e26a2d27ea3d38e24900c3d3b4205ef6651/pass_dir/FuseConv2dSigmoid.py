import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(6, -1, 4)
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    # Extract weight matrix: [512, k] from weight tensor [k, 512, 1, 1]
    weight = in_1.permute(1, 0, 2, 3).reshape(512, -1)
    bias = in_0
    B, C, H, W = in_2.shape
    k = weight.shape[1]
    return (in_2, weight, bias, B, H, W, k)

# Optimized kernel
@triton.jit
def conv2d_sigmoid_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                        B, H, W, k, BLOCK_SIZE: tl.constexpr):
    # Calculate global index
    grid_idx = tl.program_id(0)
    start_idx = grid_idx * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, B * H * W)
    
    # Mask for valid elements
    valid = tl.arange(0, BLOCK_SIZE) < end_idx - start_idx
    
    # Load input block: [BLOCK_SIZE, 512]
    input_offset = (start_idx + tl.arange(0, BLOCK_SIZE)) * 512
    x = tl.load(input_ptr + input_offset[:, None] + tl.arange(0, 512),
                mask=valid[:, None], other=0.0)
    
    # Load weight matrix [512, k]
    weight = tl.load(weight_ptr + (tl.arange(0, 512)[:, None] * k + tl.arange(0, k)[None, :]),
                     mask=tl.arange(0, 512)[:, None] < 512, other=0.0)
    
    # Matrix multiply: [BLOCK_SIZE, k]
    out = tl.dot(x, weight)
    
    # Add bias
    bias = tl.load(bias_ptr + tl.arange(0, k),
                   mask=tl.arange(0, k) < k, other=0.0)
    out += bias
    
    # Store output
    output_offset = (start_idx + tl.arange(0, BLOCK_SIZE)) * k
    tl.store(output_ptr + output_offset[:, None] + tl.arange(0, k),
             out, mask=valid[:, None] & (tl.arange(0, k) < k))

# Kernel wrapper (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_conv_sigmoid(in_2, weight, bias, B, H, W, k):
    # Reshape input: [B, 512, H, W] -> [B*H*W, 512]
    input_flat = in_2.permute(0, 2, 3, 1).reshape(B * H * W, 512)
    
    # Allocate output: [B*H*W, k]
    output = torch.empty(B * H * W, k, dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    grid_size = (B * H * W + 127) // 128
    BLOCK_SIZE = 128
    conv2d_sigmoid_kernel[(grid_size,)](
        input_flat,
        weight,
        bias,
        output,
        B, H, W, k,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape to [B, H*W, k] matching original output
    return output.reshape(B, H * W, k)

# Replacement function (returns kernel wrapper)
def replacement_func():
    return fused_conv_sigmoid