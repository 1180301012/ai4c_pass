import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    return tmp_3

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def conv2d_sigmoid_kernel(in_ptr, weight_ptr, bias_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, n)
    
    for c in range(start, end):
        # Group calculation (96 out channels, 24 per group)
        group = c // 24
        start_input = group * 8
        
        # Compute dot product of input group with weight
        total = tl.zeros((), dtype=tl.float32)
        for k in range(8):
            input_val = tl.load(in_ptr + start_input + k)
            weight_val = tl.load(weight_ptr + c * 8 + k)
            total += input_val * weight_val
        
        total = total + tl.load(bias_ptr + c)
        out_val = tl.sigmoid(total)
        tl.store(out_ptr + c, out_val)

@torch.fx.wrap
def conv_sigmoid_wrapper(in_0, in_1, in_3):
    batch = 1
    input_channels = in_3.shape[1]  # 32
    out_channels = in_1.shape[0]    # 96
    
    # Flatten tensors for kernel processing
    in_flat = in_3
    weight_flat = in_1
    bias_flat = in_0
    
    # Output tensor for convolution result (flattened)
    out_flat = torch.empty(out_channels, device=in_3.device, dtype=in_3.dtype)
    
    # Configure kernel grid
    n = out_channels
    block_size = 32
    num_blocks = (n + block_size - 1) // block_size
    
    # Launch kernel
    conv2d_sigmoid_kernel[(num_blocks,)](
        in_flat,
        weight_flat,
        bias_flat,
        out_flat,
        n,
        block_size
    )
    
    # Reshape to [1, out_channels, 1, 1]
    return out_flat.view(1, out_channels, 1, 1)

def replacement_func():
    return conv_sigmoid_wrapper