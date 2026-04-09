import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def custom_unfold_kernel_1d(
    input_ptr,
    output_ptr,
    batch_size,
    input_channels,
    input_length,
    kernel_size,
    output_channels,
    output_length,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one output position
    batch_idx = pid // (output_channels * output_length)
    channel_idx = (pid % (output_channels * output_length)) // output_length
    pos_idx = pid % output_length
    
    # Calculate input position range
    input_start = pos_idx - kernel_size // 2  # With symmetric padding
    
    if batch_idx >= batch_size or channel_idx >= output_channels or pos_idx >= output_length:
        return
    
    # Load input window
    values = tl.zeros((kernel_size,), dtype=tl.float32)
    for k in range(kernel_size):
        input_pos = input_start + k
        if 0 <= input_pos < input_length:
            offset = batch_idx * input_channels * input_length + channel_idx * input_length + input_pos
            values[k] = tl.load(input_ptr + offset)
    
    # Store output
    output_offset = batch_idx * output_channels * output_length + channel_idx * output_length + pos_idx
    tl.store(output_ptr + output_offset, tl.sum(values))

@torch.fx.wrap
def optimized_transform_pipeline(input_tensor):
    B, C, L = input_tensor.shape  # [1, 16, 45]
    
    # Add dimension: [1, 16, 45] -> [1, 16, 45, 1]
    input_4d = input_tensor.unsqueeze(-1)
    
    # Customunfold with kernel_size=[9,1]
    kernel_size = 9
    padding = 4
    dilation = 1
    stride = 1
    
    # Calculate output length
    output_length = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    # With padding=4, kernel_size=9: (45 + 8 - 8 - 1) // 1 + 1 = 45
    
    out_channels = C * kernel_size  # 16 * 9 = 144
    total_elements = B * out_channels * output_length  # 1 * 144 * 45 = 6480
    
    output = torch.empty(B, out_channels, output_length, device=input_tensor.device, dtype=input_tensor.dtype)
    
    BLOCK_N = 256
    num_programs = B * out_channels * output_length
    
    # Use the custom unfold kernel
    custom_unfold_kernel_1d[(num_programs + BLOCK_N - 1) // BLOCK_N,](
        input_ptr=input_4d,
        output_ptr=output,
        batch_size=B,
        input_channels=C,
        input_length=L,
        kernel_size=kernel_size,
        output_channels=out_channels,
        output_length=output_length,
        BLOCK_N=BLOCK_N,
    )
    
    # Transpose: [1, 144, 45] -> [1, 45, 144]
    output = output.transpose(1, 2)
    
    return output

def replacement_func():
    return optimized_transform_pipeline