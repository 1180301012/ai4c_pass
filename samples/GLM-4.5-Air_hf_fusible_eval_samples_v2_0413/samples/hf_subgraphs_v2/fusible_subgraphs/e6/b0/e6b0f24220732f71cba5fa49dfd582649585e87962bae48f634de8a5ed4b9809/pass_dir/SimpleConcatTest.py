import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    return tmp_0

@triton.jit
def simple_concat_kernel(
    in2_ptr, in3_ptr,
    out_ptr,
    N, C2, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple concatenation kernel for testing"""
    pid = tl.program_id(0)
    
    # Each program handles a single element
    total_elements = 2 * N * C2 * H * W
    if pid >= total_elements:
        return
    
    # Determine source tensor and position
    is_tensor2 = (pid // (N * C2 * H * W)) == 0
    local_pid = pid % (N * C2 * H * W)
    
    # Calculate batch, channel, and spatial indices
    batch = local_pid // (C2 * H * W)
    channel_in_tensor = (local_pid // (H * W)) % C2
    h = (local_pid // W) % H  
    w = local_pid % W
    
    # Calculate source and destination offsets
    if is_tensor2:
        # From first tensor (in_2)
        src_offset = batch * C2 * H * W + channel_in_tensor * H * W + h * W + w
        val = tl.load(in2_ptr + src_offset)
        dest_channel = channel_in_tensor
    else:
        # From second tensor (in_3) - goes to second half of channels
        src_offset = batch * C2 * H * W + channel_in_tensor * H * W + h * W + w
        val = tl.load(in3_ptr + src_offset)
        dest_channel = C2 + channel_in_tensor
    
    # Calculate destination offset
    dest_offset = batch * (2 * C2) * H * W + dest_channel * H * W + h * W + w
    
    tl.store(out_ptr + dest_offset, val)

@torch.fx.wrap
def simple_concat_test(in_2, in_3):
    N = in_2.shape[0]
    C2 = in_2.shape[1]  # Both tensors have same number of channels
    H, W = in_2.shape[2], in_2.shape[3]
    
    # Concatenated result shape
    out = torch.empty((N, 2 * C2, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # New kernel: one program per element
    total_elements = 2 * N * C2 * H * W
    
    simple_concat_kernel[total_elements,](
        in_2, in_3,
        out,
        N, C2, H, W,
        BLOCK_SIZE=1
    )
    
    return out

def replacement_args(in_2, in_3):
    return (in_2, in_3)

def replacement_func():
    return simple_concat_test