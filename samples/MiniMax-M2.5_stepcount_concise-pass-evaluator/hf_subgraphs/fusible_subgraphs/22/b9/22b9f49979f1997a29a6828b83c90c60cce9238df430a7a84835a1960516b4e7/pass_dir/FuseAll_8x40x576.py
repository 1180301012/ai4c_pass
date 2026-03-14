import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Fuse the entire sequence: cat -> reshape -> transpose -> mul -> pad
    """
    tmp_0 = torch.cat([in_0, in_1, in_2], dim=1)
    tmp_1 = tmp_0.reshape(1, 8, 40, 576)
    tmp_2 = tmp_1.transpose(-1, -2)
    tmp_3 = in_3 * tmp_2
    tmp_4 = torch.nn.functional.pad(tmp_3, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Optimized fused kernel for the entire sequence
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    N, K,
    stride_in0_1, stride_in0_2, stride_in0_3,
    stride_in1_1, stride_in1_2, stride_in1_3,
    stride_in2_1, stride_in2_2, stride_in2_3,
    stride_in3_1, stride_in3_2, stride_in3_3,
    stride_out_1, stride_out_2, stride_out_3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. Concatenate in_0, in_1, in_2 along channel
    2. Reshape to (1, 8, K, N) -> transpose to (1, 8, N, K)
    3. Multiply with in_3
    4. Pad result with zeros at row 0
    """
    # Constants for this pattern
    C0, C1, C2 = 80, 120, 120
    head_dim = 40
    
    # Get position
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N * K
    
    # Compute n and k indices
    n_idx = offs // K
    k_idx = offs % K
    
    mask_n = n_idx < N
    mask_k = k_idx < K
    
    # Determine which head this k belongs to
    head = k_idx // head_dim
    k_in_head = k_idx % head_dim
    
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load in_3: [1, 8, N, K]
    # Only process valid indices
    valid_mask = mask_n & mask_k
    in3_offset = (head * stride_in3_1 + n_idx * stride_in3_2 + k_in_head * stride_in3_3)
    in3_val = tl.load(in_3_ptr + in3_offset, mask=valid_mask, other=0.0)
    
    # Compute which input channel to load from
    channel_idx = k_idx  # This is the channel in the concatenated tensor
    
    # Load from in_0 (channels 0-79)
    if channel_idx < C0:
        in0_offset = (channel_idx * stride_in0_1 + n_idx * stride_in0_2 + k_in_head * stride_in0_3)
        in0_val = tl.load(in_0_ptr + in0_offset, mask=valid_mask, other=0.0)
        result = tl.where(valid_mask, result + in0_val, result)
    
    # Load from in_1 (channels 80-199)
    elif channel_idx < C0 + C1:
        in1_channel = channel_idx - C0
        in1_offset = (in1_channel * stride_in1_1 + n_idx * stride_in1_2 + k_in_head * stride_in1_3)
        in1_val = tl.load(in_1_ptr + in1_offset, mask=valid_mask, other=0.0)
        result = tl.where(valid_mask, result + in1_val, result)
    
    # Load from in_2 (channels 200-319)
    else:
        in2_channel = channel_idx - C0 - C1
        in2_offset = (in2_channel * stride_in2_1 + n_idx * stride_in2_2 + k_in_head * stride_in2_3)
        in2_val = tl.load(in_2_ptr + in2_offset, mask=valid_mask, other=0.0)
        result = tl.where(valid_mask, result + in2_val, result)
    
    # Multiply with in_3
    out_val = result * in3_val
    
    # Compute output offset with padding (add 1 to row index)
    # Output: [1, 9, N, K] - row 0 is padding, data starts at row 1
    out_n = n_idx + 1  # Shift by 1 for padding
    out_offset = (out_n * stride_out_2 + k_in_head * stride_out_3)
    
    # Only store if out_n is valid (within padded bounds)
    out_valid = out_n < (N + 1)
    tl.store(out_ptr + out_offset, out_val, mask=valid_mask & out_valid)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper for fused kernel.
    Input shapes:
    - in_0: [1, 80, 24, 24]
    - in_1: [1, 120, 24, 24]
    - in_2: [1, 120, 24, 24]
    - in_3: [1, 8, 576, 40]
    
    Output shape: [1, 9, 576, 40]
    """
    N = in_3.shape[2]  # 576
    K = in_3.shape[3]  # 40
    
    # Output with padding: [1, 9, N, K]
    output = torch.zeros((1, N + 1, K), dtype=torch.float32, device=in_0.device)
    
    BLOCK_SIZE = 4096
    n_elements = N * K
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=output,
        N=N,
        K=K,
        stride_in0_1=in_0.stride(1), stride_in0_2=in_0.stride(2), stride_in0_3=in_0.stride(3),
        stride_in1_1=in_1.stride(1), stride_in1_2=in_1.stride(2), stride_in1_3=in_1.stride(3),
        stride_in2_1=in_2.stride(1), stride_in2_2=in_2.stride(2), stride_in2_3=in_2.stride(3),
        stride_in3_1=in_3.stride(1), stride_in3_2=in_3.stride(2), stride_in3_3=in_3.stride(3),
        stride_out_1=output.stride(1), stride_out_2=output.stride(2), stride_out_3=output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.unsqueeze(0)


def replacement_func():
    return fused_kernel_wrapper