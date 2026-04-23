import torch
import triton
import triton.language as tl


@triton.jit
def reshape_linear_slice_fusion_kernel(
    input_ptr, weight_ptr, bias_ptr,
    out_first_ptr, out_last_ptr,
    M: tl.constexpr,  # 300
    B: tl.constexpr,  # 150 (batch size after reshape)
    K: tl.constexpr,  # 256
    N: tl.constexpr,  # 512
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused reshape + linear + slice kernel for in_4 path.
    Input: [1, 150, 1, 512] -> reshape to [300, 150, 256] -> linear -> [300, 150, 512]
    Then slice to first/last 256 channels.
    """
    # Each program handles one row of the output [M * B, N/2]
    program_idx = tl.program_id(0)
    
    # Calculate which batch element and row
    batch_idx = program_idx // B  # which of the 300 rows
    spatial_idx = program_idx % B  # which of the 150 spatial positions
    
    # Input: [1, 150, 1, 512] -> after reshape: [300, 150, 256]
    # We need to compute the right index into the flattened input
    # Original: [1, 150, 1, 512], first two dims collapse to 150
    # After reshape to [300, 150, 256]: index = batch_idx * 150 * 256 + spatial_idx * 256
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Accumulators for first and last 256 outputs
    acc_first = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc_last = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load bias
    bias_first = tl.load(bias_ptr + col_offsets, mask=col_offsets < N // 2, other=0.0)
    bias_last = tl.load(bias_ptr + N // 2 + col_offsets, mask=col_offsets < N // 2, other=0.0)
    
    # Compute linear: y = x @ W^T + b
    # Input x is [K=256], weight W is [N=512, K=256]
    # We compute first half (256) and second half (256) separately
    
    for k in range(0, K, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < K
        
        # Load input value at position k
        # In the reshaped view: [300, 150, 256]
        # index = batch_idx * B * K + spatial_idx * K + k
        input_offset = batch_idx * B * K + spatial_idx * K + k_offsets
        input_vals = tl.load(input_ptr + input_offset, mask=k_mask, other=0.0)
        
        # Weight for first 256 output channels
        weight_offset_first = k_offsets  # rows 0-255, cols k
        w_first = tl.load(weight_ptr + weight_offset_first, mask=k_mask, other=0.0)
        acc_first += input_vals * w_first
        
        # Weight for last 256 output channels
        weight_offset_last = (N // 2) * K + k_offsets  # rows 256-511, cols k
        w_last = tl.load(weight_ptr + weight_offset_last, mask=k_mask, other=0.0)
        acc_last += input_vals * w_last
    
    acc_first = acc_first + bias_first
    acc_last = acc_last + bias_last
    
    # Store outputs: [300, 150, 256] each
    out_offset = program_idx * (N // 2)
    out_first = out_first_ptr + out_offset
    out_last = out_last_ptr + out_offset
    
    tl.store(out_first + col_offsets, acc_first, mask=col_offsets < N // 2)
    tl.store(out_last + col_offsets, acc_last, mask=col_offsets < N // 2)


def pattern(in_4, in_3, in_2):
    """
    Pattern for reshape + linear + slice fusion on in_4 path.
    Matches: reshape(in_4) -> linear -> two slices.
    Output: tuple of (tmp_11[..., :256], tmp_12[..., -256:])
    """
    tmp_9 = in_4.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = linear_1[Ellipsis, slice(None, 256, None)]
    tmp_12 = linear_1[Ellipsis, slice(-256, None, None)]
    return tmp_11, tmp_12


def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)


@torch.fx.wrap
def reshape_linear_slice_wrapper(in_4, in_3, in_2):
    """
    Wrapper for fused reshape + linear + slice kernel.
    Input in_4: [1, 150, 1, 512] -> reshape to [300, 150, 256]
    Weight in_3: [512, 256]
    Bias in_2: [512]
    Returns: tuple of (first_256, last_256) each [300, 150, 256]
    """
    # After reshape: [300, 150, 256]
    M = 300
    B = 150
    K = 256
    N = 512
    
    # Output shapes: [300, 150, 256] each
    out_first = torch.empty((M, B, N // 2), dtype=in_4.dtype, device=in_4.device)
    out_last = torch.empty((M, B, N // 2), dtype=in_4.dtype, device=in_4.device)
    
    BLOCK_SIZE = 128
    grid = (M * B,)  # 300 * 150 = 45000 programs
    
    # Contiguous data for Triton
    input_flat = in_4.reshape(-1).contiguous()  # [76800]
    weight = in_3.contiguous()
    bias = in_2.contiguous()
    
    reshape_linear_slice_fusion_kernel[grid](
        input_flat, weight, bias,
        out_first, out_last,
        M, B, K, N,
        BLOCK_SIZE,
    )
    
    return out_first, out_last


def replacement_func():
    return reshape_linear_slice_wrapper