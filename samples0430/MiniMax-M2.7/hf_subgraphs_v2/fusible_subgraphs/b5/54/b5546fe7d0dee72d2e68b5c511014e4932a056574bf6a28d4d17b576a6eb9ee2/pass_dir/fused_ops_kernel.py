import torch
import triton
import triton.language as tl


@triton.jit
def fused_mult_pad_scale_add_transpose_reshape_kernel(
    data_ptr, in_6_ptr, in_4_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr,
    scale_val: tl.constexpr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for: multiply + pad + scale + add + transpose + reshape
    
    Input shapes:
    - data: [1, 8, N, K]
    - in_6: [1, 8, N, K] 
    - in_4: [1, 8, N+1, K]
    
    Output shape: [1, N+1, 8*K]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Determine position in the output [1, N+1, 8*K]
    # Output layout after transpose: [1, 8, N+1, K] -> [1, N+1, 8, K] -> [1, N+1, 8*K]
    # Position in final: out_idx = n * 8*K + h * K + k
    
    # For output position offsets, we need to find (n, h, k)
    # out_idx = ((n * 8) + h) * K + k = n * 8*K + h * K + k
    # n = out_idx // (8*K)
    # h_k = out_idx % (8*K)
    # h = h_k // K
    # k = h_k % K
    
    n = offsets // (8 * K)
    h_k = offsets % (8 * K)
    h = h_k // K
    k = h_k % K
    
    # n can range from 0 to N (inclusive due to padding)
    # For n == N: this is the padded row, no in_6 multiplication
    # For n < N: multiply data with in_6, then add scaled in_4
    
    # Input data position in [1, 8, N, K]: head * N * K + n * K + k
    data_idx = h * N * K + n * K + k
    
    # Load data[h, n, k]
    data_val = tl.load(data_ptr + data_idx, mask=mask & (n < N), other=0.0)
    
    # Load in_6[h, n, k] 
    in_6_val = tl.load(in_6_ptr + data_idx, mask=mask & (n < N), other=0.0)
    
    # Multiply data with in_6
    mult_val = data_val * in_6_val
    
    # Load in_4[h, n, k] - same shape as data
    # in_4 has shape [1, 8, N+1, K] = [8*(N+1)*K]
    # Position: head * (N+1) * K + n * K + k
    in_4_idx = h * (N + 1) * K + n * K + k
    in_4_val = tl.load(in_4_ptr + in_4_idx, mask=mask, other=0.0)
    
    # Scale in_4
    scaled_in_4 = in_4_val * scale_val
    
    # Pad: for n == N (the padded row), data contribution is 0
    # So result = 0 + scaled_in_4 = scaled_in_4
    result = mult_val + scaled_in_4
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_mult_pad_scale_add_transpose_reshape_wrapper(data, in_6, in_4, scale_val):
    """
    Wrapper for the fused kernel.
    Input: data [1, 8, N, K], in_6 [1, 8, N, K], in_4 [1, 8, N+1, K], scale_val (float)
    Output: [1, N+1, 8*K]
    """
    B, H, N, K = data.shape
    
    # Output shape: [1, N+1, 8*K] after transpose + reshape
    out_elements = (N + 1) * H * K
    
    out = torch.empty((1, N + 1, H * K), dtype=data.dtype, device=data.device)
    
    BLOCK_SIZE = 1024
    num_programs = (out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_mult_pad_scale_add_transpose_reshape_kernel[(num_programs,)](
        data, in_6, in_4, out,
        N, K,
        scale_val,
        out_elements,
        BLOCK_SIZE
    )
    
    return out


@triton.jit
def fused_cat_reshape_transpose_kernel(
    in_2_ptr, in_3_ptr, conv2d_ptr, out_ptr,
    C2: tl.constexpr, C3: tl.constexpr, Cout: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    K: tl.constexpr, N: tl.constexpr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for: cat + reshape + transpose
    
    Input shapes:
    - in_2: [1, C2, H, W]
    - in_3: [1, C3, H, W]
    - conv2d: [1, Cout, H, W]
    
    Output shape: [1, 8, N, K] (after reshape + transpose)
    
    Total channels = C2 + C3 + Cout = 8*K
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Output [1, 8, N, K] with layout: head * N * K + n * K + k
    # We need to figure out which input and what position
    # Total channels per position = 8*K, total spatial = N
    
    # For output position (head, n, k):
    # output_idx = head * N * K + n * K + k
    # n = spatial_idx % N, spatial_idx = spatial_idx // K
    # This is tricky, let me compute backwards
    
    # Actually let's compute the channel position
    # output channel = head * K + k
    # This maps to input channel = (head * K + k) in the concatenated tensor
    
    head = offsets // (N * K)
    n = (offsets % (N * K)) // K
    k = offsets % K
    
    # Concat channel index = head * K + k
    concat_c = head * K + k
    
    # Determine which input based on concat_c
    # in_2: channels [0, C2)
    # in_3: channels [C2, C2 + C3)
    # conv2d: channels [C2 + C3, C2 + C3 + Cout)
    
    # Input indices
    # in_2 has shape [1, C2, H, W] -> [C2, H*W], linearized as c * H*W + n*H + m where m < H
    # For simplicity, assuming H*W = N, the linear index is c * N + n
    
    if concat_c < C2:
        # From in_2
        input_idx = concat_c * N + n
        val = tl.load(in_2_ptr + input_idx, mask=mask, other=0.0)
    elif concat_c < C2 + C3:
        # From in_3
        input_idx = (concat_c - C2) * N + n
        val = tl.load(in_3_ptr + input_idx, mask=mask, other=0.0)
    else:
        # From conv2d
        input_idx = (concat_c - C2 - C3) * N + n
        val = tl.load(conv2d_ptr + input_idx, mask=mask, other=0.0)
    
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def fused_cat_reshape_transpose_wrapper(in_2, in_3, conv2d, K, N):
    """
    Wrapper for fused cat + reshape + transpose kernel.
    Input: in_2 [1, C2, H, W], in_3 [1, C3, H, W], conv2d [1, Cout, H, W]
    Output: [1, 8, N, K] = [1, 8, H*W, total_channels/8]
    """
    B, C2, H, W = in_2.shape
    _, C3, _, _ = in_3.shape
    _, Cout, _, _ = conv2d.shape
    
    total_channels = C2 + C3 + Cout
    H_heads = 8
    assert total_channels == H_heads * K
    
    out = torch.empty((1, H_heads, N, K), dtype=in_2.dtype, device=in_2.device)
    
    out_elements = H_heads * N * K
    
    BLOCK_SIZE = 1024
    num_programs = (out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_cat_reshape_transpose_kernel[(num_programs,)](
        in_2, in_3, conv2d, out,
        C2, C3, Cout,
        H, W, K, N,
        out_elements,
        BLOCK_SIZE
    )
    
    return out


# Dispatch function that routes to correct kernel based on route string
@torch.fx.wrap
def dispatch_kernel(*args):
    route = args[-1]
    if route == "cat_reshape_transpose":
        return fused_cat_reshape_transpose_wrapper(*args[:-1])
    elif route == "mult_pad_scale_add":
        return fused_mult_pad_scale_add_transpose_reshape_wrapper(*args[:-1])
    else:
        raise ValueError(f"Unknown route: {route}")