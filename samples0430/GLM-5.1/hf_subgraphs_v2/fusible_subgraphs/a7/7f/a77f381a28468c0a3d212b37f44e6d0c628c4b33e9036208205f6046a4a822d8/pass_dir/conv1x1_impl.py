import torch
import triton
import triton.language as tl

# Route configuration mapping
ROUTE_CONFIG = {
    "s2_k2048_slice_first": {"stride": (2, 2), "slice_end": 2048, "return_order": "slice_first"},
    "s2_k128_slice_first": {"stride": (2, 2), "slice_end": 128, "return_order": "slice_first"},
    "s2_k1024_slice_first": {"stride": (2, 2), "slice_end": 1024, "return_order": "slice_first"},
    "s2_k512_slice_first": {"stride": (2, 2), "slice_end": 512, "return_order": "slice_first"},
    "s1_k2048_slice_first": {"stride": (1, 1), "slice_end": 2048, "return_order": "slice_first"},
    "s1_k64_slice_first": {"stride": (1, 1), "slice_end": 64, "return_order": "slice_first"},
    "s1_k512_slice_first": {"stride": (1, 1), "slice_end": 512, "return_order": "slice_first"},
    "s1_k256_slice_first": {"stride": (1, 1), "slice_end": 256, "return_order": "slice_first"},
    "s1_k128_slice_first": {"stride": (1, 1), "slice_end": 128, "return_order": "slice_first"},
    "s1_k1024_slice_first": {"stride": (1, 1), "slice_end": 1024, "return_order": "slice_first"},
    "s1_k256_conv_first": {"stride": (1, 1), "slice_end": 256, "return_order": "conv_first"},
    "s1_k64_conv_first": {"stride": (1, 1), "slice_end": 64, "return_order": "conv_first"},
}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['C_in', 'C_out', 'M'],
)
@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H_in, W_in, C_out, H_out, W_out,
    stride_h, stride_w,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    wt_stride_0, wt_stride_1,
    out_stride_n, out_stride_co, out_stride_h, out_stride_w,
    M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    """
    1x1 convolution kernel that handles stride=1 and stride=2.
    Treats the operation as a matmul: output[M, C_out] = input_strided[M, C_in] @ weight[C_out, C_in]^T
    where M = N * H_out * W_out
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    
    # Swizzle for better L2 cache locality
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets for the output tile
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m_offs into (batch, h_out, w_out)
    HW_out = H_out * W_out
    batch_idx = m_offs // HW_out
    spatial_idx = m_offs % HW_out
    h_out_idx = spatial_idx // W_out
    w_out_idx = spatial_idx % W_out
    
    # Compute input spatial positions with stride
    h_in_idx = h_out_idx * stride_h
    w_in_idx = w_out_idx * stride_w
    
    # Initialize accumulator in fp32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over input channels (K dimension)
    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        # Load weight tile in transposed order: shape (BLOCK_K, BLOCK_N)
        # weight[co, ci] -> we load as wt[ci, co] = weight[co, ci]
        # Memory layout: weight_ptr + co * wt_stride_0 + ci * wt_stride_1
        # For transposed: weight_ptr + n_offs[col] * wt_stride_0 + k_offs[row] * wt_stride_1
        wt_ptrs = weight_ptr + n_offs[None, :] * wt_stride_0 + k_offs[:, None] * wt_stride_1
        wt_mask = (n_offs[None, :] < C_out) & (k_offs[:, None] < C_in)
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)
        
        # Load input tile: shape (BLOCK_M, BLOCK_K)
        # input[n, ci, h_in, w_in] at input_ptr + n*in_stride_n + ci*in_stride_c + h_in*in_stride_h + w_in*in_stride_w
        in_ptrs = input_ptr + batch_idx[:, None] * in_stride_n \
                           + k_offs[None, :] * in_stride_c \
                           + h_in_idx[:, None] * in_stride_h \
                           + w_in_idx[:, None] * in_stride_w
        in_mask = (m_offs[:, None] < M) & (k_offs[None, :] < C_in)
        inp = tl.load(in_ptrs, mask=in_mask, other=0.0)
        
        # Accumulate: inp (BLOCK_M, BLOCK_K) @ wt (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        accumulator += tl.dot(inp, wt, allow_tf32=True)
    
    # Store output tile
    # output[n, co, h_out, w_out] at output_ptr + n*out_stride_n + co*out_stride_co + h_out*out_stride_h + w_out*out_stride_w
    out_ptrs = output_ptr + batch_idx[:, None] * out_stride_n \
                          + n_offs[None, :] * out_stride_co \
                          + h_out_idx[:, None] * out_stride_h \
                          + w_out_idx[:, None] * out_stride_w
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < C_out)
    tl.store(out_ptrs, accumulator, mask=out_mask)


@torch.fx.wrap
def conv1x1_dispatch(weight, input, route):
    config = ROUTE_CONFIG[route]
    stride_h, stride_w = config["stride"]
    slice_end = config["slice_end"]
    return_order = config["return_order"]
    
    # Ensure weight is on the same device as input
    if weight.device != input.device:
        weight = weight.to(input.device)
    
    # Ensure weight dtype matches input dtype (for Triton kernel compatibility)
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    
    # Get dimensions
    # weight: [C_out, C_in, 1, 1] or [C_out, C_in]
    # input: [N, C_in, H_in, W_in]
    N, C_in, H_in, W_in = input.shape
    C_out = weight.shape[0]
    
    # Compute output spatial dimensions
    # conv2d with padding=0, dilation=1, kernel_size=1
    # H_out = (H_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1
    # For 1x1 conv: H_out = (H_in - 1) / stride_h + 1 = (H_in - 1 + stride_h) / stride_h
    # Actually: H_out = floor((H_in + 2*0 - 1*(1-1) - 1) / stride_h) + 1 = floor((H_in - 1) / stride_h) + 1
    H_out = (H_in - 1) // stride_h + 1
    W_out = (W_in - 1) // stride_w + 1
    
    # Allocate output
    output = torch.empty((N, C_out, H_out, W_out), dtype=input.dtype, device=input.device)
    
    # Compute M = N * H_out * W_out
    M = N * H_out * W_out
    
    # Get strides
    in_sn, in_sc, in_sh, in_sw = input.stride()
    # Weight is [C_out, C_in, 1, 1] - we need strides for the first two dims
    wt_s0, wt_s1 = weight.stride(0), weight.stride(1)
    out_sn, out_sc, out_sh, out_sw = output.stride()
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),
    )
    
    conv1x1_kernel[grid](
        input_ptr=input, weight_ptr=weight, output_ptr=output,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out, H_out=H_out, W_out=W_out,
        stride_h=stride_h, stride_w=stride_w,
        in_stride_n=in_sn, in_stride_c=in_sc, in_stride_h=in_sh, in_stride_w=in_sw,
        wt_stride_0=wt_s0, wt_stride_1=wt_s1,
        out_stride_n=out_sn, out_stride_co=out_sc, out_stride_h=out_sh, out_stride_w=out_sw,
        M=M,
    )
    
    # Create slice view
    sliced = output[:, :slice_end, :, :]
    
    # Return in the appropriate order
    if return_order == "slice_first":
        return (sliced, output)
    else:
        return (output, sliced)