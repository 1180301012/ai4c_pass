import torch
import triton
import triton.language as tl


@triton.jit
def conv1x1_stride1_kernel(
    input_ptr, weight_ptr, out_ptr,
    N, C_in, H, W, C_out,
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_co, stride_weight_ci,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    1x1 conv with stride=1: out[n, co, h, w] = sum_ci input[n, ci, h, w] * weight[co, ci, 0, 0]
    This is equivalent to: out[n, h*W+w, co] = sum_ci input[n, h*W+w, ci] * weight[co, ci]
    We treat it as a matmul over the spatial dimensions.
    Input reshaped: [N*H*W, C_in]  (contiguous if input is [N, C_in, H, W] contiguous)
    Weight reshaped: [C_out, C_in] (contiguous)
    Output reshaped: [N*H*W, C_out] (contiguous if output is [N, C_out, H, W] contiguous)
    """
    M_total = N * H * W
    
    # pid = tl.program_id(0)
    # num_pid_m = tl.cdiv(M_total, BLOCK_M)
    # num_pid_n = tl.cdiv(C_out, BLOCK_N)
    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n
    
    # Use grouped ordering for better L2 cache access
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < M_total
    mask_n = offs_n < C_out
    
    # For input [N, C_in, H, W], the n,c,h,w index for spatial position m:
    # m = n * H * W + h * W + w
    # input[n, ci, h, w] = input_ptr + n*stride_input_n + ci*stride_input_c + h*stride_input_h + w*stride_input_w
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, C_in, BLOCK_K):
        offs_k_curr = k_start + offs_k
        mask_k = offs_k_curr < C_in
        
        # Load input block: [BLOCK_M, BLOCK_K]
        # input[n, ci, h, w] for m = n*H*W + h*W + w
        # Decompose m into n, h, w
        n_idx = offs_m // (H * W)
        hw_idx = offs_m % (H * W)
        h_idx = hw_idx // W
        w_idx = hw_idx % W
        
        input_offsets = n_idx * stride_input_n + offs_k_curr * stride_input_c + h_idx * stride_input_h + w_idx * stride_input_w
        a = tl.load(input_ptr + input_offsets, mask=(mask_m & mask_k)[None, :] | mask_k[None, :], other=0.0)
        
        # Actually need mask_m broadcasted correctly:
        # a has shape (BLOCK_M, BLOCK_K), mask should be mask_m[:, None] & mask_k[None, :]
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(input_ptr + input_offsets, mask=a_mask, other=0.0)
        
        # Load weight block: [BLOCK_K, BLOCK_N]
        # weight[co, ci] for 1x1 conv
        w_offsets = offs_k_curr[:, None] * stride_weight_ci + offs_n[None, :] * stride_weight_co
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(weight_ptr + w_offsets, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=False)
    
    # Convert accumulator to output dtype
    out_dtype = out_ptr.type.element_ty
    acc_cast = accumulator.to(out_dtype)
    
    # Store output block: [BLOCK_M, BLOCK_N]
    # output[n, co, h, w] for m = n*H*W + h*W + w
    n_idx = offs_m // (H * W)
    hw_idx = offs_m % (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    out_offsets = n_idx * stride_out_n + offs_n * stride_out_c + h_idx * stride_out_h + w_idx * stride_out_w
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offsets, acc_cast, mask=out_mask)


@triton.jit
def conv1x1_stride2_kernel(
    input_ptr, weight_ptr, out_ptr,
    N, C_in, H_in, W_in, C_out,
    H_out, W_out,
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_co, stride_weight_ci,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    1x1 conv with stride=2: out[n, co, ho, wo] = sum_ci input[n, ci, 2*ho, 2*wo] * weight[co, ci, 0, 0]
    M_total = N * H_out * W_out
    Input indexing: input[n, ci, 2*ho, 2*wo]
    """
    M_total = N * H_out * W_out
    
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M_total, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < M_total
    mask_n = offs_n < C_out
    
    # Decompose m into n, ho, wo for output
    n_idx = offs_m // (H_out * W_out)
    hw_idx = offs_m % (H_out * W_out)
    ho_idx = hw_idx // W_out
    wo_idx = hw_idx % W_out
    
    # For stride=2: input h = 2*ho, input w = 2*wo
    hi_idx = 2 * ho_idx
    wi_idx = 2 * wo_idx
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, C_in, BLOCK_K):
        offs_k_curr = k_start + offs_k
        mask_k = offs_k_curr < C_in
        
        # Load input: input[n, ci, 2*ho, 2*wo]
        input_offsets = n_idx * stride_input_n + offs_k_curr * stride_input_c + hi_idx * stride_input_h + wi_idx * stride_input_w
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(input_ptr + input_offsets, mask=a_mask, other=0.0)
        
        # Load weight: weight[co, ci] 
        w_offsets = offs_k_curr[:, None] * stride_weight_ci + offs_n[None, :] * stride_weight_co
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(weight_ptr + w_offsets, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=False)
    
    out_dtype = out_ptr.type.element_ty
    acc_cast = accumulator.to(out_dtype)
    
    # Store output: output[n, co, ho, wo]
    out_offsets = n_idx * stride_out_n + offs_n * stride_out_c + ho_idx * stride_out_h + wo_idx * stride_out_w
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offsets, acc_cast, mask=out_mask)


@torch.fx.wrap
def conv1x1_slice_dispatch(in_0, in_1, route_str):
    """
    Dispatch wrapper for 1x1 conv + slice fusion.
    route_str format: "stride_{s}_slice_{k}_order_{ord}"
    where s is stride (1 or 2), k is slice endpoint, ord is 'sf' (slice_first) or 'cf' (conv_first)
    """
    # Parse route string
    parts = route_str.split('_')
    stride_val = int(parts[1])
    slice_k = int(parts[3])
    order = parts[5]  # 'sf' or 'cf'
    
    # Get dimensions
    N = in_1.shape[0]
    C_in = in_1.shape[1]
    H_in = in_1.shape[2]
    W_in = in_1.shape[3]
    C_out = in_0.shape[0]  # weight shape [C_out, C_in, 1, 1]
    
    # Compute output spatial dims
    if stride_val == 1:
        H_out = H_in
        W_out = W_in
    else:
        H_out = (H_in + 0) // stride_val  # padding=0
        W_out = (W_in + 0) // stride_val
    
    # Allocate output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device)
    
    # Ensure weight is on the same device and contiguous
    weight = in_0
    if weight.device != in_1.device:
        weight = weight.to(in_1.device)
    weight = weight.reshape(C_out, C_in).contiguous()
    
    # Ensure input is contiguous
    input_tensor = in_1.contiguous()
    
    M_total = N * H_out * W_out
    
    # Choose block sizes based on dimensions
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    if C_out <= 64:
        BLOCK_N = 32
        BLOCK_M = 128
    elif C_out <= 128:
        BLOCK_N = 64
        BLOCK_M = 64
    
    grid = lambda META: (
        triton.cdiv(M_total, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),
    )
    
    if stride_val == 1:
        conv1x1_stride1_kernel[grid](
            input_tensor.data_ptr(), weight.data_ptr(), out.data_ptr(),
            N, C_in, H_in, W_in, C_out,
            input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
        )
    else:
        conv1x1_stride2_kernel[grid](
            input_tensor.data_ptr(), weight.data_ptr(), out.data_ptr(),
            N, C_in, H_in, W_in, C_out,
            H_out, W_out,
            input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
        )
    
    # Slice the output
    sliced = out[:, :slice_k, :, :]
    
    # Return in correct order
    if order == 'sf':
        return (sliced, out)
    else:  # 'cf'
        return (out, sliced)