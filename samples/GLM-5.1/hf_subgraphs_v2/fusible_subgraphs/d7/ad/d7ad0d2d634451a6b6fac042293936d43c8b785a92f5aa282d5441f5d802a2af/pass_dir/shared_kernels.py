import torch
import triton
import triton.language as tl

# ============================================================
# Kernel 1: Standard linear kernel outputting [B, N, M]
# Computes: output[b,n,m] = sum_k input[b,n,k] * weight[m,k] + bias[m]
# ============================================================
@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, N, K, M,
    stride_ib, stride_in, stride_ik,
    stride_wm, stride_wk,
    stride_ob, stride_on, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_per_batch = num_pid_m * num_pid_n
    
    b = pid // num_pid_per_batch
    pid_in_batch = pid % num_pid_per_batch
    pid_m = pid_in_batch // num_pid_n
    pid_n = pid_in_batch % num_pid_n
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # acc is [BLOCK_M, BLOCK_N], stored as output[b, n, m] -> rows are n, cols are m
    # We want output[b, n, m] = sum_k input[b, n, k] * weight[m, k] + bias[m]
    # For tl.dot: we need to arrange as weight_row[m,k] @ input_row[b,n,k]^T
    # So a = weight[m_offsets, k_offsets] -> [BLOCK_M, BLOCK_K]
    # b = input[b, n_offsets, k_offsets] -> [BLOCK_K, BLOCK_N] (transpose of input view)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Weight: [BLOCK_M, BLOCK_K] - rows correspond to output dim m
        a = tl.load(weight_ptr + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk, 
                     mask=k_mask[None, :] & m_mask[:, None], other=0.0)
        
        # Input: we need [BLOCK_K, BLOCK_N] for tl.dot
        # input[b, n, k] - stride_in is for n, stride_ik is for k
        b_input = tl.load(input_ptr + b * stride_ib + k_offsets[:, None] * stride_ik + n_offsets[None, :] * stride_in,
                          mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc += tl.dot(a, b_input, allow_tf32=True)
    
    # Add bias: broadcast bias[m] over n dimension
    bias_vals = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias_vals[:, None]  # [BLOCK_M, BLOCK_N] broadcast over BLOCK_N
    
    # Store output[b, n, m] - acc is arranged with rows as m_offsets and cols as n_offsets
    # output stride: stride_ob for batch, stride_on for n, stride_om for m
    tl.store(output_ptr + b * stride_ob + n_offsets[None, :] * stride_on + m_offsets[:, None] * stride_om, 
             acc, mask=m_mask[:, None] & n_mask[None, :])


# ============================================================
# Kernel 2: Permuted linear kernel outputting [B, M, N] (contiguous)
# Same computation but stored differently: output[b,m,n] = linear_result[b,n,m] permuted
# ============================================================
@triton.jit
def linear_permuted_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, N, K, M,
    stride_ib, stride_in, stride_ik,
    stride_wm, stride_wk,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_per_batch = num_pid_m * num_pid_n
    
    b = pid // num_pid_per_batch
    pid_in_batch = pid % num_pid_per_batch
    pid_m = pid_in_batch // num_pid_n
    pid_n = pid_in_batch % num_pid_n
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        a = tl.load(weight_ptr + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk, 
                     mask=k_mask[None, :] & m_mask[:, None], other=0.0)
        
        b_input = tl.load(input_ptr + b * stride_ib + k_offsets[:, None] * stride_ik + n_offsets[None, :] * stride_in,
                          mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        acc += tl.dot(a, b_input, allow_tf32=True)
    
    bias_vals = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias_vals[:, None]
    
    tl.store(output_ptr + b * stride_ob + m_offsets[:, None] * stride_om + n_offsets[None, :] * stride_on, 
             acc, mask=m_mask[:, None] & n_mask[None, :])


# ============================================================
# Kernel 3: Bilinear interpolation from [B, M, N] (viewed as [B, M, H_in, W_in]) to [B, M, H_out, W_out]
# ============================================================
@triton.jit
def bilinear_interp_kernel(
    input_ptr, output_ptr,
    B, M, N, H_in, W_in, H_out, W_out,
    stride_ib, stride_im, stride_in,
    stride_ob, stride_om, stride_oh, stride_ow,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_per_batch = M * H_out
    b = pid // num_per_batch
    remainder = pid % num_per_batch
    m_idx = remainder // H_out
    h_out = remainder % H_out
    
    scale_h = H_in / H_out
    scale_w = W_in / W_out
    
    y_src = (h_out + 0.5) * scale_h - 0.5
    y_src = tl.where(y_src < 0.0, 0.0, y_src)
    y_src = tl.where(y_src > H_in - 1.0, H_in - 1.0, y_src)
    
    h_top = tl.floor(y_src).to(tl.int32)
    h_top = tl.where(h_top > H_in - 2, H_in - 2, h_top)
    h_bot = h_top + 1
    wy = y_src - h_top
    
    for w_start in range(0, W_out, BLOCK_W):
        w_out_offsets = w_start + tl.arange(0, BLOCK_W)
        w_mask = w_out_offsets < W_out
        
        x_src = (w_out_offsets + 0.5) * scale_w - 0.5
        x_src = tl.where(x_src < 0.0, 0.0, x_src)
        x_src = tl.where(x_src > W_in - 1.0, W_in - 1.0, x_src)
        
        w_left = tl.floor(x_src).to(tl.int32)
        w_left = tl.where(w_left > W_in - 2, W_in - 2, w_left)
        w_right = w_left + 1
        wx = x_src - w_left
        
        n_tl = h_top * W_in + w_left
        n_tr = h_top * W_in + w_right
        n_bl = h_bot * W_in + w_left
        n_br = h_bot * W_in + w_right
        
        base_offset = b * stride_ib + m_idx * stride_im
        
        v_tl = tl.load(input_ptr + base_offset + n_tl * stride_in, mask=w_mask, other=0.0)
        v_tr = tl.load(input_ptr + base_offset + n_tr * stride_in, mask=w_mask, other=0.0)
        v_bl = tl.load(input_ptr + base_offset + n_bl * stride_in, mask=w_mask, other=0.0)
        v_br = tl.load(input_ptr + base_offset + n_br * stride_in, mask=w_mask, other=0.0)
        
        result = v_tl * (1.0 - wy) * (1.0 - wx) + \
                 v_tr * (1.0 - wy) * wx + \
                 v_bl * wy * (1.0 - wx) + \
                 v_br * wy * wx
        
        tl.store(output_ptr + b * stride_ob + m_idx * stride_om + h_out * stride_oh + w_out_offsets * stride_ow,
                 result, mask=w_mask)


# ============================================================
# Shared dispatch wrapper - used by ALL pass files (same replacement_func)
# ============================================================
@torch.fx.wrap
def fused_dispatch_wrapper(bias, weight, input, route):
    B = input.shape[0]
    N = input.shape[1]
    K = input.shape[2]
    M = weight.shape[0]
    
    if route == "linear_only":
        # Just compute linear: output [B, N, M]
        output = torch.empty((B, N, M), dtype=input.dtype, device=input.device)
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        
        num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
        num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
        num_programs = B * num_pid_m * num_pid_n
        
        linear_kernel[(num_programs,)](
            input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
            B=B, N=N, K=K, M=M,
            stride_ib=input.stride(0), stride_in=input.stride(1), stride_ik=input.stride(2),
            stride_wm=weight.stride(0), stride_wk=weight.stride(1),
            stride_ob=output.stride(0), stride_on=output.stride(1), stride_om=output.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return output
    
    if route == "linear_permute":
        # Compute linear + permute: output [B, M, N] (contiguous)
        output = torch.empty((B, M, N), dtype=input.dtype, device=input.device)
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        
        num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
        num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
        num_programs = B * num_pid_m * num_pid_n
        
        linear_permuted_kernel[(num_programs,)](
            input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
            B=B, N=N, K=K, M=M,
            stride_ib=input.stride(0), stride_in=input.stride(1), stride_ik=input.stride(2),
            stride_wm=weight.stride(0), stride_wk=weight.stride(1),
            stride_ob=output.stride(0), stride_om=output.stride(1), stride_on=output.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return output
    
    # Full fused chain routes: linear + permute + reshape + interpolate
    # Output: [B, M, H_out, W_out]
    
    # Phase 1: Compute linear result in permuted layout [B, M, N] (contiguous)
    linear_result = torch.empty((B, M, N), dtype=input.dtype, device=input.device)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_programs_matmul = B * num_pid_m * num_pid_n
    
    linear_permuted_kernel[(num_programs_matmul,)](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=linear_result,
        B=B, N=N, K=K, M=M,
        stride_ib=input.stride(0), stride_in=input.stride(1), stride_ik=input.stride(2),
        stride_wm=weight.stride(0), stride_wk=weight.stride(1),
        stride_ob=linear_result.stride(0), stride_om=linear_result.stride(1), stride_on=linear_result.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    # Phase 2: Bilinear interpolation
    H_in = 16
    W_in = 16
    H_out = 128
    W_out = 128
    
    output = torch.empty((B, M, H_out, W_out), dtype=input.dtype, device=input.device)
    
    BLOCK_W = 64
    
    num_programs_interp = B * M * H_out
    bilinear_interp_kernel[(num_programs_interp,)](
        input_ptr=linear_result, output_ptr=output,
        B=B, M=M, N=N, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        stride_ib=linear_result.stride(0), stride_im=linear_result.stride(1), stride_in=linear_result.stride(2),
        stride_ob=output.stride(0), stride_om=output.stride(1), stride_oh=output.stride(2), stride_ow=output.stride(3),
        BLOCK_W=BLOCK_W,
    )
    
    return output