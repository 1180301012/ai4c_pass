import torch
import triton
import triton.language as tl


def _get_dtype_code(dtype):
    if dtype == torch.float32:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.bfloat16:
        return 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ============ Conv1x1 + BN Fusion Kernel ============
# After BN folding, the computation is:
# output[b, co, h, w] = sum_{ci} fused_weight[co, ci] * input[b, ci, h, w] + fused_bias[co]
# This is a per-pixel matrix multiplication (1x1 conv = linear projection per spatial position)

@triton.jit
def conv1x1_bn_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, C_out, H, W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    M = B * H * W
    HW = H * W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode spatial positions from flat index
    b_idx = offs_m // HW
    hw_idx = offs_m % HW
    
    m_mask = offs_m < M
    n_mask = offs_n < C_out
    
    # Accumulator for output block [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over input channels in blocks of BLOCK_K
    for k_start in range(0, C_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load input block: input[b, ci, h, w] = b*C_in*HW + ci*HW + h*W + w
        # For scattered channels-first layout, input channels are HW apart
        in_ptrs = b_idx[:, None] * (C_in * HW) + offs_k[None, :] * HW + hw_idx[:, None]
        in_mask = m_mask[:, None] & (offs_k[None, :] < C_in)
        in_block = tl.load(input_ptr + in_ptrs, mask=in_mask, other=0.0).to(tl.float32)
        
        # Load weight block in transposed layout for tl.dot
        # weight[co, ci] at offset: co*C_in + ci
        # We need wt_block[k, n] = weight[offs_n[n], offs_k[k]]
        # offset = offs_n[n]*C_in + offs_k[k]
        wt_ptrs = offs_n[None, :] * C_in + offs_k[:, None]
        wt_mask = (offs_k[:, None] < C_in) & n_mask[None, :]
        wt_block = tl.load(weight_ptr + wt_ptrs, mask=wt_mask, other=0.0).to(tl.float32)
        
        # Matrix multiply: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] = [BLOCK_M, BLOCK_N]
        acc += tl.dot(in_block, wt_block)
    
    # Add per-channel bias
    bias_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]
    
    # Cast to output dtype and store
    if DTYPE == 1:
        result = acc.to(tl.float16)
    elif DTYPE == 2:
        result = acc.to(tl.bfloat16)
    else:
        result = acc
    
    # Store output: output[b, co, h, w] = b*C_out*HW + co*HW + h*W + w
    out_ptrs = b_idx[:, None] * (C_out * HW) + offs_n[None, :] * HW + hw_idx[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def _conv1x1_bn(input_tensor, fused_weight, fused_bias):
    B, C_in, H, W = input_tensor.shape
    C_out = fused_weight.shape[0]
    
    # Allocate output in same dtype as input
    output = torch.empty((B, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    dtype_code = _get_dtype_code(input_tensor.dtype)
    
    # Block sizes - tuned for 1x1 conv typical sizes (C_in=48, C_out=192)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    M = B * H * W
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(C_out, BLOCK_N)
    
    conv1x1_bn_kernel[(grid_m, grid_n)](
        input_ptr=input_tensor,
        weight_ptr=fused_weight,
        bias_ptr=fused_bias,
        output_ptr=output,
        B=B, C_in=C_in, C_out=C_out, H=H, W=W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        DTYPE=dtype_code,
    )
    
    return output


# ============ BN Only Kernel ============
# In inference mode, BN is a per-channel affine transformation:
# output = scale * input + shift
# where scale = weight / sqrt(running_var + eps)
#       shift = bias - weight * running_mean / sqrt(running_var + eps)

@triton.jit
def bn_kernel(
    input_ptr, scale_ptr, shift_ptr, output_ptr,
    total_elements, C, HW, CHW,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    # Decode (b, c, hw) from flat index for channels-first [B, C, H, W] layout
    b_idx = offs // CHW
    c_idx = (offs % CHW) // HW
    # hw_idx = offs % HW  # not needed for offset computation
    
    # Load input value
    input_val = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Load per-channel scale and shift
    scale_val = tl.load(scale_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    shift_val = tl.load(shift_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    
    # Compute: output = scale * input + shift
    output_val = scale_val * input_val + shift_val
    
    # Cast to output dtype and store
    if DTYPE == 1:
        result = output_val.to(tl.float16)
    elif DTYPE == 2:
        result = output_val.to(tl.bfloat16)
    else:
        result = output_val
    
    tl.store(output_ptr + offs, result, mask=mask)


@torch.fx.wrap
def _bn_only(input_tensor, scale, shift):
    output = torch.empty_like(input_tensor)
    
    B, C, H, W = input_tensor.shape
    HW = H * W
    CHW = C * HW
    total_elements = B * C * H * W
    
    dtype_code = _get_dtype_code(input_tensor.dtype)
    
    BLOCK_SIZE = 1024
    grid = triton.cdiv(total_elements, BLOCK_SIZE)
    
    bn_kernel[(grid,)](
        input_ptr=input_tensor,
        scale_ptr=scale,
        shift_ptr=shift,
        output_ptr=output,
        total_elements=total_elements,
        C=C, HW=HW, CHW=CHW,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=dtype_code,
    )
    
    return output


# ============ AvgPool2D Kernel ============
# avg_pool2d(input, kernel_size=2, stride=2, padding=0, count_include_pad=True)
# output[b, c, oh, ow] = mean(input[b, c, oh*2:oh*2+2, ow*2:ow*2+2])

@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    B, C, H_in, W_in, H_out, W_out,
    HW_in, CHW_in, HW_out, CHW_out,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total = B * C * H_out * W_out
    mask = offs < total
    
    # Decode (b, c, oh, ow) from flat index for [B, C, H_out, W_out] layout
    b_idx = offs // CHW_out
    c_idx = (offs % CHW_out) // HW_out
    oh_idx = (offs % HW_out) // W_out
    ow_idx = offs % W_out
    
    # Input positions for 2x2 average pooling with stride 2
    ih = oh_idx * 2
    iw = ow_idx * 2
    
    # Base offset for this (b, c) in input tensor
    base = b_idx * CHW_in + c_idx * HW_in
    
    # Load 4 input values for the 2x2 window
    v00 = tl.load(input_ptr + base + ih * W_in + iw, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(input_ptr + base + ih * W_in + iw + 1, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(input_ptr + base + (ih + 1) * W_in + iw, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(input_ptr + base + (ih + 1) * W_in + iw + 1, mask=mask, other=0.0).to(tl.float32)
    
    # Compute average
    avg = (v00 + v01 + v10 + v11) / 4.0
    
    # Cast to output dtype and store
    if DTYPE == 1:
        result = avg.to(tl.float16)
    elif DTYPE == 2:
        result = avg.to(tl.bfloat16)
    else:
        result = avg
    
    tl.store(output_ptr + offs, result, mask=mask)


@torch.fx.wrap
def _avg_pool2d(input_tensor):
    B, C, H_in, W_in = input_tensor.shape
    H_out = H_in // 2
    W_out = W_in // 2
    
    output = torch.empty((B, C, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    HW_in = H_in * W_in
    CHW_in = C * HW_in
    HW_out = H_out * W_out
    CHW_out = C * HW_out
    total = B * C * H_out * W_out
    
    dtype_code = _get_dtype_code(input_tensor.dtype)
    
    BLOCK_SIZE = 1024
    grid = triton.cdiv(total, BLOCK_SIZE)
    
    avg_pool2d_kernel[(grid,)](
        input_ptr=input_tensor,
        output_ptr=output,
        B=B, C=C, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        HW_in=HW_in, CHW_in=CHW_in, HW_out=HW_out, CHW_out=CHW_out,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=dtype_code,
    )
    
    return output


# ============ Dispatch Wrapper ============
# All pass files share this same replacement_func via routing technique

@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    args = args[:-1]
    
    if route == "conv1x1_bn":
        return _conv1x1_bn(*args)
    elif route == "bn_only":
        return _bn_only(*args)
    elif route == "avg_pool2d":
        return _avg_pool2d(*args)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper