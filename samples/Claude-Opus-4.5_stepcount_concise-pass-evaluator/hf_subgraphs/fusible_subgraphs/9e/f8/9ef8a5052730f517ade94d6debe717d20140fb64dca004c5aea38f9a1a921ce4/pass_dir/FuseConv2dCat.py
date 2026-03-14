import torch
import triton
import triton.language as tl


def pattern(weight, x, y):
    """
    Pattern: conv2d followed by cat along channel dimension
    """
    conv_out = torch.conv2d(x, weight, None, (1, 1), (1, 1), (1, 1), 1)
    result = torch.cat((conv_out, y), 1)
    return result


def replacement_args(weight, x, y):
    return (weight, x, y)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    batch, in_c, in_h, in_w,
    out_c, out_h, out_w,
    kh, kw,
    pad_h, pad_w, stride_h, stride_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_oc, weight_stride_ic, weight_stride_kh, weight_stride_kw,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Implicit GEMM for conv2d:
    - M = batch * out_h * out_w
    - N = out_c
    - K = in_c * kh * kw
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    rm = m_start + tl.arange(0, BLOCK_M)
    rn = n_start + tl.arange(0, BLOCK_N)
    
    out_hw = out_h * out_w
    batch_idx = rm // out_hw
    rem = rm % out_hw
    oh_idx = rem // out_w
    ow_idx = rem % out_w
    
    kh_kw = kh * kw
    
    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        
        ic_idx = rk // kh_kw
        rem_k = rk % kh_kw
        kh_idx = rem_k // kw
        kw_idx = rem_k % kw
        
        ih_idx = oh_idx[:, None] * stride_h - pad_h + kh_idx[None, :]
        iw_idx = ow_idx[:, None] * stride_w - pad_w + kw_idx[None, :]
        
        valid_h = (ih_idx >= 0) & (ih_idx < in_h)
        valid_w = (iw_idx >= 0) & (iw_idx < in_w)
        valid_m = rm[:, None] < M
        valid_k = rk[None, :] < K
        valid_input = valid_h & valid_w & valid_m & valid_k
        
        input_offset = (batch_idx[:, None] * input_stride_b + 
                       ic_idx[None, :] * input_stride_c +
                       ih_idx * input_stride_h + 
                       iw_idx * input_stride_w)
        a = tl.load(input_ptr + input_offset, mask=valid_input, other=0.0)
        
        valid_n = rn[None, :] < N
        valid_k2 = rk[:, None] < K
        valid_weight = valid_k2 & valid_n
        weight_offset = (rn[None, :] * weight_stride_oc + 
                        ic_idx[:, None] * weight_stride_ic +
                        kh_idx[:, None] * weight_stride_kh + 
                        kw_idx[:, None] * weight_stride_kw)
        b = tl.load(weight_ptr + weight_offset, mask=valid_weight, other=0.0)
        
        acc += tl.dot(a, b)
    
    valid_out = (rm[:, None] < M) & (rn[None, :] < N)
    out_batch_idx = rm[:, None] // out_hw
    out_rem = rm[:, None] % out_hw
    out_oh = out_rem // out_w
    out_ow = out_rem % out_w
    
    output_offset = (out_batch_idx * output_stride_b + 
                    rn[None, :] * output_stride_c +
                    out_oh * output_stride_h + 
                    out_ow * output_stride_w)
    tl.store(output_ptr + output_offset, acc, mask=valid_out)


@triton.jit
def copy_y_kernel(
    src_ptr, dst_ptr,
    n_elements,
    batch, channels, height, width,
    src_stride_b, src_stride_c, src_stride_h, src_stride_w,
    dst_stride_b, dst_stride_c, dst_stride_h, dst_stride_w,
    channel_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    hw = height * width
    chw = channels * hw
    
    b_idx = offsets // chw
    rem = offsets % chw
    c_idx = rem // hw
    rem2 = rem % hw
    h_idx = rem2 // width
    w_idx = rem2 % width
    
    src_offset = b_idx * src_stride_b + c_idx * src_stride_c + h_idx * src_stride_h + w_idx * src_stride_w
    data = tl.load(src_ptr + src_offset, mask=mask)
    
    dst_offset = b_idx * dst_stride_b + (c_idx + channel_offset) * dst_stride_c + h_idx * dst_stride_h + w_idx * dst_stride_w
    tl.store(dst_ptr + dst_offset, data, mask=mask)


@torch.fx.wrap
def fused_conv2d_cat(weight, x, y):
    batch = x.shape[0]
    in_c = x.shape[1]
    in_h = x.shape[2]
    in_w = x.shape[3]
    
    out_c = weight.shape[0]
    kh = weight.shape[2]
    kw = weight.shape[3]
    
    pad_h, pad_w = 1, 1
    stride_h, stride_w = 1, 1
    
    out_h = (in_h + 2 * pad_h - kh) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kw) // stride_w + 1
    
    out_channels_y = y.shape[1]
    total_out_channels = out_c + out_channels_y
    
    output = torch.empty(batch, total_out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    M = batch * out_h * out_w
    N = out_c
    K = in_c * kh * kw
    
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    y_contig = y.contiguous()
    
    def grid_conv(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    conv2d_implicit_gemm_kernel[grid_conv](
        x_contig, weight_contig, output,
        M, N, K,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        kh, kw,
        pad_h, pad_w, stride_h, stride_w,
        x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
        weight_contig.stride(0), weight_contig.stride(1), weight_contig.stride(2), weight_contig.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    # Copy y
    n_y = y_contig.numel()
    BLOCK_SIZE = 2048
    grid_copy = ((n_y + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    copy_y_kernel[grid_copy](
        y_contig, output,
        n_y,
        batch, out_channels_y, out_h, out_w,
        y_contig.stride(0), y_contig.stride(1), y_contig.stride(2), y_contig.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        out_c,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_conv2d_cat