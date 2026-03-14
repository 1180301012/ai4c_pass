import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, -1)
    tmp_2 = tmp_1.view(1, -1, 1, 1)
    tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def softmax_weighted_sum_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    B, C, H, W,
    stride_in0_b, stride_in0_k, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_k, stride_in1_x, stride_in1_c,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total = B * C * H * W
    mask = offsets < total
    
    # Decompose linear index into (b, c, h, w)
    w = offsets % W
    h = (offsets // W) % H
    c = (offsets // (W * H)) % C
    b = offsets // (C * H * W)
    
    # Load softmax inputs: in_1[b, k, 0, c] for k=0,1
    # in_1 shape: [B, 2, 1, C]
    in1_idx_0 = b * stride_in1_b + 0 * stride_in1_k + 0 * stride_in1_x + c * stride_in1_c
    in1_idx_1 = b * stride_in1_b + 1 * stride_in1_k + 0 * stride_in1_x + c * stride_in1_c
    
    x0 = tl.load(in_1_ptr + in1_idx_0, mask=mask, other=0.0)
    x1 = tl.load(in_1_ptr + in1_idx_1, mask=mask, other=0.0)
    
    # Compute softmax with numerical stability
    max_x = tl.maximum(x0, x1)
    exp0 = tl.exp(x0 - max_x)
    exp1 = tl.exp(x1 - max_x)
    sum_exp = exp0 + exp1
    w0 = exp0 / sum_exp
    w1 = exp1 / sum_exp
    
    # Load in_0 values: in_0[b, k, c, h, w] for k=0,1
    # in_0 shape: [B, 2, C, H, W]
    in0_idx_0 = b * stride_in0_b + 0 * stride_in0_k + c * stride_in0_c + h * stride_in0_h + w * stride_in0_w
    in0_idx_1 = b * stride_in0_b + 1 * stride_in0_k + c * stride_in0_c + h * stride_in0_h + w * stride_in0_w
    
    v0 = tl.load(in_0_ptr + in0_idx_0, mask=mask, other=0.0)
    v1 = tl.load(in_0_ptr + in0_idx_1, mask=mask, other=0.0)
    
    # Compute weighted sum
    result = w0 * v0 + w1 * v1
    
    # Store result
    out_idx = b * stride_out_b + c * stride_out_c + h * stride_out_h + w * stride_out_w
    tl.store(out_ptr + out_idx, result, mask=mask)

@torch.fx.wrap
def softmax_weighted_sum(in_0, in_1):
    B, K, C, H, W = in_0.shape
    
    out = torch.empty(B, C, H, W, device=in_0.device, dtype=in_0.dtype)
    
    total_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    softmax_weighted_sum_kernel[(num_programs,)](
        in_0, in_1, out,
        B, C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3), in_0.stride(4),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return softmax_weighted_sum