import torch
import triton
import triton.language as tl

# Pattern for graph 4: H=49, B=361, N=3 (zuppif_maskformer-swin-small-ade_start100)
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = tmp_0[in_3]
    tmp_2 = tmp_1.view(49, 49, -1)
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_6 = in_1 + tmp_5
    tmp_7 = tmp_6.view(1, 361, 3, 49, 49)
    tmp_8 = in_2.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(0)
    tmp_10 = tmp_7 + tmp_9
    tmp_11 = tmp_10.view(-1, 3, 49, 49)
    tmp_12 = torch.nn.functional.softmax(tmp_11, dim=-1)
    tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
    return (tmp_13,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['H'],
)
@triton.jit
def fused_bias_mask_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    output_ptr,
    B, num_heads, H,
    stride_in0_pos, stride_in0_head,
    stride_in1_b, stride_in1_h, stride_in1_row, stride_in1_col,
    stride_in2_b, stride_in2_row, stride_in2_col,
    stride_out_b, stride_out_h, stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    b = pid // (num_heads * H)
    head_row = pid % (num_heads * H)
    head = head_row // H
    row = head_row % H
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    
    attn_offset = b * stride_in1_b + head * stride_in1_h + row * stride_in1_row + offsets * stride_in1_col
    attn_row = tl.load(in_1_ptr + attn_offset, mask=mask, other=-float('inf'))
    
    idx_offset = row * H + offsets
    indices = tl.load(in_3_ptr + idx_offset, mask=mask, other=0)
    bias_offset = indices * stride_in0_pos + head * stride_in0_head
    bias_row = tl.load(in_0_ptr + bias_offset, mask=mask, other=0.0)
    
    attn_mask_offset = b * stride_in2_b + row * stride_in2_row + offsets * stride_in2_col
    attn_mask_row = tl.load(in_2_ptr + attn_mask_offset, mask=mask, other=0.0)
    
    x = attn_row + bias_row + attn_mask_row
    
    max_val = tl.max(x, axis=0)
    x_stable = x - max_val
    x_exp = tl.exp(x_stable)
    sum_exp = tl.sum(x_exp, axis=0)
    out = x_exp / sum_exp
    
    out_offset = b * stride_out_b + head * stride_out_h + row * stride_out_row + offsets * stride_out_col
    tl.store(output_ptr + out_offset, out, mask=mask)

@torch.fx.wrap
def fused_bias_mask_softmax(in_0, in_1, in_2, in_3):
    B, num_heads, H, _ = in_1.shape
    
    output = torch.empty_like(in_1)
    
    num_programs = B * num_heads * H
    
    fused_bias_mask_softmax_kernel[(num_programs,)](
        in_0, in_1, in_2, in_3, output,
        B, num_heads, H,
        in_0.stride(0), in_0.stride(1),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return (output,)

def replacement_func():
    return fused_bias_mask_softmax