import torch
import triton
import triton.language as tl

def pattern(in_3, in_2, in_1):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    tmp_4 = linear.reshape(1, 49, 8, -1)
    split = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_13 = tmp_10.transpose(-2, -1)
    return (tmp_9, tmp_12, tmp_13, tmp_11)

def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)

@triton.jit
def fused_linear_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    batch_size,
    seq_len,
    hidden_size,
    out1_channels,
    out2_channels,
    out3_channels,
    in_stride_batch,
    in_stride_seq,
    in_stride_hidden,
    weight_stride_out,
    weight_stride_hidden,
    bias_stride,
    out1_stride_batch,
    out1_stride_head,
    out1_stride_seq,
    out1_stride_channels,
    out2_stride_batch,
    out2_stride_head,
    out2_stride_channels,
    out2_stride_seq,
    out3_stride_batch,
    out3_stride_head,
    out3_stride_seq,
    out3_stride_channels,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr
):
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    batch_offset = batch_id * in_stride_batch
    seq_offset = seq_id * in_stride_seq
    
    for c1 in range(out1_channels):
        out1_offset = batch_offset * out1_stride_batch + 0 * out1_stride_head + seq_id * out1_stride_seq + c1 * out1_stride_channels
        total = 0.0
        for h in range(hidden_size):
            input_val = tl.load(in_ptr + batch_offset + seq_offset + h * in_stride_hidden)
            weight_val = tl.load(weight_ptr + c1 * weight_stride_out + h * weight_stride_hidden)
            total += input_val * weight_val
        bias_val = tl.load(bias_ptr + c1)
        total += bias_val
        tl.store(out1_ptr + out1_offset, total)

    for c2 in range(out2_channels):
        out2_offset = batch_offset * out2_stride_batch + 0 * out2_stride_head + c2 * out2_stride_channels + seq_id * out2_stride_seq
        total = 0.0
        for h in range(hidden_size):
            input_val = tl.load(in_ptr + batch_offset + seq_offset + h * in_stride_hidden)
            weight_val = tl.load(weight_ptr + (32 + c2) * weight_stride_out + h * weight_stride_hidden)
            total += input_val * weight_val
        bias_val = tl.load(bias_ptr + 32 + c2)
        total += bias_val
        tl.store(out2_ptr + out2_offset, total)

    for c3 in range(out3_channels):
        out3_offset = batch_offset * out3_stride_batch + 0 * out3_stride_head + seq_id * out3_stride_seq + c3 * out3_stride_channels
        total = 0.0
        for h in range(hidden_size):
            input_val = tl.load(in_ptr + batch_offset + seq_offset + h * in_stride_hidden)
            weight_val = tl.load(weight_ptr + (64 + c3) * weight_stride_out + h * weight_stride_hidden)
            total += input_val * weight_val
        bias_val = tl.load(bias_ptr + 64 + c3)
        total += bias_val
        tl.store(out3_ptr + out3_offset, total)

@torch.fx.wrap
def fused_linear(in_3, in_2, in_1):
    batch_size, seq_len, hidden_size = in_3.shape
    out1_channels = 32
    out2_channels = 32
    out3_channels = 128
    
    out1 = torch.empty(batch_size, 8, seq_len, out1_channels, dtype=in_3.dtype, device=in_3.device)
    out2 = torch.empty(batch_size, 8, out2_channels, seq_len, dtype=in_3.dtype, device=in_3.device)
    out3 = torch.empty(batch_size, 8, seq_len, out3_channels, dtype=in_3.dtype, device=in_3.device)
    
    in_stride_batch, in_stride_seq, in_stride_hidden = in_3.stride()
    weight_stride_out, weight_stride_hidden = in_2.stride()
    bias_stride = in_1.stride()
    out1_stride_batch, out1_stride_head, out1_stride_seq, out1_stride_channels = out1.stride()
    out2_stride_batch, out2_stride_head, out2_stride_channels, out2_stride_seq = out2.stride()
    out3_stride_batch, out3_stride_head, out3_stride_seq, out3_stride_channels = out3.stride()
    
    BLOCK_BATCH = 8
    BLOCK_SEQ = 16
    BLOCK_HIDDEN = 32
    grid = (batch_size // BLOCK_BATCH + 1, seq_len // BLOCK_SEQ + 1)
    
    fused_linear_kernel[grid](
        in_3,
        in_2,
        in_1,
        out1,
        out2,
        out3,
        batch_size,
        seq_len,
        hidden_size,
        out1_channels,
        out2_channels,
        out3_channels,
        in_stride_batch,
        in_stride_seq,
        in_stride_hidden,
        weight_stride_out,
        weight_stride_hidden,
        bias_stride,
        out1_stride_batch,
        out1_stride_head,
        out1_stride_seq,
        out1_stride_channels,
        out2_stride_batch,
        out2_stride_head,
        out2_stride_channels,
        out2_stride_seq,
        out3_stride_batch,
        out3_stride_head,
        out3_stride_seq,
        out3_stride_channels,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HIDDEN=BLOCK_HIDDEN
    )
    return out1, out2, out3

def replacement_func():
    def optimized_forward(in_0, in_1, in_2, in_3):
        out1, out2, out3 = fused_linear(in_3, in_2, in_1)
        in_0_gpu = in_0.to(device=in_0.device)
        return (out1, in_0_gpu, out2, out3)
    return optimized_forward