import torch
import triton
import triton.language as tl

# Pattern to match: Add -> LayerNorm -> Slice -> Reshape(1,...) -> Permute -> ReLU
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1280,), in_1, in_0, 1e-06)
    tmp_4 = tmp_3[slice(None, None, None), slice(0, None, None)]
    tmp_5 = tmp_4.reshape(1, 16, 12, -1)
    tmp_6 = tmp_5.permute(0, 3, 1, 2)
    tmp_7 = torch.nn.functional.relu(tmp_6)
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_layernorm_reshape_permute_relu_kernel(
    in_2_ptr, in_3_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    B, SEQ, FEAT,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one 1280-element vector to normalize)
    row_idx = tl.program_id(0)
    b = row_idx // SEQ
    seq = row_idx % SEQ
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < FEAT
    
    # Load inputs - contiguous layout [B, SEQ, FEAT]
    in_offset = row_idx * FEAT + col_offsets
    x2 = tl.load(in_2_ptr + in_offset, mask=mask, other=0.0)
    x3 = tl.load(in_3_ptr + in_offset, mask=mask, other=0.0)
    
    # Add
    x = x2 + x3
    
    # Layer norm - compute mean
    mean = tl.sum(x, axis=0) / FEAT
    
    # Compute variance (mask to avoid including padding in variance)
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / FEAT
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Apply normalization: (x - mean) / std * weight + bias
    normalized = diff * inv_std * weight + bias
    
    # Apply ReLU
    output = tl.maximum(normalized, 0.0)
    
    # Write to output with permuted layout
    # Input index: (b, seq, feat)
    # Output shape after reshape+permute: [B, FEAT, 16, 12]
    # Output index: (b, feat, seq//12, seq%12)
    seq_h = seq // 12
    seq_w = seq % 12
    # Output is contiguous with strides: [FEAT*16*12, 16*12, 12, 1]
    out_offset = b * (FEAT * 16 * 12) + col_offsets * (16 * 12) + seq_h * 12 + seq_w
    tl.store(out_ptr + out_offset, output, mask=mask)

@torch.fx.wrap
def fused_add_layernorm_reshape_permute_relu(in_0, in_1, in_2, in_3):
    B = in_2.shape[0]
    SEQ = 192
    FEAT = 1280
    eps = 1e-06
    BLOCK_SIZE = 2048  # Power of 2 >= 1280
    
    # Output shape: [B, FEAT, 16, 12]
    out = torch.empty((B, FEAT, 16, 12), dtype=in_2.dtype, device=in_2.device)
    
    num_rows = B * SEQ
    fused_add_layernorm_reshape_permute_relu_kernel[(num_rows,)](
        in_2, in_3,
        in_1, in_0,  # weight, bias
        out,
        B, SEQ, FEAT,
        eps,
        BLOCK_SIZE,
    )
    
    return (out,)

def replacement_func():
    return fused_add_layernorm_reshape_permute_relu