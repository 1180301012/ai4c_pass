import torch
import triton
import triton.language as tl


@triton.jit
def mask_creation_kernel(
    input_ids_ptr,
    mask_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < seq_len
    
    # Load input_ids (int64)
    input_ids = tl.load(input_ids_ptr + offsets, mask=valid, other=0)
    
    # Check if padding (== 1)
    is_padding = input_ids == 1
    
    # Convert to float and multiply: 1.0 -> -FLT_MAX, 0.0 -> 0.0
    padding_float = is_padding.to(tl.float32)
    mask_values = padding_float * (-3.4028234663852886e+38)
    
    tl.store(mask_ptr + offsets, mask_values, mask=valid)


@triton.jit
def fused_emb_layernorm_kernel(
    input_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    output_ptr,
    seq_len,
    hidden_size,
    pos_offset,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_CODE: tl.constexpr,  # 0=float16, 1=bfloat16
):
    pid = tl.program_id(0)
    if pid >= seq_len:
        return
    
    # Read input_id at this position
    input_id = tl.load(input_ids_ptr + pid)
    
    # Position index (offset by pos_offset=2)
    pos_idx = pid + pos_offset
    
    offsets = tl.arange(0, BLOCK_SIZE)
    hmask = offsets < hidden_size
    hmask_f = hmask.to(tl.float32)
    
    # Load word embedding row (in original dtype, convert to float32)
    word_emb = tl.load(word_emb_ptr + input_id * hidden_size + offsets, mask=hmask, other=0.0).to(tl.float32)
    
    # Load position embedding row
    pos_emb = tl.load(pos_emb_ptr + pos_idx * hidden_size + offsets, mask=hmask, other=0.0).to(tl.float32)
    
    # Add embeddings
    emb_sum = word_emb + pos_emb
    
    # Layer norm: compute mean
    # Note: masked positions have 0.0 value, so they don't affect the sum
    # But we divide by hidden_size (not BLOCK_SIZE) for correct mean
    mean = tl.sum(emb_sum, axis=0) / hidden_size
    
    # Compute variance
    diff = emb_sum - mean
    # Zero out masked positions: for offsets >= hidden_size, diff = -mean
    # We need to mask these out for correct variance
    diff_masked = diff * hmask_f
    var = tl.sum(diff_masked * diff_masked, axis=0) / hidden_size
    
    # Normalize
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    normalized = diff_masked * rstd
    
    # Load weight and bias (in original dtype, convert to float32)
    ln_weight = tl.load(ln_weight_ptr + offsets, mask=hmask, other=1.0).to(tl.float32)
    ln_bias = tl.load(ln_bias_ptr + offsets, mask=hmask, other=0.0).to(tl.float32)
    
    # Apply affine transformation
    output = normalized * ln_weight + ln_bias
    
    # Convert back to original dtype and store
    if DTYPE_CODE == 1:  # bfloat16
        output_store = output.to(tl.bfloat16)
    else:  # float16
        output_store = output.to(tl.float16)
    
    tl.store(output_ptr + pid * hidden_size + offsets, output_store, mask=hmask)