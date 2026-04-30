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
    mean = tl.sum(emb_sum, axis=0) / hidden_size
    
    # Compute variance
    diff = emb_sum - mean
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


def pattern(in_0):
    tmp_5 = in_0.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 *= -3.4028234663852886e+38
    tmp_8 = tmp_6.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    return tmp_9


def replacement_args(in_0):
    return (in_0, "mask_creation")


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    
    if route == "mask_creation":
        input_ids = args[0]
        seq_len = input_ids.numel()
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        
        batch_size = 1
        mask = torch.empty((batch_size, 1, 1, seq_len), dtype=torch.float32, device=input_ids.device)
        
        num_programs = triton.cdiv(seq_len, BLOCK_SIZE)
        
        mask_creation_kernel[(num_programs,)](
            input_ids_ptr=input_ids,
            mask_ptr=mask,
            seq_len=seq_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return mask
    
    elif route == "emb_layernorm_768":
        input_ids = args[0]
        word_emb = args[1]
        pos_emb = args[2]
        ln_weight = args[3]
        ln_bias = args[4]
        
        seq_len = input_ids.shape[-1]
        hidden_size = 768
        pos_offset = 2
        BLOCK_SIZE = 1024
        
        if word_emb.dtype == torch.bfloat16:
            DTYPE_CODE = 1
            output_dtype = torch.bfloat16
        else:
            DTYPE_CODE = 0
            output_dtype = torch.float16
        
        output = torch.empty((1, seq_len, hidden_size), dtype=output_dtype, device=input_ids.device)
        
        fused_emb_layernorm_kernel[(seq_len,)](
            input_ids_ptr=input_ids,
            word_emb_ptr=word_emb,
            pos_emb_ptr=pos_emb,
            ln_weight_ptr=ln_weight,
            ln_bias_ptr=ln_bias,
            output_ptr=output,
            seq_len=seq_len,
            hidden_size=hidden_size,
            pos_offset=pos_offset,
            BLOCK_SIZE=BLOCK_SIZE,
            DTYPE_CODE=DTYPE_CODE,
        )
        
        return output
    
    elif route == "emb_layernorm_32":
        input_ids = args[0]
        word_emb = args[1]
        pos_emb = args[2]
        ln_weight = args[3]
        ln_bias = args[4]
        
        seq_len = input_ids.shape[-1]
        hidden_size = 32
        pos_offset = 2
        BLOCK_SIZE = 64
        
        if word_emb.dtype == torch.bfloat16:
            DTYPE_CODE = 1
            output_dtype = torch.bfloat16
        else:
            DTYPE_CODE = 0
            output_dtype = torch.float16
        
        output = torch.empty((1, seq_len, hidden_size), dtype=output_dtype, device=input_ids.device)
        
        fused_emb_layernorm_kernel[(seq_len,)](
            input_ids_ptr=input_ids,
            word_emb_ptr=word_emb,
            pos_emb_ptr=pos_emb,
            ln_weight_ptr=ln_weight,
            ln_bias_ptr=ln_bias,
            output_ptr=output,
            seq_len=seq_len,
            hidden_size=hidden_size,
            pos_offset=pos_offset,
            BLOCK_SIZE=BLOCK_SIZE,
            DTYPE_CODE=DTYPE_CODE,
        )
        
        return output
    
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper