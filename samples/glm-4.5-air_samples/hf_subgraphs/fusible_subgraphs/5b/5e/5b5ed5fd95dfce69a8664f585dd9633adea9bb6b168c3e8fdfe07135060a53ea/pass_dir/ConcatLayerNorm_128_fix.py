import torch
import triton
import triton.language as tl

def pattern(concat_input, cls_token, norm_weight, norm_bias):
    concat_result = torch.cat((cls_token, concat_input), dim=1)
    norm_result = torch.nn.functional.layer_norm(concat_result, (128,), norm_weight, norm_bias, 1e-06)
    return concat_result, norm_result

def replacement_args(concat_input, cls_token, norm_weight, norm_bias):
    return (concat_input, cls_token, norm_weight, norm_bias)

@triton.jit
def concat_layer_norm_kernel(
    feat_ptr,
    cls_token_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    output_ptr,
    norm_output_ptr,
    batch_size,
    feat_length,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(batch_size * (feat_length + 1) * channels, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
    
    offset = pid * BLOCK_SIZE
    remaining = BLOCK_SIZE
    chunk_size = 64
    
    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        elem_idx = tl.arange(0, current_chunk)
        global_idx = offset + elem_idx
        
        mask = global_idx < batch_size * (feat_length + 1) * channels
        
        if tl.any(mask):
            batch_idx = global_idx // ((feat_length + 1) * channels)
            pos_idx = (global_idx // channels) % (feat_length + 1)
            channel_idx = global_idx % channels
            
            is_cls = (pos_idx == 0)
            feat_pos_idx = pos_idx - is_cls.to(tl.int32)
            
            if is_cls:
                cls_offset = batch_idx * channels + channel_idx
                feat_val = tl.load(cls_token_ptr + cls_offset, mask=mask, other=0.0)
            else:
                feat_offset = batch_idx * feat_length * channels + feat_pos_idx * channels + channel_idx
                feat_val = tl.load(feat_ptr + feat_offset, mask=mask, other=0.0)
            
            norm_weight_val = tl.load(norm_weight_ptr + channel_idx, mask=mask, other=1.0)
            norm_bias_val = tl.load(norm_bias_ptr + channel_idx, mask=mask, other=0.0)
            
            normalized_val = feat_val * norm_weight_val + norm_bias_val
            
            concat_offset = global_idx
            norm_offset = global_idx
            
            tl.store(output_ptr + concat_offset, feat_val, mask=mask)
            tl.store(norm_output_ptr + norm_offset, normalized_val, mask=mask)
        
        offset += current_chunk
        remaining -= current_chunk

@torch.fx.wrap
def optimized_concat_layer_norm(concat_input, cls_token, norm_weight, norm_bias):
    batch_size, feat_length, channels = concat_input.shape
    total_length = feat_length + 1
    
    if total_length * channels >= 12800:
        BLOCK_SIZE = 2048
    elif total_length * channels >= 6400:
        BLOCK_SIZE = 4096
    else:
        BLOCK_SIZE = 8192
    
    total_elements = batch_size * total_length * channels
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    concat_result = torch.empty((batch_size, total_length, channels), 
                              dtype=concat_input.dtype, device=concat_input.device)
    norm_result = torch.empty((batch_size, total_length, channels), 
                            dtype=concat_input.dtype, device=concat_input.device)
    
    concat_layer_norm_kernel[(num_programs,)](
        feat_ptr=concat_input,
        cls_token_ptr=cls_token,
        norm_weight_ptr=norm_weight,
        norm_bias_ptr=norm_bias,
        output_ptr=concat_result,
        norm_output_ptr=norm_result,
        batch_size=batch_size,
        feat_length=feat_length,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return concat_result, norm_result

def replacement_func():
    return optimized_concat_layer_norm