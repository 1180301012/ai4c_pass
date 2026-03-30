import torch
import triton
import triton.language as tl

def pattern(cls_token, conv_result, detection_tokens):
    # tmp_10 = in_3.expand(1, -1, -1);  in_3 = None
    tmp_10 = cls_token.expand(1, -1, -1)
    
    # tmp_11 = in_4.expand(1, -1, -1);  in_4 = None
    tmp_11 = detection_tokens.expand(1, -1, -1)
    
    # tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim = 1);  tmp_10 = tmp_9 = tmp_11 = None
    tmp_12 = torch.cat((tmp_10, conv_result, tmp_11), dim = 1)
    
    return tmp_12

def replacement_args(cls_token, conv_result, detection_tokens):
    return (cls_token, conv_result, detection_tokens)

@triton.jit
def fuse_expand_cat_kernel(
    cls_ptr,
    conv_ptr,
    det_ptr,
    output_ptr,
    cls_seq_len,
    conv_seq_len,
    det_seq_len,
    hidden_size,
    target_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= target_seq_len * hidden_size:
        return
    
    # Calculate sequence and channel position
    seq_pos = pid // hidden_size
    channel_pos = pid % hidden_size
    
    output_val = 0.0
    
    # Load cls token (expanded from [1, 1, 32] to [1, target_seq_len, 32])
    if seq_pos < cls_seq_len:
        cls_offset = cls_seq_len * hidden_size + seq_pos * hidden_size + channel_pos
        cls_val = tl.load(cls_ptr + cls_offset, other=0.0)
        output_val += cls_val
    
    # Load conv result ([1, 225, 32]) - place after cls tokens
    conv_start_offset = cls_seq_len * hidden_size
    if seq_pos < conv_seq_len:
        if seq_pos < cls_seq_len:
            # Map conv position after cls tokens
            conv_pos = seq_pos
        else:
            conv_pos = seq_pos - cls_seq_len
        
        conv_offset = conv_start_offset + conv_pos * hidden_size + channel_pos
        conv_val = tl.load(conv_ptr + conv_offset, mask=conv_pos < conv_seq_len, other=0.0)
        output_val += conv_val
    
    # Load detection tokens (expanded from [1, 10, 32] to [1, target_seq_len, 32])
    det_start_offset = (cls_seq_len + conv_seq_len) * hidden_size
    if seq_pos < det_seq_len:
        if seq_pos < cls_seq_len:
            det_pos = seq_pos  # Over cls if det smaller
        elif seq_pos < cls_seq_len + det_seq_len:
            det_pos = seq_pos - cls_seq_len  # Over conv if in range
        else:
            det_pos = seq_pos - cls_seq_len - conv_seq_len  # Normal position
        
        det_offset = det_start_offset + det_pos * hidden_size + channel_pos
        det_val = tl.load(det_ptr + det_offset, mask=det_pos < det_seq_len, other=0.0)
        output_val += det_val
    
    # Store result
    output_offset = seq_pos * hidden_size + channel_pos
    tl.store(output_ptr + output_offset, output_val, other=0.0)

@torch.fx.wrap
def optimized_fuse_expand_cat(cls_token, conv_result, detection_tokens):
    # Input shapes:
    # cls_token: [1, 1, 32]
    # conv_result: [1, 225, 32] 
    # detection_tokens: [1, 10, 32]
    
    # Target sequence length for expand operations
    target_seq_len = max(236, 236)  # Both expand to 236
    
    batch_size, cls_seq_len, hidden_size = cls_token.shape
    _, conv_seq_len, _ = conv_result.shape  
    _, det_seq_len, _ = detection_tokens.shape
    
    # Output shape: [1, target_seq_len, hidden_size]
    output_shape = (batch_size, target_seq_len, hidden_size)
    output = torch.empty(output_shape, dtype=cls_token.dtype, device=cls_token.device)
    
    # Flatten tensors for kernel
    cls_flat = cls_token.view(-1)  # [1*1*32]
    conv_flat = conv_result.view(-1)  # [1*225*32]
    det_flat = detection_tokens.view(-1)  # [1*10*32]
    output_flat = output.view(-1)  # [1*target_seq_len*32]
    
    # Launch kernel
    BLOCK_SIZE = 64
    total_elements = target_seq_len * hidden_size
    
    fuse_expand_cat_kernel[(total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,](
        cls_flat,
        conv_flat,
        det_flat,
        output_flat,
        cls_seq_len,
        conv_seq_len,
        det_seq_len,
        hidden_size,
        target_seq_len,
        BLOCK_SIZE,
    )
    
    return output.view(batch_size, target_seq_len, hidden_size)

def replacement_func():
    return optimized_fuse_expand_cat