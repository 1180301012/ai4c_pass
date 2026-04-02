import torch
import triton
import triton.language as tl

def expand_concat_pattern(cls_token, conv_features, detection_tokens):
    """
    Pattern matching for the expand + concatenate operations:
    tmp_10 = in_3.expand(1, -1, -1)           # cls_token expand
    tmp_11 = in_4.expand(1, -1, -1)           # detection_tokens expand  
    tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim = 1)  # concatenate all
    
    Note: Since torch.cat is forbidden in pattern functions, we return a simpler pattern
    that matches the individual operations.
    """
    # Simple pattern that matches the individual expand operations
    expanded_cls = cls_token.expand(-1, -1, conv_features.size(1)[-1])  # tmp_10
    expanded_detection = detection_tokens.expand_as(expanded_cls)  # tmp_11
    
    # Return expanded tensors (concatenation would be blocked)
    return expanded_cls, expanded_detection, conv_features

def replacement_args(cls_token, conv_features, detection_tokens):
    """Extract all three tensors needed for the replacement"""
    return (cls_token, conv_features, detection_tokens)

@triton.jit
def fused_expand_concat_kernel(
    cls_token_ptr,           # [1, 1, 32] cls token
    conv_features_ptr,       # [1, 225, 32] convolution features  
    detection_tokens_ptr,    # [1, 10, 32] detection tokens
    output_ptr,             # [1, 236, 32] concatenated output
    batch_size,
    seq_len_conv,           # length of conv features (225)
    seq_len_cls,            # length of cls tokens (1) 
    seq_len_detection,      # length of detection tokens (10)
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that fuses expand and concatenate operations.
    Directly computes the concatenation without explicit intermediate expand operations.
    """
    pid = tl.program_id(0)
    
    # Calculate global position in output tensor
    out_seq_idx = pid // ((hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE)
    elem_idx = pid % ((hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    mask = (out_seq_idx < (seq_len_cls + seq_len_conv + seq_len_detection)) and (elem_idx < hidden_dim)
    
    if mask:
        batch_offset = batch_size * ((seq_len_cls + seq_len_conv + seq_len_detection) * hidden_dim)
        output_offset = batch_offset + out_seq_idx * hidden_dim + elem_idx
        
        # Determine which tensor this output position belongs to
        if out_seq_idx < seq_len_cls:  # CLS token section
            cls_offset = batch_size * (seq_len_cls * hidden_dim)
            input_offset = cls_offset + out_seq_idx * hidden_dim + elem_idx
            
        elif out_seq_idx < seq_len_cls + seq_len_conv:  # Conv features section
            conv_offset = batch_size * (seq_len_conv * hidden_dim)
            conv_seq_idx = out_seq_idx - seq_len_cls
            input_offset = conv_offset + conv_seq_idx * hidden_dim + elem_idx
            
        else:  # Detection tokens section  
            detection_offset = batch_size * (seq_len_detection * hidden_dim)
            det_seq_idx = out_seq_idx - seq_len_cls - seq_len_conv
            input_offset = detection_offset + det_seq_idx * hidden_dim + elem_idx
        
        # Load from appropriate input tensor and store to output
        val = tl.load(cls_token_ptr + input_offset, mask=True, other=0.0)
        tl.store(output_ptr + output_offset, val, mask=bool(mask))

def fused_expand_concat_optimized(cls_token, conv_features, detection_tokens):
    """
    Optimized function that fuses expand and concatenate operations using a single Triton kernel,
    avoiding the memory overhead of intermediate expand operations.
    """
    batch_size = cls_token.shape[0]
    seq_len_cls = cls_token.shape[1]
    seq_len_conv = conv_features.shape[1] 
    seq_len_detection = detection_tokens.shape[1]
    hidden_dim = cls_token.shape[2]
    
    total_seq_len = seq_len_cls + seq_len_conv + seq_len_detection
    
    # Create output tensor
    output = torch.empty((batch_size, total_seq_len, hidden_dim), dtype=cls_token.dtype)
    
    # Launch optimized kernel
    BLOCK_SIZE = 1024
    total_elements = batch_size * total_seq_len * hidden_dim
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_expand_concat_kernel[grid](
        cls_token,
        conv_features, 
        detection_tokens,
        output,
        batch_size,
        seq_len_conv,
        seq_len_cls,
        seq_len_detection,
        hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return output and expanded versions (for compatibility with expected outputs)
    expanded_cls = cls_token.expand(-1, -1, conv_features.size(1))
    expanded_detection = detection_tokens.expand(-1, -1, conv_features.size(1))
    
    return output, expanded_cls, expanded_detection

@torch.fx.wrap
def fused_expand_concat_wrapper(cls_token, conv_features, detection_tokens):
    """Wrapper function that calls the fused expand+concat implementation"""
    return fused_expand_concat_optimized(cls_token, conv_features, detection_tokens)

def replacement_func():
    """Returns the replacement function"""
    return fused_expand_concat_wrapper