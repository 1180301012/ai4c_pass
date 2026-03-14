import torch
import triton
import triton.language as tl

def pattern(attention_mask, slice_dims, expand_batch, expand_seq):
    # tmp_9 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_9 = attention_mask[slice(None, None, None), None, None, slice(None, None, None)]
    # tmp_10 = tmp_9.expand(batch_size, 1, seq_len, seq_len)
    tmp_10 = tmp_9.expand(expand_batch, 1, expand_seq, expand_seq)
    return tmp_10

def replacement_args(attention_mask, slice_dims, expand_batch, expand_seq):
    return (attention_mask, slice_dims, expand_batch, expand_seq)

@triton.jit
def optimized_expand_kernel(
    input_ptr,
    output_ptr,
    input_batch,
    input_seq,
    output_batch,
    output_seq,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Expand input tensor from [batch, seq] to [output_batch, 1, output_seq, output_seq]
    # where the input is broadcasted across the dimensions
    
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch_idx = output_idx // (output_seq * output_seq * 1)
    remainder = output_idx % (output_seq * output_seq * 1)
    seq1_idx = remainder // (output_seq * 1)
    seq2_idx = remainder % output_seq
    
    mask = (output_idx < (output_batch * 1 * output_seq * output_seq))
    
    # Calculate input indices (broadcasting pattern)
    input_batch_idx = batch_idx % input_batch
    input_seq_idx = seq2_idx % input_seq
    
    # Calculate input offset
    input_offset = input_batch_idx * input_seq * hidden_size + input_seq_idx * hidden_size
    
    # Calculate output offset
    output_offset = batch_idx * 1 * output_seq * output_seq * hidden_size + seq1_idx * output_seq * output_seq * hidden_size + seq2_idx * output_seq * hidden_size
    
    # Load input value (we're broadcasting, so each input element becomes many output elements)
    input_value = tl.load(input_ptr + input_offset)
    
    # Store in the output at the appropriate positions
    # This effectively broadcasts: [batch, seq] -> [output_batch, 1, output_seq, output_seq]
    tl.store(output_ptr + output_offset, input_value, mask=mask)

@torch.fx.wrap
def optimized_slice_expand(attention_mask, expand_batch, expand_seq):
    original_shape = attention_mask.shape
    hidden_size = 1  # The attention mask is usually 2D [batch, seq], but we need to infer from context
    
    # Create output tensor with expanded shape
    output_shape = (expand_batch, 1, expand_seq, expand_seq)
    output = torch.empty(output_shape, dtype=attention_mask.dtype, device=attention_mask.device)
    
    # Get input dimensions
    input_batch, input_seq = original_shape
    
    # Calculate total elements
    n_elements = expand_batch * expand_seq * expand_seq
    
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # If the input is just [batch, seq] and we're expanding to [expand_batch, 1, expand_seq, expand_seq],
    # we can optimize by directly computing the broadcasted values
    if len(original_shape) == 2:
        # Create a more efficient implementation for the specific case
        @triton.jit
        def expand_2d_to_4d_kernel(
            input_ptr,
            output_ptr,
            input_batch,
            input_seq,
            output_batch,
            output_seq,
            BLOCK_SIZE: tl.constexpr,
        ):
            # Each program handles one output element
            idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            batch = idx // (output_seq * output_seq)
            remainder = idx % (output_seq * output_seq)
            seq1 = remainder // output_seq
            seq2 = remainder % output_seq
            
            mask = idx < (output_batch * output_seq * output_seq)
            
            # Map output coordinates to input coordinates with broadcasting
            input_batch_idx = batch % input_batch
            input_seq_idx = seq2 % input_seq
            
            input_offset = input_batch_idx * input_seq
            output_offset = batch * (output_seq * output_seq) + seq1 * output_seq + seq2
            
            input_value = tl.load(input_ptr + input_offset + input_seq_idx)
            tl.store(output_ptr + output_offset, input_value, mask=mask)
        
        # Reshape input to 2D and output to 3D for easier kernel handling
        input_2d = attention_mask.reshape(-1)
        output_3d = output.reshape(output_batch, output_seq, output_seq)
        
        expand_2d_to_4d_kernel[(num_programs,)](
            input_ptr=input_2d,
            output_ptr=output_3d,
            input_batch=input_batch,
            input_seq=input_seq,
            output_batch=output_batch,
            output_seq=output_seq,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output_3d.reshape(output_shape)
    else:
        # Fallback to original implementation for other cases
        return attention_mask[slice(None, None, None), None, None, slice(None, None, None)].expand(
            expand_batch, 1, expand_seq, expand_seq
        )

def replacement_func():
    def slice_expand_wrapper(attention_mask, slice_dims, expand_batch, expand_seq):
        return optimized_slice_expand(attention_mask, expand_batch, expand_seq)
    return slice_expand_wrapper