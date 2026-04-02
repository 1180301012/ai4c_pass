import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple softmax pattern test"""
    return torch.nn.functional.softmax(x, 2, _stacklevel=5)

def replacement_args(x):
    return (x,)

@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple softmax kernel"""
    row_idx = tl.program_id(0)
    
    # Calculate pointers for current row
    x_row = x_ptr + row_idx * n_cols
    output_row = output_ptr + row_idx * n_cols
    
    # Load row data
    x = tl.load(x_row + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=float('-inf'))
    
    # Find max for numerical stability
    max_val = tl.max(x)
    
    # Compute softmax
    exp_x = tl.exp(x - max_val)
    exp_sum = tl.sum(exp_x)
    softmax_x = exp_x / exp_sum
    
    # Store results
    tl.store(output_row + tl.arange(0, BLOCK_SIZE), softmax_x, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

@torch.fx.wrap
def simple_softmax(x):
    """Simple softmax wrapper"""
    if x.dim() == 3 and x.shape[1] == 1:  # [batch, 1, seq_len]
        batch_size, _, seq_len = x.shape
        output = torch.empty_like(x)
        
        BLOCK_SIZE = min(1024, seq_len)
        grid_size = (batch_size,)
        
        softmax_kernel[grid_size](
            x,
            output,
            batch_size,
            seq_len,
            BLOCK_SIZE,
        )
        
        return output
    else:
        # For unsupported shapes, reshape to match expected pattern and reshape back
        original_shape = x.shape
        if x.dim() == 3:
            # Reshape [batch, features, seq_len] to [batch*features, seq_len] for processing
            batch, features, seq_len = x.shape
            x_reshaped = x.reshape(batch * features, seq_len)
            output_reshaped = torch.empty_like(x_reshaped)
            
            BLOCK_SIZE = min(1024, seq_len)
            grid_size = (batch * features,)
            
            softmax_kernel[grid_size](
                x_reshaped,
                output_reshaped,
                batch * features,
                seq_len,
                BLOCK_SIZE,
            )
            
            # Reshape back to original dimensions
            return output_reshaped.reshape(original_shape)
        else:
            # Fallback for other shapes
            raise NotImplementedError(f"Unsupported shape for softmax optimization: {original_shape}")

def replacement_func():
    return simple_softmax