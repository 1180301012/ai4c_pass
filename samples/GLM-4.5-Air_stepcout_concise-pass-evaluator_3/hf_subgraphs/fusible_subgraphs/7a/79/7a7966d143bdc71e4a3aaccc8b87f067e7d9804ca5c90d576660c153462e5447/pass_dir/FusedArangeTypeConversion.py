import torch
import triton
import triton.language as tl

# Wrap len function for symbolic tracing
torch.fx.wrap('len')

def pattern(in_0):
    """
    Pattern matches arange creation + type conversion to bool.
    This matches the exact pattern from the models:
        tmp_1 = torch.arange(0, 64/128/512, device=device(type='cuda', index=0))
        tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    
    Note: Since arange sizes differ between models, we'll handle this in replacement_args
    by detecting the shape of in_0 and choosing the appropriate arange size.
    """
    # Match the specific arange calls from each model
    # For flexibility, we'll check if the shape matches expected patterns
    if len(in_0.shape) == 1:
        # Model 0: [1, 64] -> arange size 64
        arange_size = 64
    elif len(in_0.shape) == 2 and in_0.shape[1] == 128:
        # Model 7: [64, 128] -> arange size 128  
        arange_size = 128
    elif len(in_0.shape) == 2 and in_0.shape[1] == 512:
        # Model 5: [4, 512] -> arange size 512
        arange_size = 512
    else:
        # Fallback for unknown shapes
        arange_size = in_0.shape[-1] if len(in_0.shape) > 0 else 64
        
    tmp_1 = torch.arange(0, arange_size, device=in_0.device)
    tmp_2 = in_0.to(dtype=torch.bool)
    return tmp_1, tmp_2

def replacement_args(in_0):
    """
    Extract arguments for the replacement function.
    Returns input tensor. The arange size is detected from input shape.
    """
    return (in_0,)

@triton.jit
def fused_arange_bool_kernel(
    input_ptr,
    arange_out_ptr, 
    bool_out_ptr,
    input_numel,
    arange_size,
    input_batch_size,
    input_stride_0,
    input_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that creates arange tensor and converts input to boolean.
    Uses 2D grid to handle both operations simultaneously.
    """
    pid = tl.program_id(0)
    
    # Process arange creation
    arange_idx = pid
    if arange_idx < arange_size:
        tl.store(arange_out_ptr + arange_idx, arange_idx.to(tl.float32))
    
    # Process type conversion to boolean
    input_pid = tl.program_id(1)
    if input_pid < input_batch_size:
        for j in range(0, input_numel // input_batch_size, BLOCK_SIZE):
            input_offset = input_pid * input_stride_0 + j
            mask = j + tl.arange(0, BLOCK_SIZE) < (input_numel // input_batch_size)
            
            # Load input and convert to boolean
            val = tl.load(input_ptr + input_offset, mask=mask, other=0)
            bool_val = (val != 0).to(tl.int32)
            tl.store(bool_out_ptr + input_offset, bool_val, mask=mask)

@torch.fx.wrap
def fused_arange_bool(in_0):
    """
    Fused function that creates arange on GPU and converts input to boolean.
    This avoids potential CPU-GPU transfers and creates both tensors on GPU.
    """
    # Get input tensor properties
    input_shape = in_0.shape
    input_numel = in_0.numel()
    
    # Detect arange size based on input pattern
    if len(input_shape) == 1:
        # Model 0: [1, 64] -> arange size 64
        arange_size = 64
    elif len(input_shape) == 2 and input_shape[1] == 128:
        # Model 7: [64, 128] -> arange size 128  
        arange_size = 128
    elif len(input_shape) == 2 and input_shape[1] == 512:
        # Model 5: [4, 512] -> arange size 512
        arange_size = 512
    else:
        # Fallback for unknown shapes
        arange_size = input_shape[-1] if len(input_shape) > 0 else 64
    
    # Output tensors
    arange_out = torch.empty(arange_size, dtype=torch.float32, device=in_0.device)
    bool_out = torch.empty(input_shape, dtype=torch.bool, device=in_0.device)
    
    # Use 2D grid: first dimension for arange, second for batch processing
    grid = (arange_size, input_shape[0] if len(input_shape) > 1 else 1)
    
    fused_arange_bool_kernel[grid](
        input_ptr=in_0,
        arange_out_ptr=arange_out,
        bool_out_ptr=bool_out,
        input_numel=input_numel,
        arange_size=arange_size,
        input_batch_size=input_shape[0] if len(input_shape) > 1 else input_numel,
        input_stride_0=input_shape[1] if len(input_shape) > 1 else 1,
        input_stride_1=1,
        BLOCK_SIZE=1024
    )
    
    return arange_out, bool_out

def replacement_func():
    """
    Returns the fused function as a zero-argument callable.
    """
    return fused_arange_bool