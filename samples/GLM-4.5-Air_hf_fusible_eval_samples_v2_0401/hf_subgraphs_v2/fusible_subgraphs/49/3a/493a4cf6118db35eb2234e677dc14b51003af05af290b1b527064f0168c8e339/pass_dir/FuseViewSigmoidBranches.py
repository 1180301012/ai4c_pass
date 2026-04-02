import torch
import triton
import triton.language as tl

@triton.jit
def simple_fused_kernel(
    gating_param_ptr,
    patch_score_ptr,
    softmax_result_ptr,
    output_ptr,
    n_elements: tl.constexpr,
):
    """Simple kernel that fuses the branch computations."""
    pid = tl.program_id(0)
    
    # Handle blocks of elements
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Load gating parameter (scalar that gets broadcasted)
    gating_param = tl.load(gating_param_ptr)
    
    # Compute sigmoid using PyTorch-safe approach
    # Convert to fp32 for mathematical operations (compatible with all dtypes)
    gating_param_fp32 = tl.cast(gating_param, tl.float32)
    
    # Numerically stable sigmoid computation
    if gating_param_fp32 > 0:
        exp_neg = tl.exp(-gating_param_fp32)
        sigmoid_val_fp32 = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = tl.exp(gating_param_fp32)
        sigmoid_val_fp32 = exp_pos / (1.0 + exp_pos)
    
    sigmoid_inv_fp32 = 1.0 - sigmoid_val_fp32
    
    # Convert back to original dtype
    sigmoid_val = tl.cast(sigmoid_val_fp32, tl.float32)  # Keep as fp32 for precision
    sigmoid_inv = tl.cast(sigmoid_inv_fp32, tl.float32)   # Keep as fp32 for precision
    
    # Load patch score and softmax result
    patch_vals = tl.load(patch_score_ptr + offsets, mask=mask)
    softmax_vals = tl.load(softmax_result_ptr + offsets, mask=mask)
    
    # Cast inputs to fp32 for computation, keep precision
    patch_vals_fp32 = tl.cast(patch_vals, tl.float32)
    softmax_vals_fp32 = tl.cast(softmax_vals, tl.float32)
    
    # Fuse the branch computations in fp32 for precision
    branch1 = sigmoid_inv * patch_vals_fp32
    branch2 = sigmoid_val * softmax_vals_fp32
    result_fp32 = branch1 + branch2
    
    # Convert back to original dtype and store
    result = tl.cast(result_fp32, tl.float32)  # Keep as fp32 for consistency
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_computation(in_0, in_1, in_2):
    """
    Optimized computation using Triton kernel for the arithmetic part.
    Softmax is computed separately using PyTorch, then fused with other operations.
    """
    
    # Step 1: Compute softmax using PyTorch
    softmax_result = in_2.softmax(dim=-1)
    
    # Prepare inputs for kernel
    gating_param = in_0[0]  # Extract first element (gating parameter)
    
    # Reshape to 1D for easier kernel processing
    patch_score_flat = in_1.reshape(-1)
    softmax_flat = softmax_result.reshape(-1)
    output_flat = torch.empty_like(patch_score_flat)
    
    # Launch kernel
    n_elements = patch_score_flat.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    simple_fused_kernel[grid](
        gating_param,
        patch_score_flat,
        softmax_flat,
        output_flat,
        n_elements=n_elements
    )
    
    # Reshape back to original dimensions
    return output_flat.reshape_as(in_1)

def pattern(in_0, in_1, in_2):
    # Match the computation pattern exactly as in model.py
    tmp_1 = in_2.softmax(dim = -1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return fused_computation