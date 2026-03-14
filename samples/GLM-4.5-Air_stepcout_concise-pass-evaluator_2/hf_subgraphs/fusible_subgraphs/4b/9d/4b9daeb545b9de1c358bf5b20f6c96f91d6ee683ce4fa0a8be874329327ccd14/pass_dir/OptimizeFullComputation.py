import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    in_0 += in_1
    tmp_0 = in_0
    tmp_0 += in_3
    tmp_1 = tmp_0
    tmp_0 = None
    tmp_2 = torch.nn.functional.relu(tmp_1, inplace=False)
    tmp_1 = None
    tmp_3 = in_2.chunk(2, dim=1)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_3 = None
    tmp_6 = tmp_2.chunk(2, dim=1)
    tmp_2 = None
    tmp_7 = tmp_6[0]
    tmp_8 = tmp_6[1]
    tmp_6 = None
    return (tmp_4, tmp_7, tmp_5, tmp_8)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel_add_relu_chunk(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    out_chunk1_ptr,
    out_relu_chunk1_ptr,
    out_chunk2_ptr,
    out_relu_chunk2_ptr,
    n_elements_common,
    n_elements_half_2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_common
    
    # Load common tensors (in_0, in_1, in_3 all same size)
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    in3 = tl.load(in3_ptr + offsets, mask=mask, other=0.0)
    
    # Fused addition and ReLU: (in0 + in1 + in3) > 0 ? (in0 + in1 + in3) : 0
    sum_val = in0 + in1 + in3
    relu_val = tl.where(sum_val > 0, sum_val, 0.0)
    
    # Store results for the first chunk (common size)
    tl.store(out_chunk1_ptr + offsets, relu_val, mask=mask)
    
    pid2 = tl.program_id(1)
    if pid2 == 0:
        # Process in_2 chunk (first half)
        in2_offsets = tl.arange(0, n_elements_half_2)
        in2_mask = in2_offsets < n_elements_half_2
        in2_chunk1 = tl.load(in2_ptr + in2_offsets, mask=in2_mask, other=0.0)
        tl.store(out_chunk2_ptr + in2_offsets, in2_chunk1, mask=in2_mask)
    
    pid3 = tl.program_id(2)
    if pid3 == 0:
        # Process in_2 chunk (second half)
        in2_start = n_elements_half_2
        in2_offsets = in2_start + tl.arange(0, n_elements_half_2)
        in2_mask = in2_offsets < n_elements_half_2 * 2
        in2_chunk2 = tl.load(in2_ptr + in2_offsets, mask=in2_mask, other=0.0)
        tl.store(out_chunk3_ptr + offsets, in2_chunk2, mask=in2_mask)

@triton.jit  
def alternative_fused_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr1, out_ptr2, out_ptr3, out_ptr4,
    n_elements_common,
    n_elements_half_2,
    BLOCK_SIZE: tl.constexpr,
):
    # Single kernel that computes everything
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.range(0, BLOCK_SIZE)
    mask = offsets < n_elements_common
    
    # Load inputs
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    in3 = tl.load(in3_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused addition + ReLU
    sum_val = in0 + in1 + in3
    relu_result = tl.maximum(sum_val, 0.0)
    
    # Store the two chunks of ReLU result
    block_size_half = BLOCK_SIZE // 2
    offset1 = block_start + tl.arange(0, block_size_half)
    offset2 = block_start + block_size_half + tl.arange(0, block_size_half)
    
    mask1 = offset1 < (n_elements_common // 2)
    mask2 = offset2 < (n_elements_common // 2)
    
    tl.store(out_ptr1 + offset1, relu_result[:block_size_half], mask=mask1)
    tl.store(out_ptr2 + offset2, relu_result[block_size_half:], mask=mask2)

@torch.fx.wrap
def optimized_full_computation(in_0, in_1, in_2, in_3):
    # Step 1: Fused addition + ReLU optimization
    def fused_add_relu(x, y, z):
        numel = x.numel()
        BLOCK_SIZE = 1024
        num_programs = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(x)
        
        # Triton kernel for fused addition + ReLU
        @triton.jit
        def kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
            
            sum_val = x + y + z
            relu_val = tl.maximum(sum_val, 0.0)
            
            tl.store(out_ptr + offsets, relu_val, mask=mask)
        
        kernel[(num_programs,)](
            x_ptr=x, y_ptr=y, z_ptr=z, out_ptr=out,
            n_elements=numel, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    
    # Step 2: Optimized chunking for in_2 - split tensor along dim=1
    def chunk_tensor_2d(tensor):
        # Efficient chunking that preserves memory layout
        # tensor shape is [B, C, H, W], chunk along C dimension (dim=1)
        half_channels = tensor.shape[1] // 2
        chunk1 = tensor[:, :half_channels, :, :].contiguous()
        chunk2 = tensor[:, half_channels:, :, :].contiguous()
        return chunk1.view(-1), chunk2.view(-1)  # Flatten for return consistency
    
    # Perform optimized computations
    temp_result = fused_add_relu(in_0, in_1, in_3)
    
    # Get all required chunks
    relu_chunk1, relu_chunk2 = chunk_tensor_2d(temp_result)
    chunk1_flat, chunk2_flat = chunk_tensor_2d(in_2)
    
    return (chunk1_flat, relu_chunk1, chunk2_flat, relu_chunk2)

def replacement_func():
    return optimized_full_computation