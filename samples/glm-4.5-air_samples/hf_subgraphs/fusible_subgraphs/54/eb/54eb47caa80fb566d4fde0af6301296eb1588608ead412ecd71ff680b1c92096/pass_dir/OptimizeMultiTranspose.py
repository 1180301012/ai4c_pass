import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    # Pattern: three consecutive transpose(0, 1) operations
    # tmp_3 = tmp_1.transpose(0, 1) 
    # tmp_4 = tmp_2.transpose(0, 1)
    # tmp_5 = in_0.transpose(0, 1)
    # return (tmp_4, tmp_3, tmp_5)
    out1 = b.transpose(0, 1)
    out2 = a.transpose(0, 1) 
    out3 = c.transpose(0, 1)
    return out2, out1, out3

def replacement_args(a, b, c):
    return (a, b, c)

@triton.jit
def multi_transpose_kernel(
    a_ptr,
    b_ptr, 
    c_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    a_dims0, a_dims1, a_dims2,
    b_dims0, b_dims1, b_dims2,
    c_dims0, c_dims1, c_dims2,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Compute total elements for each tensor and determine masks
    total_a = a_dims0 * a_dims1 * a_dims2
    total_b = b_dims0 * b_dims1 * b_dims2
    total_c = c_dims0 * c_dims1 * c_dims2
    
    mask_a = offsets < total_a
    mask_b = offsets < total_b
    mask_c = offsets < total_c
    
    # Load all input tensors
    a = tl.load(a_ptr + offsets, mask=mask_a, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask_b, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask_c, other=0.0)
    
    # For transpose(0, 1) operation: [d0, d1, d2] -> [d1, d0, d2]
    # We need to map the original index to the transposed index
    
    # Compute dimension indices for input tensors
    # For each tensor with shape [d0, d1, d2]:
    # offset = o0*(d1*d2) + o1*d2 + o2
    # After transpose(0,1): becomes [o1, o0, o2]
    # New offset = o1*(d0*d2) + o0*d2 + o2
    
    # Extract indices for tensor a (tmp_2 -> transposed_in_tmp4)
    offset_2_a = offsets % a_dims2
    offset_1_a = (offsets // a_dims2) % a_dims1
    offset_0_a = offsets // (a_dims1 * a_dims2)
    
    # Extract indices for tensor b (tmp_1 -> transposed_in_tmp3)  
    offset_2_b = offsets % b_dims2
    offset_1_b = (offsets // b_dims2) % b_dims1
    offset_0_b = offsets // (b_dims1 * b_dims2)
    
    # Extract indices for tensor c (in_0 -> transposed_in_tmp5)
    offset_2_c = offsets % c_dims2
    offset_1_c = (offsets // c_dims2) % c_dims1
    offset_0_c = offsets // (c_dims1 * c_dims2)
    
    # Compute transposed indices: [d0, d1, d2] -> [d1, d0, d2]
    # New index = o1*(d0*d2) + o0*d2 + o2
    
    # For output from a (originally tmp_2, becomes tmp_4)
    transposed_offset_a = (offset_1_a * a_dims0 + offset_0_a) * a_dims2 + offset_2_a
    transposed_mask_a = (offset_1_a < a_dims0) & (offset_0_a < a_dims1) & (offset_2_a < a_dims2)
    
    # For output from b (originally tmp_1, becomes tmp_3)
    transposed_offset_b = (offset_1_b * b_dims0 + offset_0_b) * b_dims2 + offset_2_b
    transposed_mask_b = (offset_1_b < b_dims0) & (offset_0_b < b_dims1) & (offset_2_b < b_dims2)
    
    # For output from c (originally in_0, becomes tmp_5)
    transposed_offset_c = (offset_1_c * c_dims0 + offset_0_c) * c_dims2 + offset_2_c
    transposed_mask_c = (offset_1_c < c_dims0) & (offset_0_c < c_dims1) & (offset_2_c < c_dims2)
    
    # Store transposed results in the correct output order
    # Pattern returns (out2, out1, out3) which corresponds to:
    # (transposed_a, transposed_b, transposed_c)
    tl.store(out1_ptr + transposed_offset_a, a, mask=transposed_mask_a)
    tl.store(out2_ptr + transposed_offset_b, b, mask=transposed_mask_b)
    tl.store(out3_ptr + transposed_offset_c, c, mask=transposed_mask_c)

@torch.fx.wrap
def optimized_multi_transpose(a, b, c):
    # Get input shapes
    a_shape = a.shape
    b_shape = b.shape
    c_shape = c.shape
    
    a_dims = a_shape
    b_dims = b_shape
    c_dims = c_shape
    
    # Create output tensors with transposed dimensions
    out1_shape = (a_dims[1], a_dims[0], a_dims[2])  # transpose dims 0,1
    out2_shape = (b_dims[1], b_dims[0], b_dims[2])  # transpose dims 0,1
    out3_shape = (c_dims[1], c_dims[0], c_dims[2])  # transpose dims 0,1
    
    out1 = torch.empty(out1_shape, dtype=a.dtype, device=a.device)
    out2 = torch.empty(out2_shape, dtype=b.dtype, device=b.device)
    out3 = torch.empty(out3_shape, dtype=c.dtype, device=c.device)
    
    # Get total elements and launch kernel
    total_elements = max(a.numel(), b.numel(), c.numel())
    
    # Block size for Triton
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    multi_transpose_kernel[(num_programs,)](
        a,
        b,
        c,
        out1,
        out2,
        out3,
        a_dims[0], a_dims[1], a_dims[2],
        b_dims[0], b_dims[1], b_dims[2],
        c_dims[0], c_dims[1], c_dims[2],
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out2, out1, out3

def replacement_func():
    return optimized_multi_transpose