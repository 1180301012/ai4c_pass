import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Match concatenation of 4 tensors along dimension 1
    This can be optimized with a direct memory layout operation
    """
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    return tmp_0

def replacement_args(in_0, in_1, in_2, in_3):
    """Return the input tensors that will be passed to the replacement"""
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_concat_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    n_elements_per_tensor: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    N: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    """Optimized concatenation kernel that directly copies tensors to output"""
    batch_idx = tl.program_id(0)
    
    # Calculate base offsets for each tensor
    base0 = batch_idx * (C0 * H * W)
    base1 = batch_idx * (C1 * H * W)
    base2 = batch_idx * (C2 * H * W)
    base3 = batch_idx * (C3 * H * W)
    
    # Process each element in the tensors
    for i in range(tl.program_id(1), N * H * W, tl.num_programs(1)):
        # Concatenation order: in0, in1, in2, in3
        if i < C0 * H * W:
            # Copy from in0
            offset = i
            tl.store(out_ptr + offset, tl.load(in0_ptr + base0 + offset))
        elif i < (C0 + C1) * H * W:
            # Copy from in1
            offset = i - C0 * H * W
            tl.store(out_ptr + i, tl.load(in1_ptr + base1 + offset))
        elif i < (C0 + C1 + C2) * H * W:
            # Copy from in2
            offset = i - (C0 + C1) * H * W
            tl.store(out_ptr + i, tl.load(in2_ptr + base2 + offset))
        else:
            # Copy from in3
            offset = i - (C0 + C1 + C2) * H * W
            tl.store(out_ptr + i, tl.load(in3_ptr + base3 + offset))

@torch.fx.wrap
def optimized_concat(in_0, in_1, in_2, in_3):
    """Optimized concatenation using direct memory copying"""
    N0, C0, H0, W0 = in_0.shape
    N1, C1, H1, W1 = in_1.shape
    N2, C2, H2, W2 = in_2.shape
    N3, C3, H3, W3 = in_3.shape
    
    # Validate shapes (should have same N, H, W)
    assert N0 == N1 == N2 == N3, f"Batch size mismatch: {N0}, {N1}, {N2}, {N3}"
    assert H0 == H1 == H2 == H3, f"Height mismatch: {H0}, {H1}, {H2}, {H3}"
    assert W0 == W1 == W2 == W3, f"Width mismatch: {W0}, {W1}, {W2}, {W3}"
    
    # Output shape
    C_total = C0 + C1 + C2 + C3
    N = N0  # Use first tensor's batch size
    H = H0  # Use first tensor's height
    W = W0  # Use first tensor's width
    
    # Create output tensor
    out = torch.empty(N, C_total, H, W, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate total spatial elements per tensor
    spatial_elements_per_tensor = H * W
    elements_per_channel = spatial_elements_per_tensor
    
    # Launch kernel
    total_elements = C_total * spatial_elements_per_tensor
    grid_size = (N, (total_elements + 255) // 256)  # Use N programs for batch processing
    
    optimized_concat_kernel[grid_size](
        in0_ptr=in_0, in1_ptr=in_1, in2_ptr=in_2, in3_ptr=in_3,
        out_ptr=out,
        n_elements_per_tensor=spatial_elements_per_tensor,
        C0=C0, C1=C1, C2=C2, C3=C3,
        N=N, H=H, W=W,
    )
    
    return out

def replacement_func():
    """Replacement function that returns optimized concatenation"""
    return optimized_concat