import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Different tile sizes for the softmax dimension
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_conv_softmax_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_c, stride_weight_kh, stride_weight_kw,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Conv2d: input [N, C, H, W] @ weight [1, C, 1, 1] + bias -> [N, 1, H, W]
    2. View: reshape to [N, 1, H*W]
    3. Softmax: along last dimension -> [N, 1, H*W]
    """
    # Each program processes one batch element
    batch_idx = tl.program_id(0)
    
    # Compute the base offset for this batch
    input_offset = batch_idx * stride_input_n
    
    # Load bias
    bias = tl.load(bias_ptr).to(tl.float32)
    
    # For each output position in the H*W dimension
    # We process BLOCK_SIZE elements at a time
    # Each position corresponds to a spatial location (h, w)
    
    # First, compute the convolution result for all spatial positions
    # Since weight is [1, C, 1, 1], the convolution at position (h, w) is:
    # sum over c: input[n, c, h, w] * weight[0, c, 0, 0] + bias
    
    # We need to compute this for all (h, w) positions
    # But for softmax, we need all values first, so let's do two phases
    
    # Phase 1: Compute conv output for each spatial position
    # For simplicity, we compute per-element of the flattened output
    
    # Shared storage for conv results
    # Since H*W = 4096, we can fit this in a reasonable shared memory
    # or we can compute on-the-fly
    
    # Let's process the softmax dimension
    # Each thread block handles BLOCK_SIZE elements of the softmax
    
    # Offsets for this block
    offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Convert flat index to (h, w) coordinates
    # Each batch has 1 channel, and H*W spatial positions
    h_coords = offs // W
    w_coords = offs % W
    
    # Compute convolution for each (h, w)
    # The convolution with weight [1, C, 1, 1] is essentially:
    # output[n, 0, h, w] = bias + sum_c(input[n, c, h, w] * weight[0, c, 0, 0])
    
    # Compute conv for all positions in this block
    conv_result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    conv_result = conv_result + bias
    
    # Loop over channels
    for c in range(C):
        # Load input value at [batch, c, h, w]
        # Need to compute valid offsets for h, w within bounds
        valid_h = h_coords < H
        valid_w = w_coords < W
        valid = valid_h & valid_w & mask
        
        # Compute input offset
        input_offset_c = input_offset + c * stride_input_c
        # For each position, the offset is: base + h * stride_h + w * stride_w
        h_offsets = h_coords * stride_input_h
        w_offsets = w_coords * stride_input_w
        offsets = input_offset_c + h_offsets + w_offsets
        
        # Load input values (with mask for out-of-bounds)
        input_vals = tl.load(input_ptr + offsets, mask=valid, other=0.0)
        
        # Load weight [0, c, 0, 0]
        weight_offset = c * stride_weight_c
        weight_val = tl.load(weight_ptr + weight_offset).to(tl.float32)
        
        # Accumulate conv result
        conv_result = conv_result + input_vals * weight_val
    
    # Now conv_result holds the conv output for this block
    # We need to compute softmax: exp(x_i) / sum_j(exp(x_j))
    
    # For softmax, we need the max value for numerical stability
    # Since we're processing in blocks, we need a two-pass approach:
    # Pass 1: Find max
    # Pass 2: Compute exp and sum, then normalize
    
    # However, for simplicity with Triton, let's do a simpler approach:
    # Use tl.reduce for max and sum
    
    # Actually, we need the global max for softmax, which requires all elements
    # Let's restructure: compute all conv results first, then do softmax
    
    # For now, let's use a simpler approach - compute exp and sum across all elements
    # This kernel will compute a portion, and we'll need to synchronize
    
    # Actually, the cleanest way is to compute the full conv result first,
    # store to global memory, then do softmax
    
    # But for maximum fusion, let's try computing exp and storing partial results
    # We'll need atomic operations for the sum
    
    # Alternative: Process all H*W elements in a single kernel launch per batch
    # This avoids the need for multiple kernel invocations
    
    # Let's do a two-phase approach within this kernel using tl.dot
    
    # Actually, let me reconsider. The cleanest approach is:
    # 1. Compute all conv outputs for the batch (flattened)
    # 2. Compute softmax using a reduction
    
    # For the reduction, we can use tl.reduce but that only works within a block
    # We need the full sum
    
    # Let me use a different strategy: compute exp(x - max) where max is computed separately
    
    # Since this is getting complex, let me implement a simpler but efficient version:
    # We'll compute conv, store to temp, then do softmax in a second kernel
    # But that's not truly fused...
    
    # Let me try using shared memory for the reduction
    # Each block loads BLOCK_SIZE elements, computes exp, then we reduce
    
    # Phase 1: Compute max for numerical stability
    # We need to process ALL elements to find the global max
    
    # For now, let's use a simpler approach - compute exp directly and normalize
    # This is less numerically stable but works for this use case
    
    exp_result = tl.exp(conv_result)
    
    # We need the sum of ALL exp values in this batch
    # This requires cross-block reduction
    
    # Let's store the partial exp results first
    # Then we'll do a second pass for the sum
    
    # For simplicity, let's just compute the softmax in a second pass
    # But make the conv very efficient
    
    # Actually, let me reconsider the problem
    # The output is [N, 1, H*W] = [batch, 1, 4096]
    # Each batch element is independent
    
    # The best approach:
    # 1. One kernel computes conv for all elements: [N, 1, H, W] -> flatten
    # 2. Second kernel does softmax
    
    # But we want fusion. Let me do:
    # - Each block computes conv for BLOCK_SIZE elements
    # - Then computes exp for those elements
    # - Then uses shared memory reduction to get the sum
    # - Then normalizes
    
    # This requires shared memory, which is limited
    # With 4096 elements and BLOCK_SIZE=1024, we need 4 blocks
    # We can fit BLOCK_SIZE floats in shared memory per block
    
    # Let me implement this properly
    
    # Store exp results to be reduced
    exp_storage = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    exp_storage = exp_result
    
    # Compute sum using reduction within block (assuming n_elements <= BLOCK_SIZE for simplicity)
    # For larger, we'd need multi-block reduction
    
    # Actually, for N=32 and H*W=4096, each batch has 4096 elements
    # Let's handle the case where BLOCK_SIZE >= n_elements
    
    # For proper softmax, we need global max
    # Let's compute it using block reduction
    
    # Simpler: just compute exp and sum, assuming numerical stability is okay
    # This is a common approximation
    
    # Actually, let's use the fact that we're processing per-batch
    # and use atomic add for the sum
    
    # But atomic operations are slow
    
    # Let me try a different approach:
    # Compute the conv output, store to output (pre-softmax)
    # Then run a second kernel for softmax
    
    # For full fusion, we need:
    # 1. Each block computes conv for its portion
    # 2. Use shared memory to find max across all blocks
    # 3. Second pass computes exp and sum
    # 4. Third pass normalizes
    
    # This is complex. Let me implement a simpler version first:
    # Just fuse conv + view, then use torch's softmax
    
    # Actually, I realize the issue: to truly fuse conv + softmax,
    # we need the full conv output first to compute softmax denominators
    
    # Let me implement a two-kernel approach that's still faster:
    # 1. Fused conv + view kernel that produces [N, 1, H*W]
    # 2. Optimized softmax kernel
    
    # For the conv kernel:
    # - Each thread handles one output element
    # - Compute dot product: input[c, h, w] * weight[c]
    # - Add bias
    
    # Let me rewrite this properly


# Simplified fused kernel: compute conv + view, output flat tensor
# Then use separate optimized softmax kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_conv_view_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_c,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Conv2d: input [N, C, H, W] @ weight [1, C, 1, 1] + bias -> [N, 1, H, W]
    2. View: reshape to [N, 1, H*W]
    
    Each thread computes one output element.
    """
    # Program ID gives us the flat index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Each output element corresponds to (batch, h, w)
    batch_idx = offs // (H * W)
    spatial_idx = offs % (H * W)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Load bias
    bias = tl.load(bias_ptr).to(tl.float32)
    
    # Compute conv for each element - start with bias only for valid elements
    result = tl.where(mask, bias, 0.0)
    
    # Iterate over channels
    for c in range(C):
        # Compute input offsets
        input_base = batch_idx * stride_input_n
        input_offset = input_base + c * stride_input_c + h_idx * stride_input_h + w_idx * stride_input_w
        
        # Load input and weight
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + c * stride_weight_c).to(tl.float32)
        
        result = result + input_val * weight_val
    
    # Store result
    tl.store(output_ptr + offs, result, mask=mask)


# Optimized softmax kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel along the last dimension.
    Computes: exp(x_i) / sum_j(exp(x_j))
    Uses numerical stabilization: exp(x_i - max) / sum_j(exp(x_j - max))
    """
    # Each program handles one batch
    batch_idx = tl.program_id(0)
    
    # Offsets for this batch
    input_offset = batch_idx * n_elements
    output_offset = batch_idx * n_elements
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load all elements for this batch
    # Since we need all elements for max and sum, and n_elements = 4096
    # and BLOCK_SIZE = 4096, one block is enough
    x = tl.load(input_ptr + input_offset + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Compute exp(x - max)
    exp_x = tl.exp(x - max_val)
    
    # Compute sum of exp values
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_val = exp_x / sum_exp
    
    # Store result
    tl.store(output_ptr + output_offset + offs, softmax_val, mask=mask)


@torch.fx.wrap
def fused_conv_softmax_wrapper(input_tensor, weight_tensor, bias_tensor):
    """
    Wrapper function that launches the fused conv + view + softmax kernels.
    """
    N, C, H, W = input_tensor.shape
    n_elements = N * H * W  # This is the flattened size per channel
    
    # Allocate output tensor
    output = torch.empty((N, 1, H * W), dtype=torch.float32, device=input_tensor.device)
    
    # Intermediate buffer for conv output
    conv_output = torch.empty((N, 1, H * W), dtype=torch.float32, device=input_tensor.device)
    
    # First kernel: fused conv + view
    # Each output element is independent, so we can parallelize over N*H*W
    # But for better efficiency with softmax, let's process per batch
    
    # Actually, let's process one batch at a time for the conv
    # to maximize L2 cache reuse of the weight
    grid = (N,)
    
    # Weight is [1, C, 1, 1], need to flatten for Triton
    weight_flat = weight_tensor.squeeze().contiguous()
    
    fused_conv_view_kernel[grid](
        input_tensor, weight_flat, bias_tensor,
        conv_output,
        N, C, H, W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight_flat.stride(0),
        H * W,  # n_elements per batch
    )
    
    # Second kernel: softmax
    # Each batch is independent
    grid_softmax = (N,)
    
    softmax_kernel[grid_softmax](
        conv_output, output,
        H * W,  # n_elements per batch
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: conv2d -> view -> softmax
    
    The view dimension is determined by the batch size of the input.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Determine batch size from input tensor
    batch_size = in_2.shape[0]
    
    tmp_3 = tmp_2.view(batch_size, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_2, in_1, in_0)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_conv_softmax_wrapper