import torch
import triton
import triton.language as tl

# The pattern to match:
# tmp_1 = in_1.to(dtype=torch.float32)
# tmp_2 = 1.0 - tmp_1
# tmp_3 = tmp_2 * -3.4028234663852886e+38
# And optionally slicing:
# tmp_4 = tmp_0[:, :N]

# This fuses: to(float32) + (1.0 - x) + (x * -inf) into a single kernel

NEG_INF = -3.4028234663852886e+38  # float32 -infinity

@triton.jit
def fused_attention_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load int64 values
    x = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    
    # Compute: (1.0 - x) * -inf
    # When x=1: (1-1) * -inf = 0 * -inf = 0 (or -inf depending on floating point)
    # When x=0: (1-0) * -inf = 1 * -inf = -inf
    # Actually we need to be careful here:
    # 1.0 - x: when x is int64, we need to convert first
    # Then multiply by -inf
    
    # Equivalent to: (1.0 - x) * -inf, but more numerically stable
    # For attention mask: 0 -> -inf, 1 -> 0
    result = (1.0 - x) * NEG_INF
    
    # Store float32 results
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_attention_mask(in_1):
    """
    Fused attention mask computation.
    Original: to(float32) -> 1.0 - x -> x * -inf
    This is used in transformers to create attention masks where
    0 becomes -infinity (masked) and 1 becomes 0 (attended).
    """
    # Handle different input shapes - we need to flatten and process
    original_shape = in_1.shape
    n_elements = in_1.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Flatten input for processing
    in_flat = in_1.view(-1)
    out = torch.empty_like(in_flat, dtype=torch.float32)
    
    fused_attention_mask_kernel[(num_programs,)](
        in_ptr=in_flat,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to original shape
    return out.view(original_shape)


def pattern(in_0, in_1):
    """
    Match the attention mask computation pattern:
    tmp_1 = in_1.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    
    We need to include the slice operation to match the model's full output.
    Since slice indices vary, we use __getitem__ which represents slicing.
    """
    # This is the attention mask computation to match
    tmp_1 = in_1.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    
    # The slicing - we need to match the slice operation
    # Using __getitem__ to represent the slice
    tmp_4 = in_0[:, :]

    return tmp_3, tmp_4


def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function."""
    return (in_1,)


def replacement_func():
    """Return the fused attention mask function."""
    return fused_attention_mask