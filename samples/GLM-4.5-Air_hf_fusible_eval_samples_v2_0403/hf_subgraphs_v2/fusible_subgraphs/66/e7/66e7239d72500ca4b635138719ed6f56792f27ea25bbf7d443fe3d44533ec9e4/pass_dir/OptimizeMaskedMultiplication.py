import torch
import triton
import triton.language as tl

@triton.jit
def masked_mult_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs with proper type handling
    # lhs (LayerNorm output) is float16, rhs (mask) is float32
    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0)
    
    # Multiplication
    out = lhs * rhs
    
    # Convert back to float16 and store
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)

@torch.fx.wrap
def optimized_masked_multiplication(layer_norm_out, float_mask):
    # layer_norm_out: [1, 16, 768] float16
    # float_mask: [1, 16, 768] float32
    n_elements = layer_norm_out.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(layer_norm_out)  # float16 output
    
    masked_mult_kernel[(num_programs,)](
        lhs_ptr=layer_norm_out,
        rhs_ptr=float_mask,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(in_0, in_1, in_2, in_3):
    # Match the multiplication pattern: tmp_8 = tmp_4 * tmp_7
    # We need to match what tmp_4 and tmp_7 represent:
    # tmp_4 is the layer_norm output (we need to recreate it here)
    # tmp_7 is the expand+type conversion from in_0
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    mask_expanded = in_0.unsqueeze(-1).expand_as(tmp_4).float()
    tmp_8 = tmp_4 * mask_expanded
    return tmp_8

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    # This returns a function that does LayerNorm then multiplications
    # We'll handle this more comprehensively in a separate fusion pass
    import torch.nn.functional as F
    
    def fused_forward(mask, bias, weight, input_tensor):
        # Do LayerNorm first
        tmp_4 = F.layer_norm(input_tensor, (768,), weight, bias, 1e-12)
        # Create expanded mask and multiply
        mask_expanded = mask.unsqueeze(-1).expand_as(tmp_4).float()
        tmp_8 = tmp_4 * mask_expanded
        return tmp_8, mask_expanded, tmp_4
    
    return fused_forward