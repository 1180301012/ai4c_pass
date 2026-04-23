import torch
import triton
import triton.language as tl

@triton.jit
def ccnet_attention_kernel(
    query_ptr, key_ptr, energy_ptr, 
    output_ptr, output_slice_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    stride_qb, stride_qc, stride_qh, stride_qw,
    stride_kb, stride_kc, stride_kh, stride_kw,
    stride_eb, stride_eh, stride_ew1, stride_ew2,
    BLOCK_SIZE: tl.constexpr
):
    # Get position
    pid = tl.program_id(0)
    
    # Calculate which batch and spatial position we handle
    b = pid // (H * W)
    h = (pid % (H * W)) // W
    w = pid % W
    
    # Compute einsum (query * key) for position (b, h, w, :)
    # einsum[b,h,w,j] = sum_c query[b,c,h,w] * key[b,c,h,j]
    # We compute the dot products for all j in this kernel call
    
    # Load query value at (b, :, h, w) - all channels
    q_offsets = stride_qb * b + stride_qh * h + stride_qw * w + tl.arange(0, C) * stride_qc
    q_vals = tl.load(query_ptr + q_offsets)
    
    # Compute dot products with key for all output positions j
    # For each j in [0, W), compute sum_c query[c] * key[b,c,h,j]
    offs_j = tl.arange(0, BLOCK_SIZE)
    
    # For simplicity, process W elements per j
    # Load key values for all j positions (W values per channel summed)
    # Key shape after einsum: [B, H, W, W]
    # We need key[b, c, h, j] for all c and j
    
    # Initialize output buffer for this (b, h, w) position
    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Unroll over channels
    for c in range(C):
        # Load query value at channel c
        q_offset = stride_qb * b + stride_qc * c + stride_qh * h + stride_qw * w
        q_val = tl.load(query_ptr + q_offset)
        
        # Load key values for channel c: key[b, c, h, j] for all j
        k_offset = stride_kb * b + stride_kc * c + stride_kh * h + offs_j * stride_kw
        k_vals = tl.load(key_ptr + k_offset)
        
        # Accumulate
        accum = accum + q_val * k_vals
    
    # Now accum holds einsum[b, h, w, j] for j in [0, W)
    # Convert to the dtype of output
    einsum_vals = accum.to(tl.load(query_ptr).dtype)
    
    # Now do cat + softmax
    # energy[b, h, w, :] has shape [W], concatenated with einsum
    # For softmax we need: exp(val - max) / sum(exp(val - max))
    
    # Load energy values: energy[b, h, w, :] - shape [W]
    e_offset_base = stride_eb * b + stride_eh * h + stride_ew1 * w
    energy_vals = tl.load(energy_ptr + e_offset_base + offs_j * stride_ew2)
    
    # Combined values: [energy, einsum]
    # First W elements are energy, next W elements are einsum
    # But softmax is on the last dimension which now has size 2*W
    # output position k: if k < W use energy, else use einsum[k-W]
    
    # For efficiency, we only need the first W (64) elements of softmax output
    # because we slice [..64] afterwards
    # Softmax on [energy(0:W), einsum(W:2W)]
    # We want softmax_output[0:W]
    
    # Compute softmax: the max is max(energy, einsum)
    # For softmax_output[i] where i < W:
    # softmax_output[i] = exp(energy[i] - max_val) / sum_all(exp(val - max_val))
    
    # First, find max across all 2W values
    # We need to compute max(energy_vals, einsum_vals)
    # Energy is at positions 0 to W-1, einsum at W to 2W-1
    
    # Load energy values for comparison
    energy_for_max = energy_vals
    einsum_for_max = einsum_vals
    
    max_energy = tl.max(energy_for_max)
    max_einsum = tl.max(einsum_for_max)
    max_val = tl.max(tl.cat([max_energy, max_einsum], axis=0))
    
    # Compute exp(val - max) for both
    exp_energy = tl.exp(energy_vals - max_val)
    exp_einsum = tl.exp(einsum_vals - max_val)
    
    # Sum of all exponentials
    sum_exp = tl.sum(exp_energy) + tl.sum(exp_einsum)
    
    # Final softmax values for positions 0 to W-1
    softmax_output = exp_energy / sum_exp
    
    # Store full softmax output for tmp_3
    # Output shape is [B, H, W, 2*W], but we only compute for current (b,h,w)
    # Store to output[b, h, w, :] where : is 2*W elements
    # We need energy at positions 0 to W-1 and einsum at positions W to 2W-1
    out_base = b * (H * W * 2 * W) + h * (W * 2 * W) + w * (2 * W)
    
    # Store energy softmax part (positions 0 to W-1)
    tl.store(output_ptr + out_base + offs_j * (2 * W), softmax_output)
    
    # Store einsum softmax part (positions W to 2W-1)
    exp_einsum_normalized = exp_einsum / sum_exp
    tl.store(output_ptr + out_base + W * (2 * W) + offs_j * (2 * W), exp_einsum_normalized)
    
    # Store sliced output (first W=64 elements = softmax_output)
    tl.store(output_slice_ptr + out_base + offs_j * W, softmax_output)


@torch.fx.wrap
def ccnet_attention_fused(query, key, energy):
    """
    Fused CCNet attention: einsum + cat + softmax + slice
    query: [B, C, H, W]
    key: [B, C, H, W]
    energy: [B, H, W, W]
    Returns: (softmax_result [B, H, W, 2*W], sliced_result [B, H, W, W])
    """
    B, C, H, W = query.shape
    
    # Allocate outputs
    output_full = torch.empty((B, H, W, 2 * W), dtype=query.dtype, device=query.device)
    output_slice = torch.empty((B, H, W, W), dtype=query.dtype, device=query.device)
    
    # Grid: one program per (b, h, w) position
    grid = (B * H * W,)
    
    ccnet_attention_kernel[grid](
        query, key, energy,
        output_full, output_slice,
        B, H, W, C,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        energy.stride(0), energy.stride(1), energy.stride(2), energy.stride(3),
        W,  # BLOCK_SIZE
    )
    
    return output_full, output_slice


def pattern(in_0, in_1, in_2):
    """
    Match the CCNet attention pattern:
    einsum('bchw,bchj->bhwj', in_2, in_1) + cat + softmax + slice
    """
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return ccnet_attention_fused