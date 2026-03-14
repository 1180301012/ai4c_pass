import torch
import triton
import triton.language as tl


# Pattern matching function - matches the computation pattern:
# tmp_0 = in_0
# tmp_1 = in_2 * in_1
# tmp_2 = tmp_1 + tmp_0
# tmp_3 = torch.unbind(tmp_2, dim=2)
# tmp_4 = tmp_3[0]
# tmp_5 = tmp_3[1]
# tmp_6 = tmp_5.permute(0, 2, 1)
# return (tmp_6, tmp_4)
def pattern(in_0, in_1, in_2):
    # First do the element-wise ops
    tmp_0 = in_0
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + tmp_0
    # Then unbind
    tmp_3 = torch.unbind(tmp_2, dim=2)
    # Get both slices
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    # Permute one and return both (comma-separated, not tuple)
    tmp_6 = tmp_5.permute(0, 2, 1)
    # Return order: tmp_6, tmp_4 (comma-separated)
    return tmp_6, tmp_4


# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Optimized Triton kernel that fuses multiply, add, unbind, and permute
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_0_ptr, out_1_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Grid: (B, W, H) - each block computes one element position
    # B = batch size (dim 0 of output) = 1
    # W = 128 (feature dim C)
    # H = 17 (feature dim H)
    
    # Get position
    batch_idx = tl.program_id(0)
    w_idx = tl.program_id(1)  # 0 to 127 (feature dim 128)
    h_idx = tl.program_id(2)  # 0 to 16 (feature dim 17)
    
    # Compute offsets
    # For in_0 [2, 128]: we need in_0[b, w_idx] where b = 0 or 1
    # in_0 is indexed by tmp_2[..., 0, :] and tmp_2[..., 1, :]
    # tmp_2[..., 0, :] = tmp_1[..., 0, :] + in_0[0, :]
    # tmp_2[..., 1, :] = tmp_1[..., 1, :] + in_0[1, :]
    
    # in_0 shape: [2, 128], strides: [128, 1]
    # in_0[b, w] -> b * 128 + w
    
    # in_1 shape: [1, 1, 2, 128], broadcasted from [1, 1, 2, 128]
    # in_1[0, 0, b, w] -> b * 128 + w (since leading dims are 1)
    
    # in_2 shape: [1, 17, 1, 128]
    # in_2[0, h, 0, w] -> h * 128 + w
    
    # Compute offset for in_0 slice 0: in_0[0, w_idx] = 0 * 128 + w_idx
    in_0_offset_slice0 = w_idx
    # Compute offset for in_0 slice 1: in_0[1, w_idx] = 1 * 128 + w_idx
    in_0_offset_slice1 = 128 + w_idx
    
    # Compute offset for in_1 slice 0: in_1[0, 0, 0, w_idx] = 0 * 128 + w_idx
    in_1_offset_slice0 = w_idx
    # Compute offset for in_1 slice 1: in_1[0, 0, 1, w_idx] = 1 * 128 + w_idx
    in_1_offset_slice1 = 128 + w_idx
    
    # Compute offset for in_2: in_2[0, h_idx, 0, w_idx] = h_idx * 128 + w_idx
    in_2_offset = h_idx * 128 + w_idx
    
    # Load values
    # For slice 0 (result_1 = tmp_2[..., 0, :]):
    # tmp_2[..., 0, :] = in_2 * in_1[..., 0, :] + in_0[0, :]
    in_0_val_0 = tl.load(in_0_ptr + in_0_offset_slice0)
    in_1_val_0 = tl.load(in_1_ptr + in_1_offset_slice0)
    in_2_val = tl.load(in_2_ptr + in_2_offset)
    
    # Compute result for slice 0: result_1
    # tmp_2[..., 0, :] = in_2 * in_1[..., 0, :] + in_0[0, :]
    result_1 = in_2_val * in_1_val_0 + in_0_val_0
    
    # For slice 1 (result_0 = tmp_2[..., 1, :].permute(0, 2, 1)):
    # tmp_2[..., 1, :] = in_2 * in_1[..., 1, :] + in_0[1, :]
    in_0_val_1 = tl.load(in_0_ptr + in_0_offset_slice1)
    in_1_val_1 = tl.load(in_1_ptr + in_1_offset_slice1)
    
    # Compute result for slice 1: result_0 (will be permuted)
    # tmp_2[..., 1, :] = in_2 * in_1[..., 1, :] + in_0[1, :]
    result_0 = in_2_val * in_1_val_1 + in_0_val_1
    
    # Output shapes:
    # out_0 (tmp_6): [B, C, H] = [1, 128, 17] -> indexed as [batch, w, h]
    # out_1 (tmp_4): [B, H, C] = [1, 17, 128] -> indexed as [batch, h, w]
    
    # For out_0 (result_0 permuted): indexed by [batch_idx, w_idx, h_idx]
    out_0_offset = batch_idx * 128 * 17 + w_idx * 17 + h_idx
    # For out_1 (result_1): indexed by [batch_idx, h_idx, w_idx]
    out_1_offset = batch_idx * 17 * 128 + h_idx * 128 + w_idx
    
    # Store results
    tl.store(out_0_ptr + out_0_offset, result_0)
    tl.store(out_1_ptr + out_1_offset, result_1)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    # Determine output shapes
    # in_2 shape: [B, H, 1, C] = [1, 17, 1, 128] -> B=1, H=17, C=128
    # After computation:
    # - result_0 (tmp_6): [B, C, H] = [1, 128, 17] (from permute)
    # - result_1 (tmp_4): [B, H, C] = [1, 17, 128] (from unbind)
    
    B = in_2.shape[0]  # batch size
    H = in_2.shape[1]  # 17
    W = in_2.shape[3]  # 128 (feature dim, which is C)
    
    # Create output tensors
    out_0 = torch.empty((B, W, H), dtype=torch.float32, device=in_0.device)  # [1, 128, 17]
    out_1 = torch.empty((B, H, W), dtype=torch.float32, device=in_0.device)  # [1, 17, 128]
    
    # Launch kernel with grid (B, W, H)
    # Each thread handles one output element
    grid = (B, W, H)
    
    fused_kernel[grid](
        in_0, in_1, in_2,
        out_0, out_1,
        B, H, W, 128,  # B, H=17, W=128, C=128
        BLOCK_SIZE=1
    )
    
    return out_0, out_1


def replacement_func():
    return fused_kernel_wrapper