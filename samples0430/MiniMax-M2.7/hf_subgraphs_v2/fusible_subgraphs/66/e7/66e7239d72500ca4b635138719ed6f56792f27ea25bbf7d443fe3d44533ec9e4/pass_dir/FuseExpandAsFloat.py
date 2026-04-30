import torch
import triton
import triton.language as tl

@triton.jit
def expand_as_float_kernel(
    mask_ptr,
    out_ptr,
    n_tokens,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (n_tokens, n_features / BLOCK_SIZE)
    token_idx = tl.program_id(0)
    feat_block_idx = tl.program_id(1)
    feat_offset = feat_block_idx * BLOCK_SIZE
    feat_offsets = feat_offset + tl.arange(0, BLOCK_SIZE)
    feat_mask = feat_offsets < n_features
    
    # Load mask value for this token (scalar load, broadcast across features)
    mask_val = tl.load(mask_ptr + token_idx).to(tl.float32)
    
    # Create broadcasted value for all features in this block
    mask_vals = tl.broadcast_to(mask_val, feat_offsets.shape)
    
    # Store broadcasted mask value
    out_offset = token_idx * n_features + feat_offsets
    tl.store(out_ptr + out_offset, mask_vals, mask=feat_mask)


def expand_as_float(mask_tensor, target_tensor):
    out = torch.empty(target_tensor.shape, dtype=torch.float32, device=mask_tensor.device)
    n_tokens = target_tensor.shape[1]
    n_features = target_tensor.shape[2]
    
    # Use 2D grid: tokens x (features / 64)
    BLOCK_SIZE = 64
    num_feat_blocks = triton.cdiv(n_features, BLOCK_SIZE)
    grid = (n_tokens, num_feat_blocks)
    
    expand_as_float_kernel[grid](
        mask_tensor,
        out,
        n_tokens,
        n_features,
        BLOCK_SIZE,
    )
    return out


@torch.fx.wrap
def expand_as_float_wrapper(mask_tensor, target_tensor):
    return expand_as_float(mask_tensor, target_tensor)


# Pattern matching function
def pattern(in_0, in_4):
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(in_4)
    tmp_7 = tmp_6.float()
    return tmp_7


# Argument extraction function
def replacement_args(in_0, in_4):
    return (in_0, in_4)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return expand_as_float_wrapper