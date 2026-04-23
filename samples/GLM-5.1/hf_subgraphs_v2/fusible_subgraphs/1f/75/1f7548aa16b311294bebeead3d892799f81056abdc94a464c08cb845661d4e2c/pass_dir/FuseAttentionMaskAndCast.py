import torch
import triton
import triton.language as tl
from torch import device

# Pattern for seq_len=2 (float16/9 and float32/9)
def pattern_2(in_0, in_1, in_2, in_3):
    tmp_2 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool)
    tmp_3 = torch.arange(2, device = device(type='cuda', index=0))
    tmp_3 += 0;  tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_6 = torch.arange(2, device = device(type='cuda', index=0))
    tmp_6 += 0;  tmp_7 = tmp_6
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device = device(type='cuda', index=0))
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return (tmp_13, tmp_21, tmp_22)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "route_2")


# Kernel for computing the causal + padding attention mask
@triton.jit
def attention_mask_kernel(
    mask_ptr,       # in_0: int64 [batch, seq_len]
    cache_pos_ptr,  # in_2: int64 [seq_len]
    out_ptr,        # output: bool [batch, 1, seq_len, seq_len]
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    b = pid_row // seq_len
    j = pid_row % seq_len
    col_start = pid_col * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_valid = col_offsets < seq_len
    cache_pos_j = tl.load(cache_pos_ptr + j)
    attn_mask_vals = tl.load(mask_ptr + b * seq_len + col_offsets, mask=col_valid, other=0)
    padding = attn_mask_vals != 0
    causal = col_offsets <= cache_pos_j
    result = causal & padding
    out_offsets = b * seq_len * seq_len + j * seq_len + col_offsets
    tl.store(out_ptr + out_offsets, result, mask=col_valid)


@triton.jit
def inv_freq_cast_kernel(
    inv_freq_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    inv_freq = tl.load(inv_freq_ptr + offsets, mask=mask, other=0.0)
    inv_freq_f = inv_freq.to(tl.float32)
    tl.store(out_ptr + offsets, inv_freq_f, mask=mask)


@triton.jit
def position_ids_cast_kernel(
    pos_ids_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    pos_ids = tl.load(pos_ids_ptr + offsets, mask=mask, other=0)
    pos_ids_f = pos_ids.to(tl.float32)
    tl.store(out_ptr + offsets, pos_ids_f, mask=mask)


@torch.fx.wrap
def _fused_attention_mask_and_cast_2(in_0, in_1, in_2, in_3):
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    
    out_mask = torch.empty((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device='cuda:0')
    BLOCK_SIZE = 256
    num_rows = batch_size * seq_len
    num_cols = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    attention_mask_kernel[(num_rows, num_cols)](
        mask_ptr=in_0, cache_pos_ptr=in_2, out_ptr=out_mask,
        batch_size=batch_size, seq_len=seq_len, BLOCK_SIZE=BLOCK_SIZE,
    )
    
    n_inv = in_1.shape[0]
    out_inv = torch.empty((1, n_inv, 1), dtype=torch.float32, device='cuda:0')
    BLOCK_SIZE_INV = 64
    num_inv_programs = (n_inv + BLOCK_SIZE_INV - 1) // BLOCK_SIZE_INV
    inv_freq_cast_kernel[(num_inv_programs,)](
        inv_freq_ptr=in_1, out_ptr=out_inv, n_elements=n_inv, BLOCK_SIZE=BLOCK_SIZE_INV,
    )
    
    out_pos_shape = list(in_3.shape)
    out_pos_shape.insert(1, 1)
    out_pos = torch.empty(out_pos_shape, dtype=torch.float32, device='cuda:0')
    n_pos = in_3.numel()
    BLOCK_SIZE_POS = 256
    num_pos_programs = (n_pos + BLOCK_SIZE_POS - 1) // BLOCK_SIZE_POS
    position_ids_cast_kernel[(num_pos_programs,)](
        pos_ids_ptr=in_3, out_ptr=out_pos, n_elements=n_pos, BLOCK_SIZE=BLOCK_SIZE_POS,
    )
    
    return (out_mask, out_inv, out_pos)


@torch.fx.wrap
def _fused_attention_mask_and_cast_3(in_0, in_1, in_2, in_3):
    return _fused_attention_mask_and_cast_2(in_0, in_1, in_2, in_3)

@torch.fx.wrap
def _fused_attention_mask_and_cast_64(in_0, in_1, in_2, in_3):
    return _fused_attention_mask_and_cast_2(in_0, in_1, in_2, in_3)

@torch.fx.wrap
def _fused_attention_mask_and_cast_128(in_0, in_1, in_2, in_3):
    return _fused_attention_mask_and_cast_2(in_0, in_1, in_2, in_3)

@torch.fx.wrap
def _fused_attention_mask_and_cast_256(in_0, in_1, in_2, in_3):
    return _fused_attention_mask_and_cast_2(in_0, in_1, in_2, in_3)

@torch.fx.wrap
def _fused_attention_mask_and_cast_512(in_0, in_1, in_2, in_3):
    return _fused_attention_mask_and_cast_2(in_0, in_1, in_2, in_3)


@torch.fx.wrap
def dispatch_wrapper(in_0, in_1, in_2, in_3, route):
    if route == "route_2":
        return _fused_attention_mask_and_cast_2(in_0, in_1, in_2, in_3)
    elif route == "route_3":
        return _fused_attention_mask_and_cast_3(in_0, in_1, in_2, in_3)
    elif route == "route_64":
        return _fused_attention_mask_and_cast_64(in_0, in_1, in_2, in_3)
    elif route == "route_128":
        return _fused_attention_mask_and_cast_128(in_0, in_1, in_2, in_3)
    elif route == "route_256":
        return _fused_attention_mask_and_cast_256(in_0, in_1, in_2, in_3)
    elif route == "route_512":
        return _fused_attention_mask_and_cast_512(in_0, in_1, in_2, in_3)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_wrapper