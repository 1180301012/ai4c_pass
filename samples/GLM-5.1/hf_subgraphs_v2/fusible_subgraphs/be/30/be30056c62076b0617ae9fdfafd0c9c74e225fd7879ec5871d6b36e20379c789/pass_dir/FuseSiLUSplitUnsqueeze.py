import torch
import triton
import triton.language as tl
import operator


def pattern(in_0, in_1):
    tmp_1 = torch.ops.aten.silu.default(in_1)
    split = torch.ops.aten.split.Tensor(tmp_1, [512, 512, 128], 2)
    tmp_3 = operator.getitem(split, 0)
    tmp_4 = operator.getitem(split, 1)
    tmp_5 = operator.getitem(split, 2)
    tmp_6 = torch.ops.aten.unsqueeze.default(tmp_5, 2)
    tmp_7_inter = torch.ops.aten.unsqueeze.default(in_0, 0)
    tmp_7 = torch.ops.aten.unsqueeze.default(tmp_7_inter, 0)
    return (tmp_7, tmp_3, tmp_6, tmp_4)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_silu_split_kernel(
    input_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    C: tl.constexpr,
    SPLIT0: tl.constexpr,
    SPLIT1: tl.constexpr,
    SPLIT2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input element
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute SiLU: x * sigmoid(x)
    silu_val = x * tl.sigmoid(x)

    # Determine column index within the last dimension
    c_idx = offsets % C
    # Row index (flattened B*S)
    bs_idx = offsets // C

    # Masks for each split bucket
    mask0 = (c_idx < SPLIT0) & mask
    mask1 = (c_idx >= SPLIT0) & (c_idx < SPLIT0 + SPLIT1) & mask
    mask2 = (c_idx >= SPLIT0 + SPLIT1) & mask

    # Compute output offsets for each bucket
    out0_off = bs_idx * SPLIT0 + c_idx
    out1_off = bs_idx * SPLIT1 + (c_idx - SPLIT0)
    out2_off = bs_idx * SPLIT2 + (c_idx - SPLIT0 - SPLIT1)

    # Store to appropriate output buffer
    tl.store(out0_ptr + out0_off, silu_val, mask=mask0)
    tl.store(out1_ptr + out1_off, silu_val, mask=mask1)
    tl.store(out2_ptr + out2_off, silu_val, mask=mask2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_silu_split_kernel_tuned(
    input_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    n_elements,
    C: tl.constexpr,
    SPLIT0: tl.constexpr,
    SPLIT1: tl.constexpr,
    SPLIT2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    silu_val = x * tl.sigmoid(x)

    c_idx = offsets % C
    bs_idx = offsets // C

    mask0 = (c_idx < SPLIT0) & mask
    mask1 = (c_idx >= SPLIT0) & (c_idx < SPLIT0 + SPLIT1) & mask
    mask2 = (c_idx >= SPLIT0 + SPLIT1) & mask

    out0_off = bs_idx * SPLIT0 + c_idx
    out1_off = bs_idx * SPLIT1 + (c_idx - SPLIT0)
    out2_off = bs_idx * SPLIT2 + (c_idx - SPLIT0 - SPLIT1)

    tl.store(out0_ptr + out0_off, silu_val, mask=mask0)
    tl.store(out1_ptr + out1_off, silu_val, mask=mask1)
    tl.store(out2_ptr + out2_off, silu_val, mask=mask2)


@torch.fx.wrap
def fused_silu_split_unsqueeze(in_0, in_1):
    B, S, C = in_1.shape
    n_elements = in_1.numel()

    SPLIT0, SPLIT1, SPLIT2 = 512, 512, 128

    out0 = torch.empty(B, S, SPLIT0, dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty(B, S, SPLIT1, dtype=in_1.dtype, device=in_1.device)
    out2_data = torch.empty(B, S, SPLIT2, dtype=in_1.dtype, device=in_1.device)

    # Use the tuned kernel
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_silu_split_kernel_tuned[grid](
        input_ptr=in_1,
        out0_ptr=out0,
        out1_ptr=out1,
        out2_ptr=out2_data,
        n_elements=n_elements,
        C=C,
        SPLIT0=SPLIT0,
        SPLIT1=SPLIT1,
        SPLIT2=SPLIT2,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Unsqueeze the third split result
    out2 = out2_data.unsqueeze(2)  # [B, S, 1, 128]

    # View reshape for in_0 (metadata-only operation)
    out7 = in_0[None, None, :]

    return (out7, out0, out2, out1)


def replacement_func():
    return fused_silu_split_unsqueeze