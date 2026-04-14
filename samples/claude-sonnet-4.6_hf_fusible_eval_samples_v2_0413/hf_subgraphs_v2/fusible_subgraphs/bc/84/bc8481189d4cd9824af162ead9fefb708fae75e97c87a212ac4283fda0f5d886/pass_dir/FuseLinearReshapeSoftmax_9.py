import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_linear_softmax_kernel(
    in_0_ptr,                       # bias  [OUT_F]
    in_1_ptr,                       # weight [OUT_F, HIDDEN_DIM]
    in_2_ptr,                       # input  [B_S, HIDDEN_DIM]
    out_ptr,                        # output [TOTAL_GROUPS * GROUP_SIZE]
    HIDDEN_DIM: tl.constexpr,       # 128
    GROUP_SIZE: tl.constexpr,       # 9
    PAD_SIZE: tl.constexpr,         # 16  (next power-of-2 >= 9)
    GROUPS_PER_SEQ: tl.constexpr,   # 2
    IS_BF16: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One thread-block per output group (38 blocks total)
    group_id   = tl.program_id(0)
    seq_id     = group_id // GROUPS_PER_SEQ
    feat_start = (group_id % GROUPS_PER_SEQ) * GROUP_SIZE

    input_base   = seq_id * HIDDEN_DIM
    feat_offsets = feat_start + tl.arange(0, PAD_SIZE)   # [16]
    valid_mask   = tl.arange(0, PAD_SIZE) < GROUP_SIZE    # first 9 True

    acc = tl.zeros([PAD_SIZE], dtype=tl.float32)

    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)                     # [BLOCK_K]

        x = tl.load(in_2_ptr + input_base + k_offsets).to(tl.float32)  # [BLOCK_K]

        w_ptrs = in_1_ptr + feat_offsets[:, None] * HIDDEN_DIM + k_offsets[None, :]
        w = tl.load(w_ptrs, mask=valid_mask[:, None], other=0.0).to(tl.float32)  # [16, BLOCK_K]

        acc += tl.sum(w * x[None, :], axis=1)                           # [16]

    # Add bias
    bias = tl.load(in_0_ptr + feat_offsets, mask=valid_mask, other=0.0).to(tl.float32)
    acc  += bias

    # In-register softmax over the 9 valid elements
    acc      = tl.where(valid_mask, acc, -1e9)
    max_val  = tl.max(acc, axis=0)
    exp_vals = tl.exp(acc - max_val)
    exp_vals = tl.where(valid_mask, exp_vals, 0.0)
    sm_vals  = exp_vals / tl.sum(exp_vals, axis=0)

    # Store valid outputs
    out_offsets = group_id * GROUP_SIZE + tl.arange(0, PAD_SIZE)
    if IS_BF16:
        tl.store(out_ptr + out_offsets, sm_vals.to(tl.bfloat16), mask=valid_mask)
    else:
        tl.store(out_ptr + out_offsets, sm_vals.to(tl.float16), mask=valid_mask)


@torch.fx.wrap
def fused_linear_reshape_softmax(in_0, in_1, in_2):
    # in_0 : bias   [OUT_F]
    # in_1 : weight [OUT_F, H]
    # in_2 : input  [B, S, H]

    H      = in_2.shape[-1]           # 128
    B_S    = in_2.numel() // H        # 19
    OUT_F  = in_1.shape[0]            # 18

    GROUP_SIZE     = 9
    PAD_SIZE       = 16
    GROUPS_PER_SEQ = OUT_F // GROUP_SIZE   # 2
    total_groups   = B_S * GROUPS_PER_SEQ  # 38

    is_bf16 = 1 if in_2.dtype == torch.bfloat16 else 0

    out = torch.empty((total_groups, GROUP_SIZE, 1), dtype=in_2.dtype, device=in_2.device)

    fused_linear_softmax_kernel[(total_groups,)](
        in_0,
        in_1,
        in_2,
        out,
        HIDDEN_DIM=H,
        GROUP_SIZE=GROUP_SIZE,
        PAD_SIZE=PAD_SIZE,
        GROUPS_PER_SEQ=GROUPS_PER_SEQ,
        IS_BF16=is_bf16,
        BLOCK_K=H,
        num_warps=1,
    )

    return out


def replacement_func():
    return fused_linear_reshape_softmax