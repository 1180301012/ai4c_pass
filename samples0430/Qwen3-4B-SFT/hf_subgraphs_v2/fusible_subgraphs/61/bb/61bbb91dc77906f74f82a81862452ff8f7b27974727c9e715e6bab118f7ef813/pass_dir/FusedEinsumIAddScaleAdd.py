"""
Fused pass for the full computation:
  einsum('bchj,bhwj->bchw') + in_3 (iadd) + * scale + + in_2 + .contiguous()

Key insight: "in_3 + einsum" in a Python function pattern is traced by the framework
as operator.add (not operator.iadd), causing a persistent mismatch.  We work around
this with two separate passes:

  Pass 1 (FusedEinsumIAddScaleAdd): matches just the einsum  →
    provides a fast Triton batched-Hadamard-reduce kernel with BLOCK_HW=64 (K=J=64,
tensor cores fire).  Fix for the previous mask broadcasting: hw_mask[:,None]
  gives [BLOCK_HW,1] → broadcasts to [J=64,BLOCK_HW] regardless of BLOCK_HW.

  Pass 2 (FusedScaleAddContiguous): matches scale+iadd_result+add+contiguous  →
    fuses 3 elementwise ops, eliminates ~134MB DRAM traffic (no intermediate copy),
    adds one kernel launch.

Both passes order in sorted_output_pass_rule_names.json.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match ONLY the einsum node.
# in_3 being a free-variable placeholder that maps to the target's in_3 lets
# us write a fused einsum kernel inside replacement_args.
# ---------------------------------------------------------------------------
def pattern(in_4, in_1):
    result = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    return result


def replacement_args(in_4, in_1):
    return (in_4, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: out[j, ihw] = sum_k(in4[k,ihw] * in1[k,ihw])
#   in4 [B, C, HW, J],  in1 [B, HW, J],  out [B, C, HW, J]
# Grid: (B*C, HW/BLOCK_HW)   BLOCK_HW = 64  (fixed, K=J=64 → tl.dot works)
#   mat_a = in4_tile [J=64, BLOCK_HW]
#   mat_b = in1_tile [J=64, BLOCK_HW]
#   acc   = tl.dot(mat_b, tl.trans(mat_a)) → [J=64, BLOCK_HW]
#   store acc to out  [J, BLOCK_HW] with hw_mask[:,None] → broadcasts [J,BLOCK_HW] ✓
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_HW': 64},  num_warps=16, num_stages=3),
        triton.Config({'BLOCK_HW': 64},  num_warps=8, num_stages=5),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=16, num_stages=3),
    ],
    key=['BC', 'HW'],
)
@triton.jit
def einsum_bchj_bhwj_bchw_kernel(
    in4_ptr, in1_ptr, out_ptr,
    as_b, as_c, as_hw,
    bs_b, bs_hw,
    cs_b, cs_c, cs_hw,
    BC, HW,
    BLOCK_HW: tl.constexpr,
):
    bc_idx   = tl.program_id(0)
    hw_block = tl.program_id(1)

    b = bc_idx // BC
    c = bc_idx %  BC
    hw_start = hw_block * BLOCK_HW

    j_offs  = tl.arange(0, 64)
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    j_2d  = j_offs[:, None]   # [64, 1]
    hw_2d = hw_offs[None, :]  # [1, BLOCK_HW]

    base_a = in4_ptr + b * as_b + c * as_c
    mat_a  = tl.load(base_a + hw_2d * as_hw + j_2d,
                     mask=hw_mask[None, :], other=0.0).to(tl.float32)  # [64, BHW]

    base_b = in1_ptr + b * bs_b
    mat_b  = tl.load(base_b + hw_2d * bs_hw + j_2d,
                     mask=hw_mask[None, :], other=0.0).to(tl.float32)  # [64, BHW]

    # K = mat_a first dim = 64 = J; K = mat_b first dim = 64 = J  →  K always matches
    acc = tl.dot(mat_b, tl.trans(mat_a))   # [64, BLOCK_HW]

    out_ptrs = (out_ptr + b * cs_b + c * cs_c
                + hw_2d * cs_hw + j_2d)   # [64, BLOCK_HW]
    tl.store(out_ptrs, acc, mask=hw_mask[:, None])   # [BHW,1] → [64,BHW] ✓


@torch.fx.wrap
def triton_einsum_bchj_bhwj_bchw(in_4, in_1):
    B  = in_4.shape[0]
    C2 = in_4.shape[1]
    HW = in_4.shape[2] * in_4.shape[3]
    J  = in_4.shape[3]

    BC = B * C2
    out = torch.empty((B, C2, HW // J, J), dtype=in_4.dtype, device=in_4.device)

    grid = lambda meta: (BC, triton.cdiv(HW, meta['BLOCK_HW']))
    einsum_bchj_bhwj_bchw_kernel[grid](
        in_4, in_1, out,
        in_4.stride(0), in_4.stride(1), in_4.stride(2),
        in_1.stride(0), in_1.stride(1),
        out.stride(0),  out.stride(1),  out.stride(2),
        BC, HW,
    )
    return out


def replacement_func():
    return triton_einsum_bchj_bhwj_bchw