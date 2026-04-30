import torch
import triton
import triton.language as tl


@triton.jit
def fused_patch_extract_kernel(
    out_ptr, in_0_ptr, in_1_ptr, in_2_ptr,
    n_total_patches, out_c, out_h, out_w,
    in_0_b, in_0_c, in_0_h, in_0_w,
    in_1_b, in_1_c, in_1_h, in_1_w,
    in_2_b, in_2_c, in_2_h, in_2_w,
    in_0_sb, in_0_sc, in_0_sh, in_0_sw,
    k_h, k_w, s1_h, s1_w, s2_h, s2_w,
    n_patches_1, n_patches_2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= n_total_patches:
        return
    
    patch_size = out_c * out_h * out_w
    
    if pid < n_patches_2:
        patch_local = pid
        patch_y = patch_local // 5
        patch_x = patch_local % 5
        patch_h_start = patch_y * s2_h
        patch_w_start = patch_x * s2_w
        in_ptr = in_2_ptr
        in_h, in_w = in_2_h, in_2_w
        in_sh, in_sw = in_2_h, in_2_w
    elif pid < n_patches_2 + n_patches_1:
        patch_local = pid - n_patches_2
        patch_y = patch_local // 3
        patch_x = patch_local % 3
        patch_h_start = patch_y * s1_h
        patch_w_start = patch_x * s1_w
        in_ptr = in_1_ptr
        in_h, in_w = in_1_h, in_1_w
        in_sh, in_sw = in_1_h, in_1_w
    else:
        patch_h_start = 0
        patch_w_start = 0
        in_ptr = in_0_ptr
        in_h, in_w = in_0_h, in_0_w
        in_sh, in_sw = in_0_sh, in_0_sw
    
    out_offset = pid * patch_size
    
    for i in range(BLOCK_SIZE):
        elem_idx = pid * BLOCK_SIZE + i
        if elem_idx >= n_total_patches * patch_size:
            break
        
        patch_elem = elem_idx - out_offset
        if patch_elem < 0 or patch_elem >= patch_size:
            continue
        
        c = patch_elem // (out_h * out_w)
        rem = patch_elem % (out_h * out_w)
        y = rem // out_w
        x = rem % out_w
        
        src_y = patch_h_start + y
        src_x = patch_w_start + x
        
        if src_y < in_h and src_x < in_w:
            src_offset = c * in_sh * in_sw + src_y * in_sw + x
            val = tl.load(in_ptr + src_offset)
        else:
            val = 0.0
        
        tl.store(out_ptr + elem_idx, val.to(tl.float16))


@torch.fx.wrap
def triton_fused_patch_extract(in_0, in_1, in_2):
    out_c, out_h, out_w = 3, 384, 384
    s1_h, s1_w = 192, 192
    s2_h, s2_w = 288, 288
    
    n_patches_1 = ((in_1.shape[2] - out_h) // s1_h + 1) * ((in_1.shape[3] - out_w) // s1_w + 1)
    n_patches_2 = ((in_2.shape[2] - out_h) // s2_h + 1) * ((in_2.shape[3] - out_w) // s2_w + 1)
    n_total = n_patches_1 + n_patches_2 + 1
    
    patch_size = out_c * out_h * out_w
    total_elements = n_total * patch_size
    
    BLOCK_SIZE = 1024
    num_programs = triton.cdiv(total_elements, BLOCK_SIZE)
    
    out = torch.empty((n_total, out_c, out_h, out_w), dtype=torch.float16, device=in_0.device)
    
    fused_patch_extract_kernel[(num_programs,)](
        out, in_0, in_1, in_2,
        n_total, out_c, out_h, out_w,
        in_0.size(0), in_0.size(1), in_0.size(2), in_0.size(3),
        in_1.size(0), in_1.size(1), in_1.size(2), in_1.size(3),
        in_2.size(0), in_2.size(1), in_2.size(2), in_2.size(3),
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out_h, out_w, s1_h, s1_w, s2_h, s2_w,
        n_patches_1, n_patches_2,
        BLOCK_SIZE,
    )
    
    return out


# Simple pattern for cat + dtype conversion
def pattern_cat_to(in_0, in_1, in_2):
    tmp_0 = torch.cat([in_2, in_1, in_0], dim = 0)
    tmp_1 = tmp_0.to(dtype = torch.float16)
    return tmp_1


def replacement_args_cat_to(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func_cat_to():
    return triton_fused_patch_extract


# Full pattern including unfold operations
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size = (384, 384), stride = (192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size = (384, 384), stride = (288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim = 0)
    tmp_7 = tmp_6.to(dtype = torch.float16)
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return triton_fused_patch_extract