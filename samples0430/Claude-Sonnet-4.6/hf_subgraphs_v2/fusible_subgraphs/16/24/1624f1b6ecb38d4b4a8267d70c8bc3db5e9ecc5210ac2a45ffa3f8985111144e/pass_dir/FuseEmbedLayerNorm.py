import torch
from pass_dir.shared_kernels import shared_dispatch_kernel, replacement_func  # noqa: F401


# ---------------------------------------------------------------------------
# Pattern: match flatten → transpose → tile → cat → add
#   Returns tmp_11 (single value — framework requires single return)
#   dropout(p=0) and layer_norm remain in graph, fed by the kernel output
# ---------------------------------------------------------------------------
def pattern(conv3d_out, cls_token, pos_emb):
    tmp_7  = conv3d_out.flatten(2)
    tmp_8  = tmp_7.transpose(1, 2)
    tmp_9  = cls_token.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + pos_emb
    return tmp_11


def replacement_args(conv3d_out, cls_token, pos_emb):
    # Append "preproc" route so shared_dispatch_kernel dispatches correctly
    return (conv3d_out, cls_token, pos_emb, "preproc")