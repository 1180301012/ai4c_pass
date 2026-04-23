import torch

from pass_dir.botnet_rel_logits_common import _botnet_rel_logits_impl
from pass_dir.botnet_full_attn_common import _botnet_full_attn_impl


@torch.fx.wrap
def botnet_hybrid_dispatch(*args):
    route = args[-1]
    if route == "botnet_full_attn_16":
        return _botnet_full_attn_impl(args[0], args[1], args[2], args[3], args[4], 16)
    if route == "botnet_rel_logits_8":
        return _botnet_rel_logits_impl(args[0], args[1], args[2], 8)
    raise RuntimeError(f"Unsupported route: {route}")