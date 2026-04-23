import torch
from pass_dir.shared_relpos_bias_softmax import replacement_func
from graph_net_bench.torch.backend import pass_mgr_backend as _pmb


if not hasattr(_pmb, "_ai4c_debug_print_graph_once"):
    _orig_prp_call = _pmb.PatternReplacementPass.__call__
    def _debug_prp_call(self, gm):
        print("[AI4C DEBUG] FX graph start")
        print(gm.graph)
        print("[AI4C DEBUG] FX graph end")
        try:
            print("[AI4C DEBUG] Pattern graph start")
            print(torch.fx.symbolic_trace(self.pattern).graph)
            print("[AI4C DEBUG] Pattern graph end")
        except Exception as e:
            print(f"[AI4C DEBUG] Pattern graph trace failed: {e}")
        _pmb._ai4c_debug_print_graph_once = True
        _pmb.PatternReplacementPass.__call__ = _orig_prp_call
        return _orig_prp_call(self, gm)
    _pmb.PatternReplacementPass.__call__ = _debug_prp_call


def pattern(in_0: torch.Tensor, in_1, in_2, in_3, in_4):
    linear = torch.nn.functional.linear(in_4, in_1, None)
    tmp_3 = linear.view(-1, 12)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    tmp_6 = tmp_5.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 64, 12, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 12, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return (tmp_22,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)