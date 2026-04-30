import torch
from pass_dir.layoutlm_shared import shared_layoutlm_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_13):
    tmp_15 = in_2[(slice(None, None, None), slice(None, 256, None))]
    tmp_16 = torch.nn.functional.embedding(in_0, in_9, 0, None, 2.0, False, False)
    tmp_17 = torch.nn.functional.embedding(tmp_15, in_6, None, None, 2.0, False, False)
    tmp_18 = in_13[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 = torch.nn.functional.embedding(tmp_18, in_10, None, None, 2.0, False, False)
    tmp_20 = in_13[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_21 = torch.nn.functional.embedding(tmp_20, in_11, None, None, 2.0, False, False)
    tmp_22 = in_13[(slice(None, None, None), slice(None, None, None), 2)]
    tmp_23 = torch.nn.functional.embedding(tmp_22, in_10, None, None, 2.0, False, False)
    tmp_24 = in_13[(slice(None, None, None), slice(None, None, None), 3)]
    tmp_25 = torch.nn.functional.embedding(tmp_24, in_11, None, None, 2.0, False, False)
    tmp_26 = in_13[(slice(None, None, None), slice(None, None, None), 3)]
    tmp_27 = in_13[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_28 = tmp_26 - tmp_27
    tmp_29 = torch.nn.functional.embedding(tmp_28, in_5, None, None, 2.0, False, False)
    tmp_30 = in_13[(slice(None, None, None), slice(None, None, None), 2)]
    tmp_31 = in_13[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_32 = tmp_30 - tmp_31
    tmp_33 = torch.nn.functional.embedding(tmp_32, in_8, None, None, 2.0, False, False)
    tmp_34 = torch.nn.functional.embedding(in_1, in_7, None, None, 2.0, False, False)
    tmp_35 = tmp_16 + tmp_17
    tmp_36 = tmp_35 + tmp_19
    tmp_37 = tmp_36 + tmp_21
    tmp_38 = tmp_37 + tmp_23
    tmp_39 = tmp_38 + tmp_25
    tmp_40 = tmp_39 + tmp_29
    tmp_41 = tmp_40 + tmp_33
    tmp_42 = tmp_41 + tmp_34
    tmp_43 = torch.nn.functional.layer_norm(tmp_42, (768,), in_4, in_3, 1e-12)
    tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False)
    return tmp_44


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_13):
    tmp_15 = in_2[(slice(None, None, None), slice(None, 256, None))]
    bbox0 = in_13[(slice(None, None, None), slice(None, None, None), 0)]
    bbox1 = in_13[(slice(None, None, None), slice(None, None, None), 1)]
    bbox2 = in_13[(slice(None, None, None), slice(None, None, None), 2)]
    bbox3 = in_13[(slice(None, None, None), slice(None, None, None), 3)]
    tmp_16 = torch.nn.functional.embedding(in_0, in_9, 0, None, 2.0, False, False)
    tmp_17 = torch.nn.functional.embedding(tmp_15, in_6, None, None, 2.0, False, False)
    tmp_19 = torch.nn.functional.embedding(bbox0, in_10, None, None, 2.0, False, False)
    tmp_21 = torch.nn.functional.embedding(bbox1, in_11, None, None, 2.0, False, False)
    tmp_23 = torch.nn.functional.embedding(bbox2, in_10, None, None, 2.0, False, False)
    tmp_25 = torch.nn.functional.embedding(bbox3, in_11, None, None, 2.0, False, False)
    tmp_29 = torch.nn.functional.embedding(bbox3 - bbox1, in_5, None, None, 2.0, False, False)
    tmp_33 = torch.nn.functional.embedding(bbox2 - bbox0, in_8, None, None, 2.0, False, False)
    tmp_34 = torch.nn.functional.embedding(in_1, in_7, None, None, 2.0, False, False)
    return (tmp_16, tmp_17, tmp_19, tmp_21, tmp_23, tmp_25, tmp_29, tmp_33, tmp_34, in_4, in_3, "add_ln")


def replacement_func():
    return shared_layoutlm_dispatch