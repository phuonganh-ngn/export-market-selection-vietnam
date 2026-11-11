import torch

from experiments.post_audit_nn_dl_v1.src.models import MLP, seed_everything
from experiments.post_audit_nn_dl_v1.src.models import FoldEncoder
from experiments.post_audit_nn_dl_v1.src.data import feature_columns
import pandas as pd


def test_official_tabm_member_output_shape_and_mean_semantics():
    import tabm

    model = tabm.TabM.make(n_num_features=4, cat_cardinalities=[3, 4], d_out=1,
                           n_blocks=2, d_block=16, dropout=0.0, k=8,
                           arch_type="tabm", start_scaling_init="random-signs")
    x_num = torch.randn(5, 4)
    x_cat = torch.tensor([[1, 2], [2, 3], [0, 1], [1, 1], [2, 2]], dtype=torch.long)
    output = model(x_num, x_cat)
    assert output.shape == (5, 8, 1)
    probabilities = torch.sigmoid(output.squeeze(-1)).mean(dim=1)
    assert probabilities.shape == (5,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_official_tabm_initialization_repeats_under_fixed_seed():
    import tabm
    from experiments.post_audit_nn_dl_v1.src.models import seed_everything

    seed_everything(37)
    a = tabm.TabM.make(n_num_features=3, cat_cardinalities=[4, 5], d_out=1,
                       n_blocks=2, d_block=16, dropout=0.1, k=8,
                       arch_type="tabm", start_scaling_init="random-signs").state_dict()
    seed_everything(37)
    b = tabm.TabM.make(n_num_features=3, cat_cardinalities=[4, 5], d_out=1,
                       n_blocks=2, d_block=16, dropout=0.1, k=8,
                       arch_type="tabm", start_scaling_init="random-signs").state_dict()
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_unknown_categories_map_to_reserved_zero_and_seed_is_repeatable():
    num_cols, _ = feature_columns("F4_full")
    frame = pd.DataFrame({c: [0.0, 1.0] for c in num_cols})
    frame["destination_iso3"] = ["AAA", "BBB"]
    frame["hs2"] = ["01", "02"]
    encoder = FoldEncoder.fit(frame, "TabM", "F4_full")
    unknown = frame.iloc[[0]].copy()
    unknown["destination_iso3"] = "ZZZ"
    _, x_cat = encoder.transform(unknown)
    assert x_cat[0, 0] == 0
    seed_everything(73)
    a = MLP(4, 8, 2, 0.1).state_dict()
    seed_everything(73)
    b = MLP(4, 8, 2, 0.1).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
