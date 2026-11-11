# Neural model implementation record

## Official TabM verification

The extension uses the released `tabm` Python package, version 0.0.3, from the official [Yandex Research repository](https://github.com/yandex-research/tabm), not a reimplementation or an ordinary ensemble of independent MLPs. The installed package and API were inspected in the execution environment. The official `TabM.make` API accepts numerical inputs and training-fitted categorical cardinalities; its built-in categorical path uses one-hot encoding. Destination and HS2 codes are mapped using outer-training categories, with index zero reserved for unseen test categories.

The official package documentation states that inference averages the `k` member predictions and that classification should usually average member probabilities rather than logits. The extension implements this by applying sigmoid to each TabM member and then averaging; the regression path averages raw member outputs, then applies the training-validated inverse log transform. The chosen `k=8` is one official parameter-efficient model, not eight independent seed replications. The associated paper is [TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210).

## Neural architecture and data representation

MLP and TabM receive the same outcome-origin numeric variables and the same destination and HS2 identities. Numeric median imputation, missingness indicators, and scaling are fitted on the relevant training fold only. MLP uses a training-fitted one-hot transformer with unknown categories ignored. TabM uses the official categorical one-hot module and an explicit unknown-category index. Both architectures use ReLU, AdamW, minibatches of 1,024, and a fixed two-configuration chronological search; final outer models use the selected configuration and validation-selected epoch count. Classification minimizes class-weighted binary cross entropy; the sigmoid scores are ranking scores and are not claimed to be calibrated probabilities. Regression minimizes squared error on `log1p(USD)` and uses the validation-only multiplicative scale selected to minimize USD MAE.

## Reproducibility metadata

Each outer fold and seed stores its predictions in `predictions/_runs/`, model state in `checkpoints/models/`, fold-fitted preprocessing object, selected hyperparameters, selected epoch count, validation score, training rows, cutoff, parameter count, runtime, and package version. Seeds are not treated as independent economic observations in statistical inference.
