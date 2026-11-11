# GRU method and chronological validation

The primary network is a one-layer, 32-unit CPU GRU followed by a 16-unit ReLU head with dropout 0.1 and a binary logit. The sequence contains 15 annual trade-demand, bilateral-export and macroeconomic variables; missingness channels are appended. The sequence-plus-static model concatenates a 16-unit geography/destination/product representation. Training uses AdamW (learning rate .001, weight decay .0001), batch size 256, at most 40 epochs, patience 6, BCEWithLogitsLoss with training-only positive class weight, and seeds 11, 23, 37, 51, 73.

For outer outcome year y, training labels are observed only through y−2. The latest admissible training outcome year is used as chronological validation, with inner training labels ending at validation year−2. The best validation PR-AUC epoch is frozen, then the model is refit on all outcomes available by the outer cutoff for that number of epochs. Imputation and scaling are trained within each fold. The lag-matched LightGBM flattens the same five-year sequence and uses identical static information.

The outcome is binary entry at USD 50,000; missing outcomes remain missing. This is a standard GRU application, not a new recurrent architecture.
