# InfoGAN-MetaGx
InfoGAN applied to MetaGx dataset

# Train

Create conda environment which is used in this experiment:

`$ conda env create -f crd_env.yml`

Train the model: 

`(crd_env)$ python train.py config1.yml`

Extract embeddings (features) for future testing:

`(crd_env)$ python extract_features.py configs/k5_s48_metagx_train.yml configs/k5_s48_metagx_test.yml`

# Test

To test extracted embeddings:

`(crd_env)$ python test_unbalanced.py features/s48_c31_noNormMergedAnd5k/codes.csv configs/k5_s48_metagx.yml --n_splits 20`

# Extra

See a detailed report on this experiment [here](https://docs.google.com/document/d/1_pqSUXDiuxadQF4S7H9-ZtN7p6aWi61uTK7YRVlyt_k/edit?usp=sharing).
