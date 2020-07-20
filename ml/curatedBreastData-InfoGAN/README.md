# curatedBreastData-InfoGAN
InfoGAN applied to curatedBreastData

# Train
Download and extract [data](https://drive.google.com/drive/folders/1jJO6NicisDHqqgiUswL2ZhVFhSYBozd5?usp=sharing).

Create conda environment which is used in this experiment:

`$ conda env create -f crd_env.yml`

Train the model: 

`(crd_env)$ python train.py --config config1.yml`

Extract embeddings (features) for future testing:

`(crd_env)$ python python extract_embeddings.py`

# Test

To test extracted embeddings:

`(crd_env)$ python experement2.6_combat_pcr_rfs_dfs.py data/codes_<S-DIM>.csv`

# Extra

See a detailed report on this experiment [here](https://docs.google.com/document/d/1XXD-Zion6pDzhhN8KUhSwAvKTNWuINc-FHrtPcetfH8/edit?usp=sharing).
