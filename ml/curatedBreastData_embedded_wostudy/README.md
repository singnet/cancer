# curatedBreastData-embeddings invariant to study
Discrimination/Classification model applied to curatedBreastData

# Train
Download and extract [data](https://drive.google.com/drive/folders/1jJO6NicisDHqqgiUswL2ZhVFhSYBozd5?usp=sharing).

Create conda environment which is used in this experiment:

`$ conda env create -f crd_env.yml`

Train the model: 

`(crd_env)$ python train.py config1.yml`

# Test

Model is tested during the training procedure. Have a look at saved summary:

`(crd_env)$ tensorboard --logdir=summary`

# Extra

More details in the [report](https://docs.google.com/document/d/1Fs87cJqEu0e7rWZT8xMgth-mqQqT322rEPZgDYZWyxU/edit?usp=sharing).
