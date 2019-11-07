
# DLS-Clustering

Code for reproducing key results in the paper Clustering by Directly Disentangling Latent Space by Fei Ding and Feng Luo. If you use the code, please cite our paper.

## Dependencies 

The code has been tested with the following versions of packages.

- Python 3
- Tensorflow 1.14.0
- Numpy 1.14.2

## Datasets

The datasets used in the paper can be downloaded from the Google Drive link (https://drive.google.com/open?id=1XnGkSamF5DiwnpHFG0OexmoqAwe27ucR) and (https://drive.google.com/drive/folders/0B9J-9A2jotGRT25vSDhUWTQxVWs).

Unzip the folder so that the path is : ./ClusterGAN/data/<dataset_name>

## Training

You can either train your own models on the datasets or use pre-trained models. Even though we have used a fixed seed using tf.random.seed(0), there will still be randomness introduced by CUDA. So, to reproduce the results, train 5 models and compare the Validation purity in the logs directory. Each model can be trained as follows :

```bash
$ python Image_Cluster.py --data mnist --K 10 --dz 25 --beta_n 1 --beta_c 10 --train True 
```

This will save the model along with timestamp in `checkpoint-dir/<dataset_name>`. Also, the Validation set performance will be written to `logs/Res_<dataset_name>_<model_name>.txt`. Then run the best model (with highest Validation Purity) on the Test set. 


```bash
$ python Image_Cluster.py --data mnist --K 10 --dz 25 --beta_n 1 --beta_c 10 --timestamp <best_timestamp>
```

Training the models for other datasets has a similar format.

Fashion-10 : 
```bash
$ python Image_Cluster.py --data fashion --K 10 --dz 40 --beta_n 1 --beta_c 10 --train True 
```

YTF : 
```bash
$ python Image_Cluster.py --data ytf --K 41 --dz 60 --beta_n 1 --beta_c 10 --train True 
```

Coil-100 : 
```bash
$ python Image_Cluster.py --data coil --K 100 --dz 100 --beta_n 1 --beta_c 10 --train True 
```

Single Cell 10x genomics : 
```bash
$ python Seq_Cluster.py --data 10x_73k --K 8 --dz 30 --beta_n 1 --beta_c 10 --train True 
```

Pendigits : 
```bash
$ python Pen_Cluster.py --data pendigit --K 10 --dz 5 --beta_n 1 --beta_c 10 --train True 
```

Provide the timestamp of best saved model to obtain the Test set clustering performance on all the datasets (similar to MNIST above).

## Pre-trained models

Additionally, you can also download the pre-trained models from the Google drive link (https://drive.google.com/open?id=1l9Lwq0amAaA3qHzNCiw7BrivSAFoP0em). Unzip the file in ./ClusterGAN. It should lead to the folder ./ClusterGAN/pre_trained_models

Run the following code : 

```bash
$ python Image_ClusterGAN.py --data mnist --K 10 --dz 30 --beta_n 10 --beta_c 10 
```

Similarly for the other datasets.



## Reference

- https://github.com/sudiptodip15/ClusterGAN
- https://github.com/tolstikhin/wae
