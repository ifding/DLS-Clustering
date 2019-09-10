

#python Gene_ClusterGAN.py --data 10x_73k --K 8 --dz 30 --beta_n 1 --beta_c 10 --train True

#python Gene_wgan_gp.py --data 10x_73k --sampler 'mul_cat' --K 8 --dz 30 --train True

#python Gene_ClusterGAN.py --data ibd --K 3 --dz 10 --beta_n 1 --beta_c 10 --train True

#python Gene_wgan_gp.py --data ibd --K 3 --dz 10 --train True

python Image_ClusterGAN.py --data fashion --K 10 --dz 40 --beta_n 1 --beta_c 10 --train True