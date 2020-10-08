# AMoC (Adaptive Momentum Coefficient for Neural Network Optimization)

## About

The code accompanying our ECML-PKDD 2020 paper [Adaptive Momentum Coefficient for Neural Network Optimization](https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1005.pdf).

## Summary

Adaptive Momentum Coefficient (AMoC) utilizes the inner product of the gradient and the previous update to the parameters, to effectively control the amount of weight put on the momentum term based on the change of direction in the optimization path. 

## Contents

The code contains an implementation of the AMoC optimizer along with the deep Autoencoder experiments included in the paper. Variables and hyperparameters can be modified in the scripts. The requirements are Python 3.8, Pytorch 1.4.0 along with CUDA 10.1 and CuDNN 7.6.3.



