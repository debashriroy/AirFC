# Implementation of Neural Network Training for AirFC project

This repository is to generate the trained model to validate over the air computation for fully connected neural network. The project used the widely used MNIST dataset for the handwritten digital recognition.  The overall goal of thit github repo is to generate the model weights after training a complex-valued neural netowrk to validate the over the air analog computation resembling a fully connected neural network in real testbed setup. 

## Citing This Paper
Please cite the following paper if you intend to use this code for your research.

G. Reus-Muns, K. Alemdar, S. G. Sanchez, D. Roy, and K. R. Chowdhury, “AirFC: Designing Fully Connected Layers for Neural Networks with Wireless Signals,” ACM International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc), July, 2023.

### Bibtex:

 `@INPROCEEDINGS{RoyInfocom22,
author = {Reus-Muns, Guillem  and Alemdar, Kubra and Garcia Sanchez, Sara and Roy, Debashri and Chowdhury, Kaushik R.},
title = {{AirFC: Designing Fully Connected Layers for Neural Networks with Wireless Signals}},
booktitle = {{International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc)}},
  year={2023},
  month = {July}
  }`

## Run

To run this code where we are using noise to train the model: `python main_with_noise.py --hidden_elements 16`

To run this code where we are not using noise to train the model: `python main_no_noise.py --hidden_elements 16`

## Dataset
We use MNIST dataset for the training and evaluation.



## Acknowledgments

We use the open source code published in: https://github.com/wavefrontshaping/complexPyTorch
