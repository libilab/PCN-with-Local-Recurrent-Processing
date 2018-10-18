# PCN with Local Recurrent Processing
This repository contains the code for PCN with local recurrent processing introduced in the following paper:

[Deep Predictive Coding Network with Local Recurrent Processing for Object Recognition](https://arxiv.org/abs/1805.07526) (NIPS2018)

Kuan Han, Haiguang Wen, Yizhen Zhang, Di Fu, Eugenio Culurciello, Zhongming Liu

The code is built on Pytorch

## Introduction

Deep predictive coding network (PCN) with local recurrent processing is a bi-directional and dynamical neural network with local recurrent processing, inspired by predictive coding in neuroscience. Unlike any feedforward-only convolutional neural network, PCN includes both feedback connections, which carry top-down predictions, and feedforward connections, which carry bottom-up errors of prediction. Feedback and feedforward connections enable adjacent layers to interact locally and recurrently to refine representations towards minimization of layer-wise prediction errors. When unfolded over time, the recurrent processing gives rise to an increasingly deeper hierarchy of non-linear transformation, allowing a shallow network to dynamically extend itself into an arbitrarily deep network. We train and test PCN for image classification with SVHN, CIFAR and ImageNet datasets. Despite notably fewer layers and parameters, PCN achieves competitive performance compared to classical and state-of-the-art models. The internal representations in PCN converge over time and yield increasingly better accuracy in object recognition. 

![Image of pcav1](https://github.com/libilab/PCN_v2/blob/master/figures/Figure_1.jpg)
(a) The plain model (left) is a feedforward CNN with 3×3 convolutional connections (solid arrows) and 1×1 bypass connections (dashed arrows). 

(b) On the basis of the plain model, the local PCN (right) uses additional feedback (solid arrows) and recurrent (circular arrows) connections. The PCN consists of a stack of basic building blocks. Each block runs multiple cycles of local recurrent processing between adjacent layers, and merges its input to its output through the bypass connections. The output from one block is then sent to its next block to initiate local recurrent processing in a higher block. It further continues until reaching the top of the network.

## Usages

### To train PCN with local recurrent processing on ImageNet
For dependencies and the ImageNet dataset, see the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

As an example, the following command trains a PCN with default setting on ImageNet:
```bash
python main_imagenet.py --data /Path/to/ImageNet/Dataset/Folder
```

### To train PCN with local recurrent processing on CIFAR

As an example, the following command trains a PCN with default setting on CIFAR100:
```bash
python main_cifar.py
```

## Results

## Updates
10/17/2018:

(1) readme file.
