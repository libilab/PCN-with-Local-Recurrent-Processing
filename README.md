# PCN_v2
This repository contains the code for PCN v2 introduced in the following paper:

[Deep Predictive Coding Network with Local Recurrent Processing for Object Recognition](https://arxiv.org/abs/1805.07526) (NIPS2018)

Kuan Han, Haiguang Wen, Yizhen Zhang, Di Fu, Eugenio Culurciello, Zhongming Liu

The code is built on Pytorch

## Introduction

Deep predictive coding network (PCN) v2 is a bi-directional and dynamical neural network with local recurrent processing, inspired by predictive coding in neuroscience. Unlike any feedforward-only convolutional neural network, PCN includes both feedback connections, which carry top-down predictions, and feedforward connections, which carry bottom-up errors of prediction. Feedback and feedforward connections enable adjacent layers to interact locally and recurrently to refine representations towards minimization of layer-wise prediction errors. When unfolded over time, the recurrent processing gives rise to an increasingly deeper hierarchy of non-linear transformation, allowing a shallow network to dynamically extend itself into an arbitrarily deep network. We train and test PCN for image classification with SVHN, CIFAR and ImageNet datasets. Despite notably fewer layers and parameters, PCN achieves competitive performance compared to classical and state-of-the-art models. The internal representations in PCN converge over time and yield increasingly better accuracy in object recognition. 
