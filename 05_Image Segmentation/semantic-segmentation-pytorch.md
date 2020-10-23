# PyTorch for Semantic Segmentation
This repository contains some models for semantic segmentation and the pipeline of training and testing models, 
implemented in PyTorch

<p align="center">
  <a href="https://github.com/Charmve" target="_blank"><img src="https://img.shields.io/badge/Github-Charmve-blue" alt="Charmve"></a>
  <a href="https://github.com/Charmve/Semantic-Segmentation-PyTorch/tree/master/models" target="_blank"><img src="https://img.shields.io/badge/Github-Code-green" alt="Code"></a>
  <a href="https://github.com/Charmve/Semantic-Segmentation-PyTorch" target="_blank"><img src="https://img.shields.io/badge/Github-Repo-red" alt="Repo"></a>
</p>

## Models
1. Vanilla FCN: FCN32, FCN16, FCN8, in the versions of VGG, ResNet and DenseNet respectively
([Fully convolutional networks for semantic segmentation](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf))
2. U-Net ([U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/pdf/1505.04597))
3. SegNet ([Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://arxiv.org/pdf/1511.00561))
4. PSPNet ([Pyramid scene parsing network](https://arxiv.org/pdf/1612.01105))
5. GCN ([Large Kernel Matters](https://arxiv.org/pdf/1703.02719))
6. DUC, HDC ([understanding convolution for semantic segmentation](https://arxiv.org/pdf/1702.08502.pdf))
7. Mask-RCNN ([paper](https://arxiv.org/abs/1703.06870), [code from FAIR](https://github.com/facebookresearch/Detectron), [<b>code PyTorch</b>](https://github.com/multimodallearning/pytorch-mask-rcnn))

## Requirement
1. PyTorch 0.2.0
2. TensorBoard for PyTorch. [Here](https://github.com/lanpa/tensorboard-pytorch)  to install
3. Some other libraries (find what you miss when running the code :-P)

## Preparation
1. Go to *models* directory and set the path of pretrained models in *config.py*
2. Go to *datasets* directory and do following the README

## TODO
I'm going to implement <a href="https://github.com/Charmve/PaperWeeklyAI/tree/master/05_Image%20Segmentation" target="_blank"><i>The Image Segmentation Paper Top10 Net</i></a> in PyTorch firstly.

- [ ] DeepLab v3  
- [ ] RefineNet 
- [ ] ImageNet
- [ ] GoogleNet
- [ ] More dataset (e.g. ADE)
