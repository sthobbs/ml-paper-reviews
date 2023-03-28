
## Table of Contents
1. Computer Vision  
    1. Image classification
    2. Segmentation



## Computer Vision

### Image classification
- `2015` [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
    - Initial ResNet paper. Introduced the idea of residual blocks, where the input is added (element wise) to the output, so the weights learn the difference (i.e. the residual) between in the input and the output. This innovation allows neural nets to be more efficient and with more layers, and was used in this paper to get what was SOTA image classification results at the time.

### Segmentation
- `2015` [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    - Interesting CNN architecture for image segmentation which involves a contracting path to capture context and a symmetric expanding path that enables precise localization.  Skip connections are introduced to connect the contracting and expanding paths, which allow the network to access both low- and high-level features and preserve spatial information.

