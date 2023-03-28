
## Table of Contents
1. [Computer Vision](#computer-vision)  
    1. [Image Classification](#image-classification)
    2. [Segmentation](#segmentation)
2. [Language Models](#language-models)
    1. [Attention](#attention)
3. [Normalization](#normalization)


## Computer Vision

### Image Classification
- `2015` [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
    - Initial ResNet paper. Introduced the idea of residual blocks, where the input is added (element wise) to the output, so the weights learn the difference (i.e. the residual) between in the input and the output. This innovation allows neural nets to be more efficient and with more layers, and was used in this paper to get what was SOTA image classification results at the time.

### Segmentation
- `2015` [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    - Interesting CNN architecture for image segmentation which involves a contracting path to capture context and a symmetric expanding path that enables precise localization.  Skip connections are introduced to connect the contracting and expanding paths, which allow the network to access both low- and high-level features and preserve spatial information.


## Language Models

### Attention
- `2017` [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
	- This paper introduced the Transformer, which was the next major milestone in sequence-to-sequence modeling tasks after RNNs and LSTMs, and is a key building block for many of the large language models we have today. The Transformer uses a fully self-attention-based approach, eliminating the need for recurrent or convolutional layers, and making it more parallelizable and efficient to train. The paper introduces a multi-head attention mechanism that enables the model to focus on multiple parts of the input simultaneously, providing richer information exchange between encoder and decoder.

## Normalization
- `2015` [Batch Normalization](https://arxiv.org/abs/1502.03167)
	- Batch Normalization (Batch Norm) is a widely adopted technique in deep learning still frequently used today. At each Batch Norm layer, data within each mini-batch is normalized, this leads to faster training and less care required when initializing weights.


