
## Table of Contents
1. [Tabular Models](#tabular-models)
2. [Computer Vision](#computer-vision)  
    1. [Image Classification](#image-classification)
    2. [Video Classification](#video-classification)
    3. [Segmentation](#segmentation)
3. [Language Models](#language-models)
    1. [Attention](#attention)
    2. [LLMs](#llms)
4. [Auto Encoders](#auto-encoders)
5. [GANs](#gans)
6. [Normalization](#normalization)


## Tabular Models
- `2021` [Tabular Data: Deep Learning Is Not All You Need](#https://arxiv.org/abs/2106.03253)
    - Over the past few years, several deep learning models have claimed to outperform XGBoost on tabular data. This paper comprehensively reviews these models on a bunch of datasets and concludes that XGBoost is still the best single-model approach for tabular data, however an ensemble of XGBoost and deep learning models performs even better. 

## Computer Vision

### Image Classification
- `2015` [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
    - Initial ResNet paper. Introduced the idea of residual blocks, where the input is added (element wise) to the output, so the weights learn the difference (i.e. the residual) between in the input and the output. This innovation allows neural nets to be more efficient and with more layers, and was used in this paper to get what was SOTA image classification results at the time.
- `2020` [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
    - Larger CNNs outperform smaller CNNs, but there are many ways to make a CNN bigger. You can scale the number of layers, number of channels and/or image resolution. The authors purpose a novel scaling coefficient that balances these three methods in a systematic way to achieve the best possible trade-off between model size and accuracy.
- `2022` [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
    - A modern take on CNNs architectures.

### Video Classification
- `2014` [Large-scale Video Classification with Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)
    - One of the first attempts at applying deep CNNs to video classification.

### Segmentation
- `2015` [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    - Interesting CNN architecture for image segmentation which involves a contracting path to capture context and a symmetric expanding path that enables precise localization.  Skip connections are introduced to connect the contracting and expanding paths, which allow the network to access both low- and high-level features and preserve spatial information.


## Language Models

### Attention
- `2017` [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
	- This paper introduced the Transformer, which was the next major milestone in sequence-to-sequence modeling tasks after RNNs and LSTMs, and is a key building block for many of the large language models we have today. The Transformer uses a fully self-attention-based approach, eliminating the need for recurrent or convolutional layers, and making it more parallelizable and efficient to train. The paper introduces a multi-head attention mechanism that enables the model to focus on multiple parts of the input simultaneously, providing richer information exchange between encoder and decoder.

### LLMs
- `2022` [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
    - *Emergence* is a physics term that describes how quantitative changes can lead to qualitatively different results (e.g. gradually lowering the temperature of water will at some point cause the water to turn to ice). Large language models with 100s of billions of parameters have recently shown amazing abilities (e.g. logical reasoning, arithmetic). Interestingly, as you increase the size of the model, many of these abilities are suddenly acquired rather than gradually improved. Smaller models can't do these tasks at all, they're no better than random, but once the model is large enough, they can do them. This paper studies how a quantitative change in model size leads to a qualitative change in abilities.



## Auto Encoders
- `2020` [Auto Encoders](https://arxiv.org/abs/2003.05991)
    - This paper gives an overview of auto encoders and lists many interesting applications for them.

## GANs
- `2014` [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
    - The initial GANs paper. The idea is that you train a Generator model which takes random noise as input and outputs synthetic data, and a Discriminator model which classifies data as either real or synthetic. These two models are iteratively trained and both get gradually better.

## Normalization
- `2015` [Batch Normalization](https://arxiv.org/abs/1502.03167)
	- Batch Normalization (Batch Norm) is a widely adopted technique in deep learning still frequently used today. At each Batch Norm layer, data within each mini-batch is normalized, this leads to faster training and less care required when initializing weights. It also acts as a regularizer.


