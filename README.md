
## Table of Contents
1. [Tabular Models](#tabular-models)
2. [Computer Vision](#computer-vision)  
    1. [Image Classification](#image-classification)
    2. [Object Detection](#object-detection)
    3. [Video Classification](#video-classification)
    4. [Segmentation](#segmentation)
    5. [Super-Resolution](#super-resolution)
    6. [Text-to-Image](#text-to-image)
3. [Language Models](#language-models)
    1. [Attention](#attention)
    2. [LLMs](#llms)
4. [Reinforcement Learning](#reinforcement-learning)
5. [Auto Encoders](#auto-encoders)
6. [GANs](#gans)
7. [General Deep Learning](#general-deep-learning)
8. [Normalization](#normalization)
9. [Stochastic Regularization](#stochastic-regularization)
10. [Optimization](#optimization)
    1. [Gradient-Based Optimization](#gradient-based-optimization)
    2. [Bayesian Optimization](#bayesian-optimization)

## Tabular Models
- `2021` [Tabular Data: Deep Learning Is Not All You Need](https://arxiv.org/abs/2106.03253)
    - Over the past few years, several deep learning models have claimed to outperform XGBoost on tabular data. This paper comprehensively reviews these models on a bunch of datasets and concludes that XGBoost is still the best single-model approach for tabular data, however an ensemble of XGBoost and deep learning models performs even better. 
- `2022` [Why do tree-based models still outperform deep learning on tabular data](https://arxiv.org/abs/2207.08815)
    - This paper reinforces the claim that tree-based models outperform deep learning models on tabular data and tries to explain why that is. The paper suggests 3 reasons:
        - Deep learning models provide smooth solutions, tree-based models don't, and often the true underlying relationship of tabular data is not smooth.
        - Tabular data tends to have more useless features (i.e. features that aren't predictive at all), and tabular models are better at handling these.
        - Many deep learning models have a rotation invariance property where if you multiply the training/test data by a rotation matrix it will train just as well. Trying this with tree-based models gives the best results when using the original orientation, which suggests that rotation invariance is an undesirable property.


## Computer Vision

### Image Classification
- `2015` [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet)
    - Initial ResNet paper. Introduced the idea of residual blocks, where the input is added (element wise) to the output, so the weights learn the difference (i.e. the residual) between in the input and the output. This innovation allows neural nets to be more efficient and with more layers, and was used in this paper to get what was SOTA image classification results at the time.
- `2016` [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
    - Adds residual connections to the Inception architecture.
- `2020` [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
    - Larger CNNs outperform smaller CNNs, but there are many ways to make a CNN bigger. You can scale the number of layers, number of channels and/or image resolution. The authors propose a novel scaling coefficient that balances these three methods in a systematic way to achieve the best possible trade-off between model size and accuracy.
- `2022` [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
    - A modern take on CNNs architectures.

### Object Detection
- `2015` [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
    - This object detection model has two parts, (1) a network that generates candidate bounding boxes around potential objects in the image, and (2) a network that classifies the proposed regions. I prefer the You Only Look Once (YOLO) formulation for object detection since the whole end-to-end pipeline is trained jointly.
- `2016` [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640#)
    - The authors propose a real-time object detection system based on a single neural network architecture, which only passes over the image once, making it much faster than traditional object detection systems.

### Video Classification
- `2014` [Large-scale Video Classification with Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)
    - One of the first attempts at applying deep CNNs to video classification.
- `2015` [Beyond Short Snippets: Deep Networks for Video Classification](https://arxiv.org/abs/1503.08909)
    - The paper combines CNN layers and LSTM for video classification on longer clips.

### Segmentation
- `2015` [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
    - Interesting CNN architecture for image segmentation which involves a contracting path to capture context and a symmetric expanding path that enables precise localization.  Skip connections are introduced to connect the contracting and expanding paths, which allow the network to access both low- and high-level features and preserve spatial information.

### Super-Resolution
- `2015` [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
    - Image enhancement with a straightforward application of CNNs.

### Text-to-Image
- `2022` [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)
    - SOTA text-to-image generation, outperforming DALL-E 2.

## Language Models

### Attention
- `2017` [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
	- This paper introduced the Transformer, which was the next major milestone in sequence-to-sequence modeling tasks after RNNs and LSTMs, and is a key building block for many of the large language models we have today. The Transformer uses a fully self-attention-based approach, eliminating the need for recurrent or convolutional layers, and making it more parallelizable and efficient to train. The paper introduces a multi-head attention mechanism that enables the model to focus on multiple parts of the input simultaneously, providing richer information exchange between encoder and decoder.
- `2019` [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
	- Another big step forward in language models. BERT learns full-context word representations by using a bidirectional transformer architecture. The model learns in 2 stages: (1) pre-training on unlabeled text data, and (2) fine-tuning on a specific task.


### LLMs
- `2022` [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
    - The author's create an open source replica of GPT3 and discuss ML ethics.
- `2022` [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
    - *Emergence* is a physics term that describes how quantitative changes can lead to qualitatively different results (e.g. gradually lowering the temperature of water will at some point cause the water to turn to ice). Large language models with 100s of billions of parameters have recently shown amazing abilities (e.g. logical reasoning, arithmetic). Interestingly, as you increase the size of the model, many of these abilities are suddenly acquired rather than gradually improved. Smaller models can't do these tasks at all, they're no better than random, but once the model is large enough, they can do them. This paper studies how a quantitative change in model size leads to a qualitative change in abilities.

### Other Language Models
- `2018` [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) (ULMFiT)
    - From the fastai creator Jeremy Howard, this paper describes several techniques for improved transfer learning in NLP, including (1) gradually unfreezing the layers, (2) having exponentially smaller learning rates for earlier layers, and (3) having the learning rate schedule set so that the learning rate quickly increase linearly, then slowly decrease linearly (giving a slanted triangular shape) as training progresses. This paper also describes a 2-stage pretraining process. This technique was developed by Jeremy when putting together his fastai course and it was SOTA at the time.


## Reinforcement Learning
- `2015` [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    - Agent plays Atari at a superhuman level.
- `2022` [Mastering the Game of Stratego with Model-Free Multi-agent Reinforcement Learning](https://arxiv.org/abs/2206.15378)
    - Stratego is a board game that involves long term global strategy and bluffing, and is much more difficult for computers than Chess, Go or even Poker, which made it one of the last games holding out against AI.
- `2022` [A Generalist Agent](https://arxiv.org/abs/2205.06175)
    - This paper presents a single model (with one set of weights) that can perform various tasks across different domains such chat, image captioning, playing Atari games, and controlling a physical robot arm to stack blocks.


## Auto Encoders
- `2020` [Auto Encoders](https://arxiv.org/abs/2003.05991)
    - This paper gives an overview of auto encoders and lists many interesting applications for them.


## GANs
- `2014` [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
    - The initial GANs paper. The idea is that you train a Generator model which takes random noise as input and outputs synthetic data, and a Discriminator model which classifies data as either real or synthetic. These two models are iteratively trained and both get gradually better.


## General Deep Learning
- `2015` [Deep Learning](https://www.researchgate.net/publication/277411157_Deep_Learning)
    - High-level overview of early deep learning up to LSTM.


## Normalization
- `2015` [Batch Normalization](https://arxiv.org/abs/1502.03167)
	- Batch Normalization (Batch Norm) is a widely adopted technique in deep learning still frequently used today. At each Batch Norm layer, data within each mini-batch is normalized, this leads to faster training and less care required when initializing weights. It also acts as a regularizer.
- `2016` [Layer Normalization](https://arxiv.org/abs/1607.06450)
    - Layer normalization (LN) is an alternative to batch normalization (BN), where you normalize activations across the layer, rather than across the mini-batch. Like BN, LN prevents exploding/vanishing gradients, and enables much faster training of deep neural nets without the need to carefully initialize the weights. LN has become very popular in the past few years and has surpassed BN in popularity in recent ML papers by a significant margin. One advantage to LN over BN is that data samples within a mini-batch are independent and therefore parallelizable. This allows you to fine tune large models without an expensive high-memory GPU by splitting the mini-batch into multiple "sub-mini-batches" to compute the mini-batch gradient piece-by-piece. LN is also more easily applied to RNNs than BN when different data points have different input sizes.


## Stochastic Regularization
- `2014` [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    - Dropout is a fundamental deep learning regularization technique that is widely used today. This paper was first introduced as a Master's thesis and was rejected at NIPS, yet it ended up being one of the most important papers in deep learning with over 40k citations.


## Optimization

### Gradient-Based Optimization
- `2014` [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
    - Adam (named from Adaptive Moment Estimation) is a popular gradient-based deep learning optimizer. It's the default optimizer for scikit-learn's MLPClassifier class. Adam has a reputation of being less sensitive to hyperparameter changes than SGD with momentum.
- `2017` [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW)
    - This paper discusses AdamW, an enhancement to the Adam optimizer, which has been showing up in a lot of recent ML papers. The terms "weight decay" and "L2 penalty" are often used interchangeably, and while this is correct for basic SGD, it's not correct for adaptive learning rate optimization algorithms like Adam (which uses an L2 penalty in most implementations). AdamW modifies Adam to use weight decay. This is advantageous because (1) it improves performance, and (2) the hyperparameters for learning rate and L2 penalty are highly dependent on each other, whereas the hyperparameters for learning rate and weight decay are mostly decoupled, which leads to hyperparameters being tuned more easily.

### Bayesian Optimization
- `2011` [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
    - This paper gives a good explanation of the two most popular Bayesian Hyperparameter tuning algorithms. (1) Gaussian Processes (GP), and (2) Tree-structured Parzen Estimators (TPE). TPE tends to outperform GP.
 - `2020` [A new automatic machine learning based hyperparameter](https://journals.sagepub.com/doi/pdf/10.1177/0020294020932347)
    - This paper is not good. There are several typos and incorrect equations (missing brackets, etc.), some of the diagrams aren't quite right, and the equations for their main algorithm don't seem to do what they say it does. Tree Parzen Estimators (TPE) is a good Bayesian hyperparameter tuning algorithm. The HyperOpt package implements Adaptive Tree Parzen Estimators (ATPE), which is an enhancement to TPE. I was looking for a paper on ATPE and found this one, which references the Hyperopt package and calls their algorithm "Adaptive Tree Parzen Estimators (ATPE)", but it doesn't seem to be the *real* algorithm. The ATPE algorithm in the HyperOpt codebase doesn't seem to resemble what's in this paper at all. On the plus side, the paper does give a solid review of existing literature.
- `2022` [A Comparative study of Hyper-Parameter Optimization Tools](https://arxiv.org/abs/2201.06433)
    - Bayesian hyperparameter tuning algorithms are an underutilized tool to improve ML models. This paper compares various packages that implement these algorithms and seems to favour Optuna.


