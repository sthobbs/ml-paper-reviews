
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
    1. [RNN](#rnn)
    2. [Transformer](#transformer)
    3. [LLMs](#llms)
4. [Reinforcement Learning](#reinforcement-learning)
5. [Auto Encoders](#auto-encoders)
6. [GANs](#gans)
7. [General Deep Learning](#general-deep-learning)
8. [Normalization](#normalization)
9. [Activation Functions](#activation-functions)
10. [Data Augmentation](#data-augmentation)
11. [Other Architectural Tools](#other-architectural-tools)
12. [Stochastic Regularization](#stochastic-regularization)
13. [Optimization](#optimization)
    1. [Gradient-Based Optimization](#gradient-based-optimization)
    2. [Bayesian Optimization](#bayesian-optimization)
    3. [Evolutionary Algorithms](#evolutionary-algorithms)
14. [Model Explainability](#model-explainability)
15. [Miscellaneous](#miscellaneous)


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
- `2017` [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) (ResNeXt)
    - This paper outlines the ResNeXt architecture for CNNs, which creates blocks of parallel branches. This approach allows you to scale CNNs more efficiently than increasing the depth or width.
- `2018` [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
    - Elegant CNN architecture with SOTA performance (at the time), where every layer is connected to every other layer (in the same block).
- `2020` [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (EfficientNet)
    - Larger CNNs outperform smaller CNNs, but there are many ways to make a CNN bigger. You can scale the number of layers, number of channels and/or image resolution. The authors propose a novel scaling coefficient that balances these three methods in a systematic way to achieve the best possible trade-off between model size and accuracy.
- `2021` [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ViT)
    - Transformers are the main tool for language models, whereas CNNs are the go-to approach for image models. This paper describes Vision Transformers (ViT), which involves the application of transformers to image classification by splitting each image into a sequence of patches and treating each patch like a word.
- `2022` [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (ConvNeXt)
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

### RNN
- `2018` [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) (ELMo)
    - ELMo uses a bidirectional-LSTM of the surrounding words to generate word embeddings that consider the context of the sentence. This approach was SOTA on several language tasks for a brief period before transformer-based models like BERT surpassed them.

### Transformer
- `2017` [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
	- This paper introduced the Transformer, which was the next major milestone in sequence-to-sequence modeling tasks after RNNs and LSTMs, and is a key building block for many of the large language models we have today. The Transformer uses a fully self-attention-based approach, eliminating the need for recurrent or convolutional layers, and making it more parallelizable and efficient to train. The paper introduces a multi-head attention mechanism that enables the model to focus on multiple parts of the input simultaneously, providing richer information exchange between encoder and decoder.
- `2019` [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (BERT)
	- Another big step forward in language models. BERT learns full-context word representations by using a bidirectional transformer architecture. The model learns in 2 stages: (1) pre-training on unlabeled text data, and (2) fine-tuning on a specific task.
- `2019` [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (ALBERT)
    - An improved version of BERT with 3 enhancements:
        1. Instead of using a large embedding matrix to pass one-hot encoded words directly into the hidden layer, it first passes them into a low-dimensional representation, then passes that into the hidden layer. This reduces the number of parameters.
        2. Parameter sharing across layers to reduce the number of parameters.
        3. An improved loss function involving next-sentence prediction.
- `2022` [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814) (AlphaCode)
    - DeepMind's paper on creating an AI that can compete in programming competitions (at the level of an average competitor). The paper is interesting, but quite long. AlphaCode generates thousands of possible solutions, then filters them by actually running the code locally on the examples from the question, then it submits the most promising solutions.

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


## Activation Functions
- `2013` [Rectifiers Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
    - The authors show that ReLU (Rectified Linear Unit) and LReLU (Leaky ReLU) activations outperform tanh activations on speech recognition tasks.
- `2016` [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    - The paper proposes the ELU (Exponential Linear Unit) activation, f(x) = x if x>0, and f(x) = a (e^x - 1) if x <= 0, and shows that it outperforms classic activation functions like ReLU, LReLU, and SReLU (Shifted ReLU).
- `2017` [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    - The authors argues that the Scaled Exponential Linear Unit (SELU) activation function, f(x) = hx if x>0, and f(x) = ha(e^x - 1) if x <= 0, helps keep activation distributions having 0-mean and 1-variance throughout the network, which helps address the vanishing/exploding gradient problem. The paper also introduces alpha-dropout, a slightly modified version of dropout that sets dropped values to the minimum possible value of -ha, which works better with this activation function.
- `2017` [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    - The paper uses automatic search techniques to discover new activation functions. The best resulting activation function was the swish function f(x) = x * sigmoid(b*x), where b is either a constant (usually 1) or a learned parameter. Swish is extensively evaluated against other activation functions. Notably, both the Swish and GELU papers claim their activation function outperforms the other. Mish, Swish, and GELU seem to be the three best, although the order is unclear.
- `2020` [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) (GELU)
    - GELU is a modern activation function that I'm seeing in a lot of recent deep learning papers, and it seems to work better than a lot of the other common ones. GELU can be viewed as a smoothed version of ReLU. It's defined as f(x) = x * Phi(x), where Phi(x) is the cdf of an N(0,1) random variable. 
- `2020` [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681)
    - Mish is a modern activation function that seems to outperform the commonly used ones. It's shape is very similar to GELU and Swish. Mish is defined as f(x) = x * tanh(softplus(x)), where softplus(x) = ln(1+e^x). The author claims that this form acts as a regularizer by smoothing the gradient.


## Data Augmentation
- `2017` [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](https://arxiv.org/abs/1712.04621)
    - The paper proposes an architecture to jointly train two prepended neural networks as a data augmentation strategy. (1) An augmentation network, which takes 2 images and outputs 1 image combining the two (similar to style transfer), and (2) a classification network which takes the output of the augmentation network as input. At test time, only the classification network is used.

## Other Architectural Tools
- `2013` [Maxout Networks](https://arxiv.org/abs/1302.4389)
    - In a maxout layer, instead of computing one linear combination and applying an activation function, we compute k linear combinations and take the maximum. This can be thought of as automatically learning the activation functions with piecewise-linear convex functions.
- `2015` [Highway Networks](https://arxiv.org/abs/1505.00387)
    - Highway networks use skip connections similar to ResNet, except the data flowing through both the main branch and the skip connection are gated by learned parameters that regulate the flow of information (similar to LSTM).
- `2020` [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (GLUs)
    - Gated Linear Units (GLUs) are neural network layers where you multiply the output of a regular layer (element-wise) by a linear transformation of the input (i.e. f(xW+b) * (xV+c) for some activation function f). This architectural design seems to improve the performance of transformer models.


## Stochastic Regularization
- `2014` [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    - Dropout is a fundamental deep learning regularization technique that is widely used today. This paper was first introduced as a Master's thesis and was rejected at NIPS, yet it ended up being one of the most important papers in deep learning with over 40k citations.
- `2015` [Adding Gradient Noise Improves Learning for Very Deep Networks](https://arxiv.org/abs/1511.06807)
    - The authors improved NN performance (on both the training and test set) by adding a gradually decreasing amount of Gaussian noise to the gradients during training. This may work by helping the model escape local minima and traverse saddle points more quickly.
- `2016` [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
    - Stochastic depth is a regularization technique similar to dropout, but we're dropping layers instead of nodes. At test time, all layers are included, but they're scaled down by their probability of being included in a training iteration (like dropout). This helps with vanishing gradients and forward-flow, and allows very deep NNs to be trained (e.g. 1000+ layers), reduces training time, and improves test error.
- `2017` [Shake-Shake Regularization](https://arxiv.org/abs/1705.07485)
    - For a multi-branch neural networks (e.g. ResNeXt), shake-shake regularization replaces the branch sum with a stochastic linear combination of the branches. We use different coefficients for each forward and backward pass. This helps decorrelate the branches and has a regularization effect.
- `2018` [DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890)
    - Dropout doesn't work too well on CNNs because neighbouring pixels are so strongly correlated that you're not really dropping much information. This paper presents an enhancement to dropout for CNNs, DropBlock, where contiguous rectangular regions are randomly dropped.

## Optimization

### Gradient-Based Optimization
- `2014` [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Adam)
    - Adam (named from Adaptive Moment Estimation) is a popular gradient-based deep learning optimizer. It's the default optimizer for scikit-learn's MLPClassifier class. Adam has a reputation of being less sensitive to hyperparameter changes than SGD with momentum.
- `2017` [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
    - The author describes how using a non-monotonic learning rate schedule (e.g. triangular, sinusoidal, or one that cycles between gradually decreasing upper and lower bounds) can sometimes improve performance. The benefits seem minimal at best.
- `2017` [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW)
    - This paper discusses AdamW, an enhancement to the Adam optimizer, which has been showing up in a lot of recent ML papers. The terms "weight decay" and "L2 penalty" are often used interchangeably, and while this is correct for basic SGD, it's not correct for adaptive learning rate optimization algorithms like Adam (which uses an L2 penalty in most implementations). AdamW modifies Adam to use weight decay. This is advantageous because (1) it improves performance, and (2) the hyperparameters for learning rate and L2 penalty are highly dependent on each other, whereas the hyperparameters for learning rate and weight decay are mostly decoupled, which leads to hyperparameters being tuned more easily.
- `2019` [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) (RAdam)
    - RAdam (Rectified Adam) is an optimization algorithm that improves upon Adam (and AdamW). One limitation of Adam/AdamW is that they can get stuck in poor local minima unless the learning rate starts off very small (e.g. with linear warm up schedule). This paper investigates why this is and concludes that it's caused by the variance of the adaptive moment diverging in the early iterations. The paper goes on to purpose a way to correct this variance with theoretical justification. With RAdam, a linear warmup is no longer required, which results in simpler hyperparameter tuning.

### Bayesian Optimization
- `2011` [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
    - This paper gives a good explanation of the two most popular Bayesian Hyperparameter tuning algorithms. (1) Gaussian Processes (GP), and (2) Tree-structured Parzen Estimators (TPE). TPE tends to outperform GP.
 - `2020` [A new automatic machine learning based hyperparameter](https://journals.sagepub.com/doi/pdf/10.1177/0020294020932347)
    - This paper is not good. There are several typos and incorrect equations (missing brackets, etc.), some of the diagrams aren't quite right, and the equations for their main algorithm don't seem to do what they say it does. Tree Parzen Estimators (TPE) is a good Bayesian hyperparameter tuning algorithm. The HyperOpt package implements Adaptive Tree Parzen Estimators (ATPE), which is an enhancement to TPE. I was looking for a paper on ATPE and found this one, which references the Hyperopt package and calls their algorithm "Adaptive Tree Parzen Estimators (ATPE)", but it doesn't seem to be the *real* algorithm. The ATPE algorithm in the HyperOpt codebase doesn't seem to resemble what's in this paper at all. On the plus side, the paper does give a solid review of existing literature.
- `2022` [A Comparative study of Hyper-Parameter Optimization Tools](https://arxiv.org/abs/2201.06433)
    - Bayesian hyperparameter tuning algorithms are an underutilized tool to improve ML models. This paper compares various packages that implement these algorithms and seems to favour Optuna.

### Evolutionary Algorithms
- `1995` [Particle Swarm Optimization](https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)
    - Particle Swarm Optimization (PSO) is an evolutionary algorithm for optimizing functions, where you initialize a bunch of "particles" at random points, then iterate the following steps: evaluate the function at each particle location, then for each particle, takes a random-sized step in the direction of the previous best location (i.e. best function value) for that particle, and another random step in the direction of the best location among all particles. The step sizes can be large and can overshoot the previous best locations.
    - In modern machine learning, evolutionary algorithms only have a minor role because gradient-based algorithms like SGD are far superior at optimizing neural networks. For hyperparameter tuning (where gradients don't exist, so SGD doesn't work), Bayesian methods tend to outperform evolutionary ones, but evolutionary methods are often used as part of the Bayesian optimization. Sequential Model-Based Optimization (SMBO), which includes Bayesian hyperparameter optimization as a special case, works as follows in general. We have a "true function" that we're trying to optimize (e.g. hyperparameters -> the validation error from the model after training) which is slow to compute, a "surrogate function" which approximates the true function and is fast to compute, and an "acquisition function" (e.g. hyperparameters -> expected improvement of the surrogate function over the previous best value, or hyperparameters -> probability of improvement of the surrogate function over the previous best value) which is optimized to determine the next hyperparameters to evaluate the true function on (which balances exploitation vs exploration). An evolutionary algorithm is often used to optimize this acquisition function, however it's often an evolutionary algorithm other than PSO, such as Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).


## Model Explainability
- `2017` [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (SHAP)
    - This paper introduces SHAP (SHapely Additive exPlanations) and how they can be used for feature importance. Shapely values come from cooperative game theory, where you have a coalition that creates a value and you want to fairly allocate credit to each member of the coalition (e.g. employees of a company generating a profit). In the ML context, we are trying to assign credit to features to determine how valuable they are to the model performance. The true shapely values are impractically computationally intense for modestly-sized datasets, so the paper provides methods to approximate them.

## Miscellaneous
- `2018` [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
    - On newer GPUs, we can use half-precision arithmetic that accumulates into single-precision output for faster training and lower memory usage, while maintaining similar accuracy.
