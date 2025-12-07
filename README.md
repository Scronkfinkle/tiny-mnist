# Tiny MNIST

A ~300 line from-scratch* neural network implementation in Rust for MNIST digit classification, inspired by the 3Blue1Brown series on neural networks.

## Overview

This project implements a multi-layer perceptron (MLP) neural network from the ground up, without using any ML frameworks. It implements the core concepts of:
- Forward propagation
- Backpropagation with gradient descent
- Cross-entropy loss with softmax activation
- Weight initialization and numerical stability techniques  

I made it because after watching some videos on how neural networks function at a low level, I wanted to apply it to see what it actually takes to build something practical.

## Pre-requisites
You should watch these videos to help explain everything that's going on here:  

3Blue1Brown Neural Network Series:
* [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
* [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
* [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8)

Artom Kirsanov's Cross-Entropy explanation:
* [The Key Equation Behind Probability](https://www.youtube.com/watch?v=KHVR587oW8I)

(Optional) Andrej Karpathy's Micrograd tutorial:
* [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
## Quick Start
This code is not very efficient, and `cargo run` will be very slow. It's recommended you build and run this with
```bash
cargo run --release
```

It takes about 4-5 minutes on my machine to train but you should eventually see something like this:  
If it fails with NaN (or if you want to try and get better), you can try tuning the parameters/architecture in `main.rs`
```bash
# Model Performance (Training Data)
# Correct: 7701  Incorrect: 52299 Accuracy: 12.84%
# Model Performance (Test Data)
# Correct: 1238  Incorrect: 8762 Accuracy: 12.38%
Initial Loss 2.3021650570112273
Training...
(Epoch 1/10) Loss: 2.006211042872563
(Epoch 2/10) Loss: 1.6804801431140421
(Epoch 3/10) Loss: 1.5403268631741156
(Epoch 4/10) Loss: 1.4416650354726195
(Epoch 5/10) Loss: 1.421127595082744
(Epoch 6/10) Loss: 1.4114806765025723
(Epoch 7/10) Loss: 1.4064681376955193
(Epoch 8/10) Loss: 1.4039222909552094
(Epoch 9/10) Loss: 1.403353387461272
(Epoch 10/10) Loss: 1.4032988632194991
# Model Performance (Training Data)
# Correct: 56579  Incorrect: 3421 Accuracy: 94.30%
# Model Performance (Test Data)
# Correct: 9436  Incorrect: 564 Accuracy: 94.36%
```

## Dataset
The MNIST dataset files are included in this repository to make it easy to get started. The original dataset docs can be found at [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/), but downloading and extracting can be tricky (currently the links are 403 forbidden), so the binary files are vendored in this repositority as:

- `data/train-images-idx3-ubyte` - Training images (60,000 samples)
- `data/train-labels-idx1-ubyte` - Training labels
- `data/t10k-images-idx3-ubyte` - Test images (10,000 samples)  
- `data/t10k-labels-idx1-ubyte` - Test labels

## Implementation Details

- **Forward Pass**: Matrix multiplication with activation functions (`src/mlp.rs:73-98`)
- **Backward Pass**: Chain rule application for gradient computation (`src/mlp.rs:100-201`)
- **Weight Updates**: Gradient descent with learning rate and clipping (`src/mlp.rs:176`)
- **Data Processing**: Pixel normalization and one-hot encoding (`src/mnist.rs`)

## Additional Math Resources

This implementation uses several mathematical concepts beyond what's covered in the 3Blue1Brown videos. Each is documented with links in the code:

- **Softmax Function**: Used for converting raw outputs to probability distributions (`src/functions.rs:5-14`) - [GeeksforGeeks explanation](https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/)
- **Cross-Entropy Loss Derivative**: The mathematical shortcut for softmax + cross-entropy (`src/functions.rs:55-63`) - [GeeksforGeeks derivation](https://www.geeksforgeeks.org/machine-learning/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss/)
- **Leaky ReLU**: Prevents dead neuron problem (`src/functions.rs:65-75`) - [GeeksforGeeks explanation](https://www.geeksforgeeks.org/machine-learning/Leaky-Relu-Activation-Function-in-Deep-Learning/)
- **He Weight Initialization**: Proper weight initialization for ReLU-like activations (`src/utils.rs:7-22`) - [Medium article](https://medium.com/@sanjay_dutta/understanding-glorot-and-he-initialization-a-guide-for-college-students-00f3dfae0393)
- **Hadamard Product**: Element-wise matrix multiplication (`src/utils.rs:24-38`) - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
- **Matrix Math Intuition**: How neural networks map to matrix operations (`src/mlp.rs:82-87`) - [Giles Thomas explanation](https://www.gilesthomas.com/2025/02/basic-neural-network-matrix-maths-part-1)

## Numerical Stability (NUM_STABLE markers)

In the real world, we need to handle the fact that we're doing floating point arithmetic and can very easily break things. You'll see `NUM_STABLE` comments marking places where numerical stability techniques were needed to prevent NaN errors:

- **Weight Initialization** (`src/utils.rs:10-18`): Uses He initialization with additional 0.1 scaling to prevent weights from becoming too large
- **Gradient Clipping** (`src/mlp.rs:160-172`): Caps gradient magnitudes to prevent exploding gradients
- **Log Epsilon Addition** (`src/functions.rs:44-48`): Adds small epsilon (1e-12) before taking logarithm to avoid ln(0)
- **Leaky ReLU** (`src/functions.rs:66-75`): Prevents completely dead neurons that would stop learning
- **Raw Logits Output** (`src/mlp.rs:32-40`): Uses raw final layer outputs instead of activated values for numerical stability in loss calculation

These modifications were essential to get the network to train successfully without hitting numerical instabilities that would cause the training to fail with NaN values.

>*Excluding one dependency to make matrices first-class citizens in the language, and another for generating random numbers