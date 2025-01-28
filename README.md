# NeuralNetwork-CPP

A simple neural network implementation in C++ for training and testing on the MNIST dataset. This project uses Eigen for efficient matrix and vector operations and implements core concepts like forward propagation, backpropagation, and gradient descent.

## Project Structure
```plaintext
neural-network-cpp/
├── NeuralNetwork-CPP/
│   ├── include/
│   │   └── NeuralNetwork.h       # Header file for the NeuralNetwork class
│   ├── src/
│   │   ├── NeuralNetwork.cpp     # Implementation of the NeuralNetwork class
│   │   └── main.cpp              # Entry point with training and testing logic
│   └── data/                     # MNIST dataset files (place them here)
│       ├── (TO DOWNLOAD) train-images-idx3-ubyte
│       ├── (TO DOWNLOAD)  train-labels-idx1-ubyte
│       ├── t10k-images-idx3-ubyte
│       └── t10k-labels-idx1-ubyte
└── README.md                     # Project description and usage guide
```

## Features
- Implements a fully connected neural network for MNIST digit classification.
- Key components:
  - Forward Propagation
  - Backpropagation 
  - Gradient Descent
- Supports customizable architectures with dynamic layer sizes.
- Outputs training statistics (loss, accuracy, runtime).

## Dependencies
- Eigen: A C++ template library for linear algebra.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neural-network-cpp.git
cd neural-network-cpp
```

### 2. Download the MNIST Dataset
- Download the files from MNIST Official Website:
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`
- Place these files in the `data/` folder.

### 3. Run the project in Xcode (NeuralNetwork-CPP.xcodeproj)

## Code Overview

### 1. Main Program
Located in `main.cpp`:
- Loads MNIST images and labels
- Initializes neural network with customizable architecture
- Trains the network using `start_training_loop`
- Calculates training and test accuracy

### 2. Neural Network Class
Located in `NeuralNetwork.h` and `NeuralNetwork.cpp`:
- Implements:
  - `forward`: Computes activations for each layer
  - `backpropagate`: Computes gradients for weights and biases
  - `train`: Trains on a single sample
  - `start_training_loop`: Handles training loop
  - `predict`: Makes predictions
  - `calculate_accuracy`: Evaluates performance

## Example Output
```plaintext
Loading MNIST data...
Creating neural network...
Total images: 60000, Batch size: 64, Total batches per epoch: 937, Total epochs: 5, Learning rate: 0.1
Starting training... train_images.size()=60000

[00:00:01] Epoch 1, Batch 10, Loss: 0.122
[00:00:02] Epoch 1, Batch 20, Loss: 0.098
...

Training completed. Final Results:
> Training Accuracy: 98.5%
> Test Accuracy: 97.3%
> Total training time: 12 minutes and 34 seconds
```

## Notes
- Optimized for MNIST but extendable to other datasets
- Ensure Eigen is installed and included

## To-Do
- Add support for saving/loading trained models
- Extend functionality for other datasets
- Implement additional activation functions

## License
MIT License
```
