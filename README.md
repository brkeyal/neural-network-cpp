# neural-network-cpp
A simple and optimized neural network implementation in C++ for training and testing on the MNIST dataset. This project uses Eigen for efficient matrix and vector operations and implements core concepts like forward propagation, backpropagation, and gradient descent.

## Project Structure
neural-network-cpp/
├── include/
│   └── NeuralNetwork.h       # Header file for the NeuralNetwork class
├── src/
│   ├── NeuralNetwork.cpp     # Implementation of the NeuralNetwork class
│   ├── main.cpp              # Entry point with training and testing logic
├── data/                     # MNIST dataset files (place them here)
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── CMakeLists.txt            # (Optional) CMake configuration
└── README.md                 # Project description and usage guide
