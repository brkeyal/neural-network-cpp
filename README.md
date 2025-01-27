
# NeuralNetwork-CPP

A neural network implementation in C++ for MNIST digit classification using Eigen.

## Project Structure
```plaintext
neural-network-cpp/
├── include/
│   └── NeuralNetwork.h
├── src/
│   ├── NeuralNetwork.cpp
│   ├── main.cpp
├── data/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── CMakeLists.txt
```

## Dependencies
- Eigen library

## Setup
1. Clone repository
2. Download MNIST dataset files to data/
3. Install Eigen: `brew install eigen` (macOS) or `sudo apt-get install libeigen3-dev` (Linux)

## Build
```bash
g++ -std=c++17 -I/opt/homebrew/include/eigen3 -o NeuralNetwork src/main.cpp
```

Or with CMake:
```cmake
cmake_minimum_required(VERSION 3.15)
project(NeuralNetwork-CPP)
set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(NeuralNetwork src/main.cpp)
```

## Run
```bash
./NeuralNetwork
```

## Features
- Forward/backward propagation
- Gradient descent
- SIMD optimization
- Custom network architecture
- Training statistics output

## License
MIT License
```
