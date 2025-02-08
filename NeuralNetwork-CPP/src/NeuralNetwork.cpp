#include "NeuralNetwork.h"
//#include <algorithm>
//#include <chrono>
//#include <numeric>
//#include <stdexcept>
#include <random>

using namespace std;
using namespace Eigen;

// ===========================
// NeuralNetwork Class
// ===========================
NeuralNetwork::NeuralNetwork(const vector<int>& layer_sizes)
    : learning_rate(0.1), layer_sizes(layer_sizes) {

    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        Layer layer;
        layer.input_size = layer_sizes[i];
        layer.output_size = layer_sizes[i + 1];
        
        // He initialization (good for ReLU)
        double scale = sqrt(2.0 / layer.input_size);
        layer.weights = MatrixXd::Random(layer.input_size, layer.output_size) * scale;
        
        // Biases can be initialized to zero.
        layer.biases = VectorXd::Zero(layer.output_size);
        
        network_layers.push_back(layer);
    }
}

// ---------------------------
// Single–Sample Forward Pass
// ---------------------------
// Returns a vector where each element is the activation for a layer.
vector<VectorXd> NeuralNetwork::forward(const VectorXd &input) {
    vector<VectorXd> activations;
    activations.push_back(input);

    for (size_t i = 0; i < network_layers.size(); i++) {
        const Layer &layer = network_layers[i];
        VectorXd z = (layer.weights.transpose() * activations[i]) + layer.biases;
        VectorXd a;
        if (i == network_layers.size() - 1) {
            // Use softmax for the output layer.
            double maxVal = z.maxCoeff(); // For numerical stability
            VectorXd expz = (z.array() - maxVal).exp();
            a = expz / expz.sum();
        } else {
            // Use ReLU for hidden layers.
            a = z.unaryExpr([](double x) { return std::max(0.0, x); });
        }
        activations.push_back(a);
    }
    return activations;
}


// ---------------------------
// Batch Forward Pass
// ---------------------------
// Input: inputBatch of size (input_dim, batch_size)
// Returns: A vector of matrices (one per layer) where each column is one sample.
vector<MatrixXd> NeuralNetwork::forwardBatch(const MatrixXd &inputBatch) {
    vector<MatrixXd> activations;
    activations.push_back(inputBatch);

    for (size_t i = 0; i < network_layers.size(); i++) {
        const Layer &layer = network_layers[i];
        // Compute z = (weights^T * activation) + biases (added column–wise)
        MatrixXd z = (layer.weights.transpose() * activations[i]).colwise() + layer.biases;
        MatrixXd a;
        if (i == network_layers.size() - 1) {
            // Softmax for output layer.
            a = MatrixXd(z.rows(), z.cols());
            for (int col = 0; col < z.cols(); col++) {
                VectorXd z_col = z.col(col);
                double maxVal = z_col.maxCoeff();
                VectorXd expz = (z_col.array() - maxVal).exp();
                a.col(col) = expz / expz.sum();
            }
        } else {
            // ReLU for hidden layers.
            a = z.unaryExpr([](double x) { return std::max(0.0, x); });
        }
        activations.push_back(a);
    }
    return activations;
}


// ---------------------------
// Batch Backpropagation
// ---------------------------
// Uses cross–entropy loss with softmax output.
void NeuralNetwork::backpropagateBatch(const MatrixXd &targetBatch, const vector<MatrixXd> &activations) {
    int L = activations.size();  // Total number of layers (input + hidden(s) + output)
    int batch_size = targetBatch.cols();
    vector<MatrixXd> deltas(L);

    // Output layer delta: for softmax with cross–entropy, δ = output – target.
    deltas[L - 1] = activations.back() - targetBatch;

    // Backpropagate through hidden layers.
    for (int l = L - 2; l >= 0; l--) {
        // Derivative of ReLU: 1 if activation > 0, else 0.
        MatrixXd relu_deriv = activations[l].unaryExpr([](double x) { return (x > 0) ? 1.0 : 0.0; });
        // Note: network_layers[l] connects activation l to activation l+1.
        deltas[l] = (network_layers[l].weights * deltas[l + 1]).cwiseProduct(relu_deriv);
    }

    // Update the weights and biases.
    for (size_t i = 0; i < network_layers.size(); i++) {
        MatrixXd gradW = (activations[i] * deltas[i + 1].transpose()) / static_cast<double>(batch_size);
        VectorXd gradb = deltas[i + 1].rowwise().sum() / static_cast<double>(batch_size);
        network_layers[i].weights -= learning_rate * gradW;
        network_layers[i].biases -= learning_rate * gradb;
    }
}

// ---------------------------
// Single Batch Training Step
// ---------------------------
void NeuralNetwork::trainBatch(const MatrixXd &inputBatch, const MatrixXd &targetBatch) {
    vector<MatrixXd> activations = forwardBatch(inputBatch);
    backpropagateBatch(targetBatch, activations);
}

// ---------------------------
// Prediction (Single Sample)
// ---------------------------
int NeuralNetwork::predict(const VectorXd &input) {
    vector<VectorXd> activations = forward(input);
    const VectorXd &output = activations.back();
    int predicted_class = 0;
    output.maxCoeff(&predicted_class);
    return predicted_class;
}

// ---------------------------
// Accuracy Calculation
// ---------------------------
double NeuralNetwork::calculate_accuracy(const vector<VectorXd> &inputs, const vector<VectorXd> &targets) {
    if (inputs.size() != targets.size())
        throw runtime_error("Mismatch between number of inputs and targets.");

    int correct = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        int pred = predict(inputs[i]);
        int actual = 0;
        targets[i].maxCoeff(&actual);
        if (pred == actual)
            correct++;
    }
    return (static_cast<double>(correct) / inputs.size()) * 100.0;
}
