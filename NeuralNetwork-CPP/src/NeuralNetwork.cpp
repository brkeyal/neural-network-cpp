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
// TODO: unite logics with forwardBatch / remove dup code
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


/**
 * Runs a batch of inputs through the neural network.
 * Each layer:
 * 1. Multiplies inputs by weights.
 * 2. Adds biases.
 * 3. Applies ReLU (hidden layers) or Softmax (output layer).
 *
 * @param inputBatch Matrix of input samples (features × batch_size)
 * @return Activations for each layer (including final output)
 */
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
            // NOTE: ReLU Just Clears Negative Numbers
            // aleternative (same operation but supposed to be faster): a = z.cwiseMax(0.0);
            a = z.unaryExpr([](double x) { return std::max(0.0, x); });
        }
        activations.push_back(a);
    }
    
    // Letting RVO (Return Value Optimization) do its job and avoid copying. For clearer approach we can refactor to get activations by reference, or save it as memeber on the class
    return activations;
}


/**
 * Updates the neural network using backpropagation.
 * Each layer:
 * 1. Computes error (difference between prediction and target).
 * 2. Propagates the error backward through the network.
 * 3. Calculates gradients and updates weights & biases.
 *
 * @param targetBatch Matrix of true labels (one-hot encoded).
 * @param activations Activations from forward propagation.
 */
void NeuralNetwork::backpropagateBatch(const MatrixXd &targetBatch, const vector<MatrixXd> &activations) {
    int L = activations.size();  // Total number of layers (input + hidden(s) + output)
    int batch_size = targetBatch.cols();
    vector<MatrixXd> deltas(L);

    // Output layer delta: for softmax with cross–entropy, δ = output – target.
    // Note: When using Softmax activation combined with Cross-Entropy loss, the gradient simplifies to output – target
    
    /* Algorithm:
     * 1. Take the network's prediction (activations.back() → Softmax output).
     * 2. Subtract the true label (targetBatch) from it.
     * 3. Store this as the error (delta) for the output layer.
     */
    deltas[L - 1] = activations.back() - targetBatch;

    // Backpropagate through hidden layers.
    // computes deltas (δ) - represent the gradient of the loss with respect to activations.
    /* Algorithm:
     * 1. Take the error from the next layer.
     * 2. Multiply by current layer’s weights to distribute the error backward.
     * 3. Apply ReLU derivative to remove gradients from inactive neurons.
     * 4. Store the new delta (error) for this layer.
     */
    for (int l = L - 2; l >= 0; l--) {
        // Derivative of ReLU: 1 if activation > 0, else 0.
        // Note: If a neuron was active (ReLU > 0), its gradient is 1, otherwise 0, This ensures only active neurons contribute to learning.
        MatrixXd relu_deriv = activations[l].unaryExpr([](double x) { return (x > 0) ? 1.0 : 0.0; });
        // Note: network_layers[l] connects activation l to activation l+1.
        // cwiseProduct - Element-wise multiplication (applies a transformation to each value individually).
        deltas[l] = (network_layers[l].weights * deltas[l + 1]).cwiseProduct(relu_deriv);
    }

    // Update the weights and biases.
    /* Algorithm:
     * 1. Compute the weight gradient (gradW)
     *      - Multiply the activation of the previous layer by the error from the next layer.
     *      - Average it over the batch.
     * 2. Compute the bias gradient (gradb)
     *      - Sum up all the errors for this layer (bias is the same for all inputs).
     *      - Average it over the batch.
     * 3. Update the weights
     *      - Subtract learning_rate * gradW to adjust weights opposite to the gradient.
     * 4. Update the biases
     *      - Subtract learning_rate * gradb to adjust biases opposite to the gradient.
     **/
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

double NeuralNetwork::calculate_accuracy_batch(const vector<VectorXd> &inputs, const vector<VectorXd> &targets, int batch_size) {
    if (inputs.size() != targets.size())
        throw runtime_error("Mismatch between number of inputs and targets.");

    int num_samples = inputs.size();
    int correct = 0;

    // Iterate over mini-batches
    for (int start_idx = 0; start_idx < num_samples; start_idx += batch_size) {
        int current_batch_size = min(batch_size, num_samples - start_idx);

        // Create input batch (each column is an input sample)
        MatrixXd inputBatch(inputs[0].size(), current_batch_size);
        for (int i = 0; i < current_batch_size; i++) {
            inputBatch.col(i) = inputs[start_idx + i];
        }

        // Forward pass on the batch
        vector<MatrixXd> activations = forwardBatch(inputBatch);
        MatrixXd outputBatch = activations.back();  // Softmax probabilities

        // Compute predictions for the batch
        VectorXi predictedLabels(current_batch_size);
        for (int i = 0; i < current_batch_size; i++) {
            outputBatch.col(i).maxCoeff(&predictedLabels(i));  // Get index of max probability
        }

        // Compute actual labels for the batch
        VectorXi actualLabels(current_batch_size);
        for (int i = 0; i < current_batch_size; i++) {
            targets[start_idx + i].maxCoeff(&actualLabels(i));  // Get index of correct class
        }

        // Count correct predictions in this batch
        correct += (predictedLabels.array() == actualLabels.array()).count();
    }

    // Return overall accuracy
    return (static_cast<double>(correct) / num_samples) * 100.0;
}

