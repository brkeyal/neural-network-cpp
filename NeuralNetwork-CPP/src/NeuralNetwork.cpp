//
//  NeuralNetwork.cpp
//  MyCPPProject
//
//  Created by Eyal Barak on 27/01/2025.
//
#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) : layer_sizes(layer_sizes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        Layer layer;
        layer.input_size = layer_sizes[i];
        layer.output_size = layer_sizes[i + 1];

        double scale = std::sqrt(2.0 / (layer.input_size + layer.output_size));
        layer.weights = MatrixXd::Random(layer.input_size, layer.output_size) * scale;
        layer.biases = VectorXd::Random(layer.output_size) * scale;

        network_layers.push_back(layer);
    }

    for (const auto& size : layer_sizes) {
        layers_activations.emplace_back(VectorXd::Zero(size));
    }

    std::cout << "Neural Network initialized with memory pools" << std::endl;
}

void NeuralNetwork::start_training_loop(const std::vector<Eigen::VectorXd>& train_images,
                                        const std::vector<Eigen::VectorXd>& train_labels,
                                        const std::vector<Eigen::VectorXd>& test_images,
                                        const std::vector<Eigen::VectorXd>& test_labels,
                                        int epochs,
                                        int batch_size,
                                        int start_idx,
                                        int end_idx)
{
    // Training loop
    std::cout << "Starting training... train images size: " << end_idx-start_idx << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (size_t i = start_idx; i < end_idx; i++) {
            train(train_images[i], train_labels[i]);

            // Calculate batch loss (optional)
            if ((i + 1) % batch_size == 0) {
                VectorXd output = layers_activations.back();
                
                // Compute the Mean Squared Error (MSE) for the batch:
                // 1. Calculate the difference (output - target).
                // 2. squaredNorm() computes the sum of squared differences (||output - target||^2).
                // 3. Divide by the output size to get the average loss per output neuron.
                double batch_loss = (output - train_labels[i]).squaredNorm() / output.size();
                total_loss += batch_loss;

                auto now = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);
//                std::cout << std::put_time(std::localtime(&time), "[%H:%M:%S] ")
//                          << "Epoch " << epoch + 1
//                          << ", Batch " << (i + 1) / batch_size
//                          << ", Loss: " << batch_loss << std::endl;
            }
        }

        // Print epoch stats
        std::cout << "\nEpoch " << epoch + 1
                  << " completed. Average loss: " << total_loss / (train_images.size() / batch_size) << std::endl;
    }

}

void NeuralNetwork::forward(const VectorXd& input) {
    layers_activations[0] = input;

    for (size_t i = 0; i < network_layers.size(); i++) {
        const Layer& layer = network_layers[i];
        layers_activations[i + 1] = (layer.weights.transpose() * layers_activations[i] + layer.biases)
            .unaryExpr([](double x) {
                return 1.0 / (1.0 + std::exp(-x)); // Sigmoid activation function
            });
    }
}

void NeuralNetwork::backpropagate(const VectorXd& target) {
    std::vector<VectorXd> deltas(layer_sizes.size());

    deltas.back() = (layers_activations.back() - target)
        .cwiseProduct(layers_activations.back().cwiseProduct(
            VectorXd::Ones(layers_activations.back().size()) - layers_activations.back()));

    for (int i = layer_sizes.size() - 2; i >= 0; i--) {
        deltas[i] = (network_layers[i].weights * deltas[i + 1])
            .cwiseProduct(layers_activations[i].cwiseProduct(
                VectorXd::Ones(layers_activations[i].size()) - layers_activations[i]));
    }

    for (size_t i = 0; i < network_layers.size(); i++) {
        Layer& layer = network_layers[i];
        layer.weights -= learning_rate * (layers_activations[i] * deltas[i + 1].transpose());
        layer.biases -= learning_rate * deltas[i + 1];
    }
}

void NeuralNetwork::train(const VectorXd& input, const VectorXd& target) {
    forward(input);
    backpropagate(target);
}

int NeuralNetwork::predict(const VectorXd& input) {
    forward(input);
    const VectorXd& output = layers_activations.back();
    int predicted_class = 0;
    output.maxCoeff(&predicted_class); // Get the index of the largest activation
    return predicted_class;
}

double NeuralNetwork::calculate_accuracy(const std::vector<VectorXd>& inputs,
                                         const std::vector<VectorXd>& targets) {
    int correct = 0;

    for (size_t i = 0; i < inputs.size(); i++) {
        int predicted = predict(inputs[i]);
        int actual = 0;
        targets[i].maxCoeff(&actual);

        if (predicted == actual) {
            correct++;
        }
    }

    return static_cast<double>(correct) / inputs.size() * 100.0;
}

VectorXd NeuralNetwork::predict_probabilities(const VectorXd &input) {
    forward(input);
    return layers_activations.back();
}

static int ensemble_predict(std::vector<NeuralNetwork>& models,
                          const VectorXd &input) {
    std::vector<double> avg_probs(10, 0.0);
    
    for (auto& model : models) {
        auto probs = model.predict_probabilities(input);
        for (size_t i = 0; i < probs.size(); i++) {
            avg_probs[i] += probs[i];
        }
    }

    for (auto& p : avg_probs) {
        p /= models.size();
    }

    return std::max_element(avg_probs.begin(), avg_probs.end()) - avg_probs.begin();
}

double ensemble_accuracy(std::vector<NeuralNetwork>& models,
                                        const std::vector<VectorXd>& inputs,
                                        const std::vector<VectorXd>& targets) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        int predicted = ensemble_predict(models, inputs[i]);
        int actual = std::max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();
        if (predicted == actual) correct++;
    }
    return static_cast<double>(correct) / inputs.size() * 100.0;
}
