//
//  NeuralNetwork.h
//  MyCPPProject
//
//  Created by Eyal Barak on 27/01/2025.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#include <Eigen/Dense>

using namespace Eigen;

class NeuralNetwork {
private:
    std::vector<int> layer_sizes;

    struct Layer {
        MatrixXd weights;
        VectorXd biases;
        int input_size;
        int output_size;
    };

    std::vector<Layer> network_layers; // Hidden layers (not including output layer)
    std::vector<VectorXd> layers_activations; // Pre-allocated activations for each layer

public:
    double learning_rate = 0.1;

    NeuralNetwork(const std::vector<int>& layer_sizes);
//    void start_training_loop(auto train_images, auto train_labels, auto test_images, auto test_labels, int epochs, int batch_size);
    void start_training_loop(const std::vector<Eigen::VectorXd>& train_images,
                             const std::vector<Eigen::VectorXd>& train_labels,
                             const std::vector<Eigen::VectorXd>& test_images,
                             const std::vector<Eigen::VectorXd>& test_labels,
                             int epochs,
                             int batch_size,
                             int start_idx,
                             int end_idx);
    
    void forward(const VectorXd& input);
    void backpropagate(const VectorXd& target);
    void train(const VectorXd& input, const VectorXd& target);
    int predict(const VectorXd& input);
    VectorXd predict_probabilities(const VectorXd &input);
    double calculate_accuracy(const std::vector<VectorXd>& inputs, const std::vector<VectorXd>& targets);

};

#endif /* NeuralNetwork_h */
