#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

//#include <iostream>
//#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class NeuralNetwork {
public:
    double learning_rate;

    struct Layer {
        MatrixXd weights;
        VectorXd biases;
        int input_size;
        int output_size;
    };

    vector<int> layer_sizes;
    vector<Layer> network_layers;

    NeuralNetwork(const vector<int>& layer_sizes);

    vector<VectorXd> forward(const VectorXd &input);
    vector<MatrixXd> forwardBatch(const MatrixXd &inputBatch);
    void backpropagateBatch(const MatrixXd &targetBatch, const vector<MatrixXd> &activations);
    void trainBatch(const MatrixXd &inputBatch, const MatrixXd &targetBatch);
    int predict(const VectorXd &input);
    double calculate_accuracy(const vector<VectorXd> &inputs, const vector<VectorXd> &targets);
};

#endif // NEURALNETWORK_H
