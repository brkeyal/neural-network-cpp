#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class NeuralNetwork {
private:
    double learning_rate;

    struct Layer {
        MatrixXd weights;
        VectorXd biases;
        int input_size;
        int output_size;
    };

    vector<Layer> network_layers;
    vector<int> layer_sizes;

    vector<VectorXd> forward(const VectorXd &input);
    int predict(const VectorXd &input);

    vector<MatrixXd> forwardBatch(const MatrixXd &inputBatch);
    void backpropagateBatch(const MatrixXd &targetBatch, const vector<MatrixXd> &activations);
    
public:
    NeuralNetwork(const vector<int>& layer_sizes);
    
    void trainBatch(const MatrixXd &inputBatch, const MatrixXd &targetBatch);
    double calculate_accuracy(const vector<VectorXd> &inputs, const vector<VectorXd> &targets);
};

#endif // NEURALNETWORK_H
