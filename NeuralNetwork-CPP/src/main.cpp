#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include "NeuralNetwork.h"

using namespace std;
using namespace Eigen;

// -------------------------
// MNIST Data Loading Functions (unchanged)
// -------------------------

// MNIST data loading functions
std::vector<VectorXd> load_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the magic number and number of images
    int32_t magic_number = 0;
    int32_t number_of_images = 0;
    int32_t rows = 0;
    int32_t cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Convert from big-endian to host byte order
    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic_number != 2051) {  // Expected magic number for image files
        throw std::runtime_error("Invalid MNIST image file: " + filename);
    }

    std::vector<VectorXd> images;
    const int image_size = rows * cols;

    for (int i = 0; i < number_of_images; ++i) {
        std::vector<unsigned char> buffer(image_size);
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);

        VectorXd image(image_size);
        for (int j = 0; j < image_size; ++j) {
            image[j] = static_cast<double>(buffer[j]) / 255.0;  // Normalize to [0, 1]
        }

        images.push_back(image);
    }

    file.close();
    return images;
}

std::vector<VectorXd> load_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the magic number and the number of items
    int32_t magic_number = 0;
    int32_t number_of_items = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));

    // Convert from big-endian to host byte order
    magic_number = __builtin_bswap32(magic_number);
    number_of_items = __builtin_bswap32(number_of_items);

    if (magic_number != 2049) {  // Expected magic number for label files
        throw std::runtime_error("Invalid MNIST label file: " + filename);
    }

    std::vector<VectorXd> labels;

    // Read the labels and one-hot encode them
    for (int i = 0; i < number_of_items; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));

        // Create a one-hot encoded vector for the label
        VectorXd one_hot = VectorXd::Zero(10);  // 10 classes for digits 0-9
        one_hot[label] = 1.0;

        labels.push_back(one_hot);
    }

    file.close();
    return labels;
}



// ===========================
// Main Function
// ===========================
int main() {
    try {
        cout << "Loading MNIST data..." << endl;
        vector<VectorXd> train_images = load_mnist_images("train-images-idx3-ubyte");
        vector<VectorXd> train_labels = load_mnist_labels("train-labels-idx1-ubyte");
        vector<VectorXd> test_images = load_mnist_images("t10k-images-idx3-ubyte");
        vector<VectorXd> test_labels = load_mnist_labels("t10k-labels-idx1-ubyte");

        cout << "Creating neural network..." << endl;
        NeuralNetwork nn({784, 128, 64, 10});

        // Training parameters.
        const int epochs = 5;
        const int batch_size = 64;
        random_device rd;
        mt19937 gen(rd());

        cout << "Starting training..." << endl;
        auto training_start_time = chrono::steady_clock::now();

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle the training data indices.
            vector<int> indices(train_images.size());
            iota(indices.begin(), indices.end(), 0); // Fill the vector with sequential numbers (0, 1, 2, ... , N-1)
            shuffle(indices.begin(), indices.end(), gen);

            // Process mini–batches using the shuffled indices.
            for (size_t i = 0; i < indices.size(); i += batch_size) {
                
                // Ensures batches contain at most batch_size elements. Handles last batch correctly when fewer than batch_size elements remain.
                int current_batch_size = min(batch_size, static_cast<int>(indices.size() - i));
                
                // Eigen::MatrixXd mat(rows, cols);
                // If images are 28×28 pixels (flattened to 784 features), and current_batch_size = 64, the matrix size will be: (784 × 64)
                // Each column represents one image (a sample).
                // Each row represents a specific pixel (feature) across all images.
                MatrixXd batch_inputs(train_images[0].size(), current_batch_size);
                MatrixXd batch_targets(train_labels[0].size(), current_batch_size);

                for (int j = 0; j < current_batch_size; j++) {
                    int idx = indices[i + j];
                    batch_inputs.col(j) = train_images[idx];
                    batch_targets.col(j) = train_labels[idx];
                }
                nn.trainBatch(batch_inputs, batch_targets);
            }
            cout << "Epoch " << (epoch + 1) << " complete." << endl;
        }

        auto training_end_time = chrono::steady_clock::now();
        auto duration_seconds = chrono::duration_cast<chrono::seconds>(training_end_time - training_start_time);

        double accuracy = nn.calculate_accuracy(test_images, test_labels);
//        double accuracy = nn.calculate_accuracy(train_images, train_labels);

        cout << "\nTraining completed. Final Results:" << endl;
        cout << "> Test Accuracy: " << accuracy << "%" << endl;
        cout << "> Total training time: "
             << duration_seconds.count() / 60 << " minutes and "
             << duration_seconds.count() % 60 << " seconds" << endl;
    }
    catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}

