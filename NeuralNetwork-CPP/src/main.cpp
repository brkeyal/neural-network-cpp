#include <iostream>
#include <cassert>
#include <fstream>
#include <Eigen/Dense>
using namespace Eigen;
#include <thread>

#include "NeuralNetwork.h"

double ensemble_accuracy(std::vector<NeuralNetwork>& models, const std::vector<VectorXd>& inputs, const std::vector<VectorXd>& targets);

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

int main() {
    try {
        // Load MNIST data
        std::cout << "Loading MNIST data..." << std::endl;
        auto train_images = load_mnist_images("train-images-idx3-ubyte");
        auto train_labels = load_mnist_labels("train-labels-idx1-ubyte");
        auto test_images = load_mnist_images("t10k-images-idx3-ubyte");
        auto test_labels = load_mnist_labels("t10k-labels-idx1-ubyte");

        // Create neural network
        std::cout << "Creating neural network..." << std::endl;
//        NeuralNetwork nn({784, 128, 64, 10});  // Example architecture

        std::vector<std::thread> threads;
        int num_threads = 3;
        int chunk_size = train_images.size() / num_threads;
         
        std::vector<NeuralNetwork> models;
        for (int i = 0; i < num_threads; i++) {
            std::cout << "\nCreating neural network #" << i << std::endl;
            models.emplace_back(std::vector<int>{784, 128, 64, 10});
        }
        
        // Training parameters
        int epochs = 5;
        int batch_size = 64;

        std::cout << "Total images: " << train_images.size() << ", Batch size: " << batch_size
                  << ", Total batches per epoch: " << train_images.size() / batch_size
                  << ", Total epochs: " << epochs << std::endl;

        auto training_start_time = std::chrono::steady_clock::now();

        for (int t = 0; t < num_threads; t++) {
            int start_idx = t * chunk_size;
            int end_idx = (t == num_threads - 1) ? train_images.size() : (t + 1) * chunk_size;
            
            threads.emplace_back([&models, t, &train_images, &train_labels, start_idx, end_idx, batch_size, epochs, test_images, test_labels]() {
                
                models[t].start_training_loop(train_images, train_labels, test_images, test_labels, epochs, batch_size, start_idx, end_idx);
                
            });
        }
        
        // Wait for threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        std::cout << "\nAll " << num_threads << " threads completed." << std::endl;

        // Calculate final accuracy
        auto training_end_time = std::chrono::steady_clock::now();
        auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(training_end_time - training_start_time);

        double final_ensembled_accuracy = ensemble_accuracy(models, train_images, train_labels);
        std::cout << "\nTraining completed. Final Results:"
                  << "\n> [ensemble] Test Accuracy: " << final_ensembled_accuracy << "%"
                    <<"\n> Total training time: " << duration_seconds.count() / 60 << " minutes and "
                    << duration_seconds.count() % 60 << " seconds" << std::endl;

        

//        assert(final_ensembled_accuracy > 95);
        
//        for (int i = 0; i < num_threads; i++) {
//            double final_test_accuracy = models[i].calculate_accuracy(test_images, test_labels);
//            std::cout << "\nTraining completed. Final Results:"
//                      << "\n> [model] Test Accuracy: " << final_test_accuracy << "%"
//                      << "\n> Total training time: " << duration_seconds.count() / 60 << " minutes and "
//                      << duration_seconds.count() % 60 << " seconds" << std::endl;
//    
//        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
