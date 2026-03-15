#include <iostream>

// Forward declaration telling the compiler this function exists in cnn.cpp
void mnist_cnn(const float input_image[28][28][1], int &prediction);

int main() {
    // Create a blank 28x28 image (all black / 0.0f)
    float test_image[28][28][1] = {0};
    // a vertical line down the middle to simulate a handwritten '1'
    for(int i = 5; i < 24; i++) {
        test_image[i][14][0] = 1.0f;  // Solid center line
        test_image[i][13][0] = 0.5f;  // Slightly gray left edge
        test_image[i][15][0] = 0.5f;  // Slightly gray right edge
    }

    std::cout << "--- MNIST HLS C-Simulation ---" << std::endl;
    std::cout << "Input image generated (Simulated '1')." << std::endl;
    std::cout << "Running Inference..." << std::endl;

    // 3. Set up the output variable
    int prediction = -1;

    // 4. Run your top-level hardware function!
    mnist_cnn(test_image, prediction);

    // 5. Print the results
    std::cout << "\n===================================" << std::endl;
    std::cout << "   NETWORK PREDICTION: " << prediction << std::endl;
    std::cout << "===================================\n" << std::endl;

    return 0;
}