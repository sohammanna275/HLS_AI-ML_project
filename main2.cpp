#include <iostream>

// 1. Include the stb_image library
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Forward declaration of your IP block
void mnist_cnn(const float input_image[28][28][1], int &prediction);

int main() {
    int width, height, channels;
    
    // 2. Load the image! 
    // The "1" at the end forces it to load as grayscale (1 channel)
    const char* filename = "my_digit2.png";
    unsigned char *img_data = stbi_load(filename, &width, &height, &channels, 1);

    if (img_data == NULL) {
        std::cout << "Error: Could not load image " << filename << std::endl;
        return -1;
    }

    if (width != 28 || height != 28) {
        std::cout << "Error: Image must be exactly 28x28 pixels! Yours is " 
                  << width << "x" << height << std::endl;
        stbi_image_free(img_data);
        return -1;
    }

    // 3. Prepare our 3D float array for the neural network
    float test_image[28][28][1] = {0};

    // 4. Map the 1D image data into our 3D array and normalize it
    for (int h = 0; h < 28; h++) {
        for (int w = 0; w < 28; w++) {
            // stb_image loads data as a flat 1D array of bytes (0 to 255)
            int pixel_index = (h * 28) + w;
            
            // Neural networks like small numbers, so we divide by 255.0
            // to convert the range from [0, 255] to [0.0, 1.0]
            test_image[h][w][0] = img_data[pixel_index] / 255.0f;
        }
    }

    // Free the memory used by the image loader
    stbi_image_free(img_data);

    std::cout << "--- MNIST HLS C-Simulation ---" << std::endl;
    std::cout << "Loaded " << filename << " successfully." << std::endl;
    std::cout << "Running Inference..." << std::endl;

    // 5. Run the hardware function!
    int prediction = -1;
    mnist_cnn(test_image, prediction);

    std::cout << "\n===================================" << std::endl;
    std::cout << "   NETWORK PREDICTION: " << prediction << std::endl;
    std::cout << "===================================\n" << std::endl;

    return 0;
}