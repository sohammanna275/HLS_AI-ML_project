#include <iostream>
#include "resnet20.h"
using namespace std;
int main() {
    // 1. Allocate memory for the input image and the output predictions
    data_t test_image[IMAGE_CHANNELS][IMAGE_SIZE][IMAGE_SIZE];
    data_t predictions[NUM_CLASSES];

    cout << "Generating test image data..." << endl;
    // 2. Create a dummy test image (filling it with 0.5)
    // Later, you can swap this to read a real CIFAR-10 image from a text file!
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
        for (int h = 0; h < IMAGE_SIZE; h++) {
            for (int w = 0; w < IMAGE_SIZE; w++) {
                test_image[c][h][w] = 0.5; 
            }
        }
    }

    cout << "Starting ResNet-20 Inference..." << endl;
    
    // 3. Execute your network!
    resnet20_top(test_image, predictions);
    
    cout << "Inference Complete!\n" << endl;

    // 4. Print the final results
    cout << "===================================" << endl;
    cout << "       CLASS PREDICTIONS           " << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        // We cast to float just so it prints nicely to the console
        cout << "Class " << i << ": " << (float)predictions[i] << endl;
    }

    // Returning 0 tells Vitis HLS the simulation passed without crashing
    return 0; 
}