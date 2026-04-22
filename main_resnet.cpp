#include <iostream>
#include "resnet20.h"

using namespace std;

int main() {
    data_t test_image[IMAGE_CHANNELS][IMAGE_SIZE][IMAGE_SIZE];
    data_t predictions[NUM_CLASSES];

    cout << "Generating test image data..." << endl;
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
        for (int h = 0; h < IMAGE_SIZE; h++) {
            for (int w = 0; w < IMAGE_SIZE; w++) {
                test_image[c][h][w] = 0.5; 
            }
        }
    }

    cout << "Starting ResNet-20 Inference..." << endl;
    
    resnet20_top(test_image, predictions);
    
    cout << "Inference Complete!\n" << endl;

    cout << "===================================" << endl;
    cout << "        CLASS PREDICTIONS           " << endl;
    cout << "===================================" << endl;
    for (int i = 0; i < NUM_CLASSES; i++) {
        cout << "Class " << i << ": " << (float)predictions[i] << endl;
    }

    return 0; 
}
