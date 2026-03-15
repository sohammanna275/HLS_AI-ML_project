#include "weights.h"
// dense 1 layer 
void compute_dense_1(const float input[400], float output[64]){
    #pragma HLS loop_flatten
    for(int j = 0; j < 64; j++){ // columns
        output[j] = dense1_biases[j]; // output with bias for each neuron
        #pragma HLS pipeline
        for(int i = 0; i < 400; i++){ // rows
            // since it is a 1d array for c++
            // the 1D index formula is: (row * num_columns) + col
            // Multiply input by weights and accumulate
            // row = i, col = j, num_columns = 64
            output[j] += input[i] * dense1_weights[(i * 64) + j];
        }
        // ReLU function
        if(output[j] < 0.0f){
            output[j] = 0.0f;
        }
    }
}
// dense 2 layer
void compute_dense_2(const float input[64], float output[10]){
    #pragma HLS pipeline
    for(int j = 0; j < 10; j++){ // columns
        output[j] = output_biases[j];
        // Dot pdt
        for(int i = 0; i < 64; i++){ // rows
            output[j] += input[i] * output_weights[(i * 10) + j];
        }
    }
}

// convolution layer 1 28x28x1 input -> 26x26x8 output
// Height (3) x Width (3) x Input Channels (1) x Output Filters (8) = 72 total weights
void compute_conv2d_1(const float input[28][28][1], float output[26][26][8]){
    // Looping over every pixel in the OUTPUT image (26x26)
    #pragma HLS loop_flatten
    for(int h = 0; h < 26; h++){
        #pragma HLS loop_flatten
        for(int w = 0; w < 26; w++){
            // Looping over every filter in the convolutional layer
            #pragma HLS pipeline
            for(int c_out = 0; c_out < 8; c_out++){
                // setting the output pixel value to the bias for that filter
                output[h][w][c_out] = conv1_biases[c_out];
                // Looping over the 3x3 filter and the single input channel
                for(int c_in = 0; c_in < 1; c_in++){
                    for(int kh = 0; kh < 3; kh++){// row
                        for(int kw = 0; kw < 3; kw++){// col
                            // pixel needed at input
                            float pixel = input[h + kh][w + kw][c_in];
                            // weight needed for the convolution
                            // Keras stores the weights for a Conv2D layer in a 4D tensor with this exact shape: [Kernel_Height][Kernel_Width][Input_Channels][Output_Channels]
                            // float weight = conv1_weights[(kh * 3 * 1 * 8) + (kw * 1 * 8) + (c_in * 8) + c_out];
                            float weight = conv1_weights[(kh * 24) + (kw * 8) + (c_in * 8) + c_out];
                            // Multiply and accumulate
                            output[h][w][c_out] += pixel * weight;
                        }
                    }
                }
                // ReLU Activation Function (just like the dense layer)
                if(output[h][w][c_out] < 0.0f){
                    output[h][w][c_out] = 0.0f;
                }
            }
        }
    }
}
// MaxPool Layer 1: 26x26x8 input -> 13x13x8 output
void compute_maxpool_1(const float input[26][26][8], float output[13][13][8]){
    // Window: 2x2, Stride: 2
    #pragma HLS loop_flatten
    for(int h = 0; h < 13; h++){
        #pragma HLS loop_flatten
        for(int w = 0; w < 13; w++){
            #pragma HLS pipeline
            for(int c = 0; c < 8; c++){
                int start_h = h * 2;
                int start_w = w * 2;
                float max_val = input[start_h][start_w][c];
                // for maxpooling - Comparing it against the other 3 pixels in that 2x2 window
                if(input[start_h + 1][start_w][c] > max_val){
                    max_val = input[start_h + 1][start_w][c];
                }
                if(input[start_h][start_w + 1][c] > max_val){
                    max_val = input[start_h][start_w + 1][c];
                }
                if(input[start_h + 1][start_w + 1][c] > max_val){
                    max_val = input[start_h + 1][start_w + 1][c];
                }
                output[h][w][c] = max_val;
            }
        }
    }
}
// Conv2D Layer 2: 13x13x8 input -> 11x11x16 output
// Weights shape: [3][3][8][16] = 1152 elements
void compute_conv2d_2(const float input[13][13][8], float output[11][11][16]){
    #pragma HLS loop_flatten
    for(int h = 0; h < 11; h++){
        #pragma HLS loop_flatten
        for(int w = 0; w < 11; w++){
            #pragma HLS loop_flatten
            for(int c_out = 0; c_out < 16; c_out++){// Output Channels (16 filters)
                output[h][w][c_out] = conv2_biases[c_out]; // bias for each filter
                #pragma HLS loop_flatten
                for(int c_in = 0; c_in < 8; c_in++){
                    #pragma HLS loop_flatten
                    for(int kh = 0; kh < 3; kh++){
                        #pragma HLS pipeline
                        for(int kw = 0; kw < 3; kw++){
                            float pixel = input[h + kh][w + kw][c_in];
                            // weight index = (kh * 3 * 8 * 16) + (kw * 8 * 16) + (c_in * 16) + c_out
                            float weight = conv2_weights[(kh * 384) + (kw * 128) + (c_in * 16) + c_out];
                            output[h][w][c_out] += pixel * weight;
                        }
                    }
                }
                // ReLU Activation
                if(output[h][w][c_out] < 0.0f){
                    output[h][w][c_out] = 0.0f;
                }
            }
        }
    }
}

// MaxPool Layer 2: 11x11x16 input -> 5x5x16 output
void compute_maxpool_2(const float input[11][11][16], float output[5][5][16]){
    #pragma HLS loop_flatten
    for(int h = 0; h < 5; h++){
        #pragma HLS loop_flatten
        for(int w = 0; w < 5; w++){
            #pragma HLS pipeline
            for(int c = 0; c < 16; c++){
                int start_h = h * 2;
                int start_w = w * 2;
                float max_val = input[start_h][start_w][c];
                if(input[start_h + 1][start_w][c] > max_val){
                    max_val = input[start_h + 1][start_w][c];
                }
                if(input[start_h][start_w + 1][c] > max_val){
                    max_val = input[start_h][start_w + 1][c];
                }
                if(input[start_h + 1][start_w + 1][c] > max_val){
                    max_val = input[start_h + 1][start_w + 1][c];
                }
                output[h][w][c] = max_val;
            }
        }
    }
}

void mnist_cnn(const float input_image[28][28][1], int &prediction){
    // 1. Declare the intermediate memory buffers
    float conv1_out[26][26][8] = {0};
    float pool1_out[13][13][8] = {0};
    float conv2_out[11][11][16] = {0};
    float pool2_out[5][5][16] = {0};
    float flat_out[400] = {0};
    float dense1_out[64] = {0};
    float dense2_out[10] = {0};
    // 2. Run the network layers sequentially
    compute_conv2d_1(input_image, conv1_out);
    compute_maxpool_1(conv1_out, pool1_out);
    compute_conv2d_2(pool1_out, conv2_out);
    compute_maxpool_2(conv2_out, pool2_out);
    // 3. Flatten the 5x5x16 pool2 output into the 400-element 1D array
    int flat_idx = 0;
    #pragma HLS loop_flatten
    for (int h = 0; h < 5; h++) {
        #pragma HLS loop_flatten
        for (int w = 0; w < 5; w++) {
            #pragma HLS pipeline
            for (int c = 0; c < 16; c++) {
                flat_out[flat_idx] = pool2_out[h][w][c];
                flat_idx++;
            }
        }
    }
    // 4. Run the Dense Layers
    compute_dense_1(flat_out, dense1_out);
    compute_dense_2(dense1_out, dense2_out);
    // 5. Argmax: Find the index of the highest logit
    float max_val = dense2_out[0];
    prediction = 0;
    #pragma HLS pipeline
    for (int i = 1; i < 10; i++) {
        if (dense2_out[i] > max_val) {
            max_val = dense2_out[i];
            prediction = i; // The index IS the predicted digit
        }
    }
}
