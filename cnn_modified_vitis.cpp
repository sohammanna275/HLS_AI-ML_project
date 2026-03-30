#include "weights.h"
#include <ap_fixed.h>

// This creates a custom 16-bit hardware integer!
// 16 total bits, 6 bits for the integer part, 10 bits for the fraction.
typedef ap_fixed<16,6> fixed_t;

// dense 1 layer 
void compute_dense_1(const fixed_t input[400], fixed_t output[64]){
    for(int j = 0; j < 64; j++){ // columns
        fixed_t sum = dense1_biases[j]; 
        for(int i = 0; i < 400; i++){ // rows
            #pragma HLS PIPELINE II=1
            // Multiply input by weights and accumulate
            sum += input[i] * dense1_weights[(i * 64) + j];
        }
        // ReLU function
        if(sum < 0){
            output[j] = 0;
        } else {
            output[j] = sum;
        }
    }
}

// dense 2 layer
void compute_dense_2(const fixed_t input[64], fixed_t output[10]){
    for(int j = 0; j < 10; j++){ // columns
        fixed_t sum = output_biases[j];
        // Dot pdt
        for(int i = 0; i < 64; i++){ // rows
            #pragma HLS PIPELINE II=1
            sum += input[i] * output_weights[(i * 10) + j];
        }
        output[j] = sum;
    }
}

// convolution layer 1 28x28x1 input -> 26x26x8 output
void compute_conv2d_1(const fixed_t input[28][28][1], fixed_t output[26][26][8]){
    // Looping over every pixel in the OUTPUT image (26x26)
    for(int h = 0; h < 26; h++){
        for(int w = 0; w < 26; w++){
            #pragma HLS PIPELINE II=1
            
            // Looping over every filter in the convolutional layer
            for(int c_out = 0; c_out < 8; c_out++){
                fixed_t sum = conv1_biases[c_out];
                
                // Looping over the 3x3 filter and the single input channel
                for(int c_in = 0; c_in < 1; c_in++){
                    for(int kh = 0; kh < 3; kh++){// row
                        for(int kw = 0; kw < 3; kw++){// col
                            fixed_t pixel = input[h + kh][w + kw][c_in];
                            fixed_t weight = conv1_weights[(kh * 24) + (kw * 8) + (c_in * 8) + c_out];
                            sum += pixel * weight;
                        }
                    }
                }
                // ReLU Activation Function
                if(sum < 0){
                    output[h][w][c_out] = 0;
                } else {
                    output[h][w][c_out] = sum;
                }
            }
        }
    }
}

// MaxPool Layer 1: 26x26x8 input -> 13x13x8 output
void compute_maxpool_1(const fixed_t input[26][26][8], fixed_t output[13][13][8]){
    // Window: 2x2, Stride: 2
    for(int h = 0; h < 13; h++){
        for(int w = 0; w < 13; w++){
            #pragma HLS PIPELINE II=1
            
            for(int c = 0; c < 8; c++){
                int start_h = h * 2;
                int start_w = w * 2;
                fixed_t max_val = input[start_h][start_w][c];
                
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
void compute_conv2d_2(const fixed_t input[13][13][8], fixed_t output[11][11][16]){
    for(int h = 0; h < 11; h++){
        for(int w = 0; w < 11; w++){
            for(int c_out = 0; c_out < 16; c_out++){// Output Channels (16 filters)
                #pragma HLS PIPELINE II=1

                fixed_t sum = conv2_biases[c_out]; // bias for each filter
                for(int c_in = 0; c_in < 8; c_in++){
                    for(int kh = 0; kh < 3; kh++){
                        for(int kw = 0; kw < 3; kw++){
                            fixed_t pixel = input[h + kh][w + kw][c_in];
                            fixed_t weight = conv2_weights[(kh * 384) + (kw * 128) + (c_in * 16) + c_out];
                            sum += pixel * weight;
                        }
                    }
                }
                // ReLU Activation
                if(sum < 0){
                    output[h][w][c_out] = 0;
                } else {
                    output[h][w][c_out] = sum;
                }
            }
        }
    }
}

// MaxPool Layer 2: 11x11x16 input -> 5x5x16 output
void compute_maxpool_2(const fixed_t input[11][11][16], fixed_t output[5][5][16]){
    for(int h = 0; h < 5; h++){
        for(int w = 0; w < 5; w++){
            #pragma HLS PIPELINE II=1
            
            for(int c = 0; c < 16; c++){
                int start_h = h * 2;
                int start_w = w * 2;
                fixed_t max_val = input[start_h][start_w][c];
                
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
    // 1. Control signals (Start, Done, Idle, Ready)
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // 2. High-speed memory bus for the image (Accepts standard floats from CPU)
    #pragma HLS INTERFACE m_axi port=input_image bundle=gmem0 offset=slave
    #pragma HLS INTERFACE s_axilite port=input_image bundle=CTRL

    // 3. Lightweight bus for the final digit prediction
    #pragma HLS INTERFACE s_axilite port=prediction bundle=CTRL
    
    // --- Hardware Conversion Buffer ---
    fixed_t internal_image[28][28][1];
    for(int h = 0; h < 28; h++){
        for(int w = 0; w < 28; w++){
            #pragma HLS PIPELINE II=1
            internal_image[h][w][0] = (fixed_t)input_image[h][w][0];
        }
    }
    
    // 1. Declare the intermediate memory buffers
    fixed_t conv1_out[26][26][8] = {0};
    fixed_t pool1_out[13][13][8] = {0};
    fixed_t conv2_out[11][11][16] = {0};
    fixed_t pool2_out[5][5][16] = {0};
    fixed_t flat_out[400] = {0};
    fixed_t dense1_out[64] = {0};
    fixed_t dense2_out[10] = {0};
    
    // 2. Run the network layers sequentially
    compute_conv2d_1(internal_image, conv1_out);
    compute_maxpool_1(conv1_out, pool1_out);
    compute_conv2d_2(pool1_out, conv2_out);
    compute_maxpool_2(conv2_out, pool2_out);
    
    // 3. Flatten the 5x5x16 pool2 output into the 400-element 1D array
    for (int h = 0; h < 5; h++) {
        for (int w = 0; w < 5; w++) {
            for (int c = 0; c < 16; c++) {
                #pragma HLS PIPELINE II=1
                // Replaced flat_idx++ with direct math to prevent hardware bottleneck
                flat_out[(h * 80) + (w * 16) + c] = pool2_out[h][w][c];
            }
        }
    }
    
    // 4. Run the Dense Layers
    compute_dense_1(flat_out, dense1_out);
    compute_dense_2(dense1_out, dense2_out);
    
    // 5. Argmax: Find the index of the highest logit
    fixed_t max_val = dense2_out[0];
    prediction = 0;
    for (int i = 1; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        if (dense2_out[i] > max_val) {
            max_val = dense2_out[i];
            prediction = i; 
        }
    }
}
