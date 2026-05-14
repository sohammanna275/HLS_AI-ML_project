#include "resnet20.h"
#include "resnet_weights.h"
#include <ap_fixed.h>
#include <hls_math.h>


template<int CH_IN, int CH_OUT, int SIZE> void conv3x3(data_t input[CH_IN][SIZE][SIZE], const weight_t weights[CH_OUT * CH_IN * 3 * 3], data_t output[CH_OUT][SIZE][SIZE]){
    for(int co = 0; co < CH_OUT; co++){
        for(int h = 0; h < SIZE; h++){
            for(int w = 0; w < SIZE; w++){
                data_t sum = 0;
                for(int ci = 0; ci < CH_IN; ci++){
                    for(int kh = 0; kh < 3; kh++){
                        for(int kw = 0; kw < 3; kw++){
                            int h_in = h + kh - 1;
                            int w_in = w + kw - 1;
                            if(h_in >= 0 && h_in < SIZE && w_in >= 0 && w_in < SIZE){
                                // 4d [Output_Ch][Input_Ch][Kernel_H][Kernel_W] to 1d index for weights: co * (CH_IN * 3 * 3) + ci * (3 * 3) + kh * 3 + kw
                                int weight_idx = co * (CH_IN * 3 * 3) + ci * (3 * 3) + (kh) * 3 + (kw);
                                sum += input[ci][h_in][w_in] * weights[weight_idx];
                            }
                        }
                    }
                }
                output[co][h][w] = sum;
            }
        }
    }
}
// batch normalization: output = (input - mean) * gamma / sqrt(variance + epsilon) + beta
template <int CHANNELS, int SIZE> void batch_norm(data_t input[CHANNELS][SIZE][SIZE], const weight_t gamma[CHANNELS], const weight_t beta[CHANNELS], const weight_t mean[CHANNELS], const weight_t variance[CHANNELS], data_t output[CHANNELS][SIZE][SIZE]){
    data_t epsilon = 1e-5;
    for(int c = 0; c < CHANNELS; c++){
        data_t var_eps = variance[c] + epsilon;
        data_t stddev = hls::sqrt(var_eps);
        data_t scale = gamma[c] / stddev;
        data_t shift = beta[c] - (mean[c] * scale);
        for(int h = 0; h < SIZE; h++){
            for(int w = 0; w < SIZE; w++){
                output[c][h][w] = input[c][h][w] * scale + shift;
            }
        }
    }
}

// ReLU activation f(x)=max(0,x)
template <int CHANNELS, int SIZE> void relu(data_t input[CHANNELS][SIZE][SIZE], data_t output[CHANNELS][SIZE][SIZE]){
    for(int c = 0; c < CHANNELS; c++){
        for(int h = 0; h < SIZE; h++){
            for(int w = 0; w < SIZE; w++){
                output[c][h][w] = (input[c][h][w] > 0) ? input[c][h][w] : (data_t)0;
            }
        }
    }
}

// residual block with 2 conv layers and a skip connection
template<int CHANNELS, int SIZE> void residual_block_basic(data_t input[CHANNELS][SIZE][SIZE], const weight_t conv1_w[], const weight_t bn1_g[], const weight_t bn1_b[], const weight_t bn1_m[], const weight_t bn1_v[], const weight_t conv2_w[], const weight_t bn2_g[], const weight_t bn2_b[], const weight_t bn2_m[], const weight_t bn2_var[], data_t output[CHANNELS][SIZE][SIZE]){
    data_t conv1_out[CHANNELS][SIZE][SIZE];
    data_t bn1_out[CHANNELS][SIZE][SIZE];
    data_t relu1_out[CHANNELS][SIZE][SIZE];
    data_t conv2_out[CHANNELS][SIZE][SIZE];
    data_t bn2_out[CHANNELS][SIZE][SIZE];
    // First Convolution layer -> Batch Norm -> ReLU
    conv3x3<CHANNELS, CHANNELS, SIZE> (input, conv1_w, conv1_out);
    batch_norm<CHANNELS, SIZE> (conv1_out, bn1_g, bn1_b, bn1_m, bn1_v, bn1_out);
    relu<CHANNELS, SIZE> (bn1_out, relu1_out);
    // Second Convolution layer -> Batch Norm
    conv3x3<CHANNELS, CHANNELS, SIZE> (relu1_out, conv2_w, conv2_out);
    batch_norm<CHANNELS, SIZE> (conv2_out, bn2_g, bn2_b, bn2_m, bn2_var, bn2_out);
    // Now adding the original input to the output of the second batch norm and then ReLU
    for(int c = 0; c < CHANNELS; c++){
        for(int h = 0; h < SIZE; h++){
            for(int w = 0; w < SIZE; w++){
                data_t sum = input[c][h][w] + bn2_out[c][h][w];
                output[c][h][w] = (sum > 0) ? sum : (data_t)0;
            }
        }
    }
}

// 3x3 convolution with stride = 2
// weights[output_channel][input_channel][kernel_height][kernel_width]
template<int CH_IN, int CH_OUT, int SIZE_IN> void conv3x3_stride2(data_t input[CH_IN][SIZE_IN][SIZE_IN], const weight_t weights[CH_OUT * CH_IN * 9], data_t output[CH_OUT][SIZE_IN/2][SIZE_IN/2]){
    const int SIZE_OUT = SIZE_IN/2;
    for(int co = 0; co < CH_OUT; co++){
        for(int h = 0; h < SIZE_OUT; h++){
            for(int w = 0; w < SIZE_OUT; w++){
                data_t sum = 0;
                for(int ci = 0; ci < CH_IN; ci++){
                    for(int kh = 0; kh < 3; kh++){
                        for(int kw = 0; kw < 3; kw++){
                            int h_in = h * 2 + kh - 1;
                            int w_in = w * 2 + kw - 1;
                            if(h_in >= 0 && h_in < SIZE_IN && w_in >= 0 && w_in < SIZE_IN){
                                // weights[co][ci][kh][kw]
                                int weight_idx = co * (CH_IN * 9) + ci * 9 + kh * 3 + kw;
                                sum += input[ci][h_in][w_in] * weights[weight_idx];
                            }
                        }
                    }
                }
                output[co][h][w] = sum;
            }
        }
    }
}
// 1x1 convolution with stride = 2 for the shortcut path
// The shortcut path uses a 1x1 convolution with a stride of 2 to match the new shape
template<int CH_IN, int CH_OUT, int SIZE_IN> void conv1x1_stride2(data_t input[CH_IN][SIZE_IN][SIZE_IN], const weight_t weights[CH_OUT * CH_IN], data_t output[CH_OUT][SIZE_IN/2][SIZE_IN/2]){
    const int SIZE_OUT = SIZE_IN/2;
    for(int co = 0; co < CH_OUT; co++){
        for(int h = 0; h < SIZE_OUT; h++){
            for(int w = 0; w < SIZE_OUT; w++){
                data_t sum = 0;
                for(int ci = 0; ci < CH_IN; ci++){
                    int weight_idx = co * CH_IN + ci;
                    // Stride jump without the 3x3 loop overhead
                    sum += input[ci][h * 2][w * 2] * weights[weight_idx];
                }
                output[co][h][w] = sum;
            }
        }
    }
}

// residual block for downsampling
template<int CH_IN, int CH_OUT, int SIZE_IN> void residual_block_downsample(data_t input[CH_IN][SIZE_IN][SIZE_IN], const weight_t conv1_w[], const weight_t bn1_g[], const weight_t bn1_b[], const weight_t bn1_m[], const weight_t bn1_v[], const weight_t conv2_w[], const weight_t bn2_g[], const weight_t bn2_b[], const weight_t bn2_m[], const weight_t bn2_v[], const weight_t skip_w[], data_t output[CH_OUT][SIZE_IN/2][SIZE_IN/2]){
    const int SIZE_OUT = SIZE_IN/2;
    data_t conv1_out[CH_OUT][SIZE_OUT][SIZE_OUT];
    data_t bn1_out[CH_OUT][SIZE_OUT][SIZE_OUT];
    data_t relu1_out[CH_OUT][SIZE_OUT][SIZE_OUT];
    data_t conv2_out[CH_OUT][SIZE_OUT][SIZE_OUT];
    data_t bn2_out[CH_OUT][SIZE_OUT][SIZE_OUT];
    data_t skip_out[CH_OUT][SIZE_OUT][SIZE_OUT];
    // main path
    conv3x3_stride2<CH_IN, CH_OUT, SIZE_IN> (input, conv1_w, conv1_out);
    batch_norm<CH_OUT, SIZE_OUT> (conv1_out, bn1_g, bn1_b, bn1_m, bn1_v, bn1_out);
    relu<CH_OUT,SIZE_OUT> (bn1_out, relu1_out);
    // the second conv is a standard stride-1 convolution on the smaller size
    conv3x3<CH_OUT, CH_OUT, SIZE_OUT> (relu1_out, conv2_w, conv2_out);
    batch_norm<CH_OUT, SIZE_OUT> (conv2_out, bn2_g, bn2_b, bn2_m, bn2_v, bn2_out);
    // shortcut connection
    conv1x1_stride2<CH_IN, CH_OUT, SIZE_IN>(input, skip_w, skip_out);
    // addition and relu
    for(int c = 0; c < CH_OUT; c++) {
        for(int h = 0; h < SIZE_OUT; h++) {
            for(int w = 0; w < SIZE_OUT; w++) {
                data_t sum = bn2_out[c][h][w] + skip_out[c][h][w];
                output[c][h][w] = (sum > 0) ? sum : (data_t)0;
            }
        }
    }
}

// Global Average Pooling
template<int CHANNELS, int SIZE> void global_average_pooling(data_t input[CHANNELS][SIZE][SIZE], data_t output[CHANNELS]){
    // averaging an 8x8 grid so we divide by 64
    for(int c = 0; c < CHANNELS; c++){
        data_t sum = 0;
        for(int h = 0; h < SIZE; h++){
            for(int w = 0; w < SIZE; w++){
                sum += input[c][h][w];
            }
        }
        output[c] = sum / (SIZE * SIZE);
    }
}

// Dense Layer
template <int IN_FEATURES, int OUT_FEATURES> void dense_layer(data_t input[IN_FEATURES], const weight_t weights[IN_FEATURES * OUT_FEATURES], const weight_t biases[OUT_FEATURES], data_t output[OUT_FEATURES]){
    for(int o = 0; o < OUT_FEATURES; o++){
        data_t sum = biases[o];
        for(int i = 0; i < IN_FEATURES; i++){
            // weights[i][o]
            int weight_idx = i * OUT_FEATURES + o;
            sum += input[i] * weights[weight_idx];
        }
        output[o] = sum;
    }
}

// Top-Level ResNet-20 Function
void resnet20_top(data_t input_image[IMAGE_CHANNELS][IMAGE_SIZE][IMAGE_SIZE], data_t predictions[NUM_CLASSES]){
    // 32x32 spatial resolution buffers
    data_t fm_16_32_A[16][32][32];
    data_t fm_16_32_B[16][32][32];

    // 16x16 spatial resolution buffers
    data_t fm_32_16_A[32][16][16];
    data_t fm_32_16_B[32][16][16];
    
    // 8x8 spatial resolution buffers
    data_t fm_64_8_A[64][8][8];
    data_t fm_64_8_B[64][8][8];
    
    // Flattened outputs
    data_t pool_out[64];

    // Initial Layer
    conv3x3<3, 16, 32>(input_image, layer_1_weights, fm_16_32_A);
    batch_norm<16, 32>(fm_16_32_A, layer_2_gamma, layer_2_beta, layer_2_mean, layer_2_variance, fm_16_32_B);
    relu<16, 32>(fm_16_32_B, fm_16_32_A);

    // Stack 1: 16 channels, 32x32 (3 Basic Blocks)
    residual_block_basic<16, 32>(fm_16_32_A, layer_4_weights, layer_5_gamma, layer_5_beta, layer_5_mean, layer_5_variance, layer_7_weights, layer_8_gamma, layer_8_beta, layer_8_mean, layer_8_variance, fm_16_32_B);
    residual_block_basic<16, 32>(fm_16_32_B, layer_11_weights, layer_12_gamma, layer_12_beta, layer_12_mean, layer_12_variance, layer_14_weights, layer_15_gamma, layer_15_beta, layer_15_mean, layer_15_variance, fm_16_32_A);
    residual_block_basic<16, 32>(fm_16_32_A, layer_18_weights, layer_19_gamma, layer_19_beta, layer_19_mean, layer_19_variance, layer_21_weights, layer_22_gamma, layer_22_beta, layer_22_mean, layer_22_variance, fm_16_32_B);

    // Stack 2: 32 channels, 16x16 (1 Downsample, 2 Basic)
    
    // Block 1: Downsample
    residual_block_downsample<16, 32, 32>(fm_16_32_B, layer_25_weights, layer_26_gamma, layer_26_beta, layer_26_mean, layer_26_variance, layer_28_weights, layer_30_gamma, layer_30_beta, layer_30_mean, layer_30_variance, layer_29_weights, fm_32_16_A);
    
    // Block 2: Basic (Using the layer arrays you just sent: 33, 34, 36, 37)
    residual_block_basic<32, 16>(fm_32_16_A, layer_33_weights, layer_34_gamma, layer_34_beta, layer_34_mean, layer_34_variance, layer_36_weights, layer_37_gamma, layer_37_beta, layer_37_mean, layer_37_variance, fm_32_16_B);
    
    // Block 3: Basic (Using the layer arrays you just sent: 40, 41, 43, 44)
    residual_block_basic<32, 16>(fm_32_16_B, layer_40_weights, layer_41_gamma, layer_41_beta, layer_41_mean, layer_41_variance, layer_43_weights, layer_44_gamma, layer_44_beta, layer_44_mean, layer_44_variance, fm_32_16_A);

    // Stack 3: 64 channels, 8x8 (1 Downsample, 2 Basic)
    // Block 1: Downsample (Using the arrays you sent: 47, 48, 50, 52... skip is 51)
    residual_block_downsample<32, 64, 16>(fm_32_16_A, layer_47_weights, layer_48_gamma, layer_48_beta, layer_48_mean, layer_48_variance, layer_50_weights, layer_52_gamma, layer_52_beta, layer_52_mean, layer_52_variance, layer_51_weights, fm_64_8_B);
    
    // Block 2: Basic (Continuing the Keras numbering pattern)
    residual_block_basic<64, 8>(fm_64_8_B, layer_55_weights, layer_56_gamma, layer_56_beta, layer_56_mean, layer_56_variance, layer_58_weights, layer_59_gamma, layer_59_beta, layer_59_mean, layer_59_variance, fm_64_8_A);
    
    // Block 3: Basic 
    residual_block_basic<64, 8>(fm_64_8_A, layer_62_weights, layer_63_gamma, layer_63_beta, layer_63_mean, layer_63_variance, layer_65_weights, layer_66_gamma, layer_66_beta, layer_66_mean, layer_66_variance, fm_64_8_B);

    // Final Layers
    global_average_pooling<64, 8>(fm_64_8_B, pool_out);
    
    // Final Dense Layer (10 classes for CIFAR-10)
    dense_layer<64, 10>(pool_out, layer_70_weights, layer_70_bias, predictions);
}
