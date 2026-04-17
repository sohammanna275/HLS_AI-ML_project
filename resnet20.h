#pragma once
#include <ap_fixed.h>
#include "resnet_weights.h" 
//16 bits total, 6 integer bits (matches the weight_t type)
typedef ap_fixed<16, 6> data_t;
const int IMAGE_SIZE = 32;
const int IMAGE_CHANNELS = 3;
const int NUM_CLASSES = 10;
void resnet20_top(data_t input_image[IMAGE_CHANNELS][IMAGE_SIZE][IMAGE_SIZE], data_t predictions[NUM_CLASSES]);