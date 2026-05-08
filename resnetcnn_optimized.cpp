
#include "resnet20.h"
#include "resnet_weights.h"
#include <ap_fixed.h>
#include <hls_math.h>

static const data_t EPS = 0.001;


// Fused Batch-Norm + ReLU

template<int CH, int SZ>
void bn_relu(
    data_t         fm [CH][SZ][SZ],
    const weight_t  g [CH], const weight_t b[CH],
    const weight_t  m [CH], const weight_t v[CH])
{
    #pragma HLS INLINE
    for (int c = 0; c < CH; c++) {
        ap_fixed<32,16> scale_w = (ap_fixed<32,16>)g[c] / hls::sqrt((ap_fixed<32,16>)v[c] + (ap_fixed<32,16>)EPS);
        ap_fixed<32,16> shift_w = (ap_fixed<32,16>)b[c] - (ap_fixed<32,16>)m[c] * scale_w;
        const data_t scale = (data_t)scale_w;
        const data_t shift = (data_t)shift_w;
        for (int h = 0; h < SZ; h++) {
            for (int w = 0; w < SZ; w++) {
                #pragma HLS PIPELINE II=1
                const data_t val = fm[c][h][w] * scale + shift;
                fm[c][h][w] = (val > (data_t)0) ? val : (data_t)0;
            }
        }
    }
}


// Fused Batch Norm only (no ReLU)

template<int CH, int SZ>
void bn_only(
    data_t         fm [CH][SZ][SZ],
    const weight_t  g [CH], const weight_t b[CH],
    const weight_t  m [CH], const weight_t v[CH])
{
    #pragma HLS INLINE
    for (int c = 0; c < CH; c++) {
        ap_fixed<32,16> scale_w = (ap_fixed<32,16>)g[c] / hls::sqrt((ap_fixed<32,16>)v[c] + (ap_fixed<32,16>)EPS);
        ap_fixed<32,16> shift_w = (ap_fixed<32,16>)b[c] - (ap_fixed<32,16>)m[c] * scale_w;
        const data_t scale = (data_t)scale_w;
        const data_t shift = (data_t)shift_w;
        for (int h = 0; h < SZ; h++) {
            for (int w = 0; w < SZ; w++) {
                #pragma HLS PIPELINE II=1
                fm[c][h][w] = fm[c][h][w] * scale + shift;
            }
        }
    }
}


// 3×3 Convolution, same-padding, stride=1

template<int CH_IN, int CH_OUT, int SIZE>
void conv3x3(
    data_t         in [CH_IN ][SIZE][SIZE],
    const weight_t  W [CH_OUT * CH_IN * 9],
    data_t         out[CH_OUT][SIZE][SIZE])
{
    #pragma HLS ARRAY_PARTITION variable=in  cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=in  cyclic factor=3 dim=2  
    #pragma HLS ARRAY_PARTITION variable=in  cyclic factor=3 dim=3  
    #pragma HLS ARRAY_PARTITION variable=out cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=W   cyclic factor=4 dim=1  

    for (int co = 0; co < CH_OUT; co++) {
        for (int h = 0; h < SIZE; h++) {
            for (int w = 0; w < SIZE; w++) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 16> acc = 0;
                for (int ci = 0; ci < CH_IN; ci++) {
                    #pragma HLS UNROLL factor=4

                    const int base = (co * CH_IN + ci) * 9;

                    const data_t p00 = (h>0      && w>0     ) ? in[ci][h-1][w-1] : (data_t)0;
                    const data_t p01 = (h>0                 ) ? in[ci][h-1][w  ] : (data_t)0;
                    const data_t p02 = (h>0      && w<SIZE-1) ? in[ci][h-1][w+1] : (data_t)0;
                    const data_t p10 = (            w>0     ) ? in[ci][h  ][w-1] : (data_t)0;
                    const data_t p11 =                          in[ci][h  ][w  ];
                    const data_t p12 = (            w<SIZE-1) ? in[ci][h  ][w+1] : (data_t)0;
                    const data_t p20 = (h<SIZE-1 && w>0     ) ? in[ci][h+1][w-1] : (data_t)0;
                    const data_t p21 = (h<SIZE-1            ) ? in[ci][h+1][w  ] : (data_t)0;
                    const data_t p22 = (h<SIZE-1 && w<SIZE-1) ? in[ci][h+1][w+1] : (data_t)0;

                    acc += p00*W[base+0] + p01*W[base+1] + p02*W[base+2]
                         + p10*W[base+3] + p11*W[base+4] + p12*W[base+5]
                         + p20*W[base+6] + p21*W[base+7] + p22*W[base+8];
                }
                out[co][h][w] = (data_t)acc;
            }
        }
    }
}


// 3×3 Convolution, same-padding, stride=2

template<int CH_IN, int CH_OUT, int SIZE_IN>
void conv3x3_s2(
    data_t         in [CH_IN ][SIZE_IN  ][SIZE_IN  ],
    const weight_t  W [CH_OUT * CH_IN * 9],
    data_t         out[CH_OUT][SIZE_IN/2][SIZE_IN/2])
{
    const int SZ = SIZE_IN / 2;
    #pragma HLS ARRAY_PARTITION variable=in  cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=in  cyclic factor=3 dim=2  
    #pragma HLS ARRAY_PARTITION variable=in  cyclic factor=3 dim=3  
    #pragma HLS ARRAY_PARTITION variable=out cyclic factor=4 dim=1 
    #pragma HLS ARRAY_PARTITION variable=W   cyclic factor=4 dim=1 

    for (int co = 0; co < CH_OUT; co++) {
        for (int h = 0; h < SZ; h++) {
            for (int w = 0; w < SZ; w++) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 16> acc = 0;
                for (int ci = 0; ci < CH_IN; ci++) {
                    #pragma HLS UNROLL factor=4

                    const int base = (co * CH_IN + ci) * 9; 
                    const int hi = h * 2;
                    const int wi = w * 2;

                    const data_t p00 = (hi>0         && wi>0        ) ? in[ci][hi-1][wi-1] : (data_t)0;
                    const data_t p01 = (hi>0                        ) ? in[ci][hi-1][wi  ] : (data_t)0;
                    const data_t p02 = (hi>0         && wi<SIZE_IN-1) ? in[ci][hi-1][wi+1] : (data_t)0;
                    const data_t p10 = (                wi>0        ) ? in[ci][hi  ][wi-1] : (data_t)0;
                    const data_t p11 =                                  in[ci][hi  ][wi  ];
                    const data_t p12 = (                wi<SIZE_IN-1) ? in[ci][hi  ][wi+1] : (data_t)0;
                    const data_t p20 = (hi<SIZE_IN-1 && wi>0        ) ? in[ci][hi+1][wi-1] : (data_t)0;
                    const data_t p21 = (hi<SIZE_IN-1                ) ? in[ci][hi+1][wi  ] : (data_t)0;
                    const data_t p22 = (hi<SIZE_IN-1 && wi<SIZE_IN-1) ? in[ci][hi+1][wi+1] : (data_t)0;

                    acc += p00*W[base+0] + p01*W[base+1] + p02*W[base+2]
                         + p10*W[base+3] + p11*W[base+4] + p12*W[base+5]
                         + p20*W[base+6] + p21*W[base+7] + p22*W[base+8];
                }
                out[co][h][w] = (data_t)acc;
            }
        }
    }
}

// 1×1 Convolution, stride=2  (shortcut projection)

template<int CH_IN, int CH_OUT, int SIZE_IN>
void conv1x1_s2(
    data_t         in [CH_IN ][SIZE_IN  ][SIZE_IN  ],
    const weight_t  W [CH_OUT * CH_IN],
    data_t         out[CH_OUT][SIZE_IN/2][SIZE_IN/2])
{
    const int SZ = SIZE_IN / 2;
    #pragma HLS ARRAY_PARTITION variable=in  cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=out cyclic factor=4 dim=1 
    #pragma HLS ARRAY_PARTITION variable=W   cyclic factor=4 dim=1  

    for (int co = 0; co < CH_OUT; co++) {
        for (int h = 0; h < SZ; h++) {
            for (int w = 0; w < SZ; w++) {
                #pragma HLS PIPELINE II=1
                data_t acc=0;
                for (int ci = 0; ci < CH_IN; ci++) {
                    #pragma HLS UNROLL factor=4
                    acc += in[ci][h*2][w*2] * W[co*CH_IN + ci];
                }
                out[co][h][w] = acc;
            }
        }
    }
}
// Residual Block — same spatial dimensions

template<int CH, int SZ>
void res_basic(
    data_t x  [CH][SZ][SZ],
    const weight_t c1w[],
    const weight_t b1g[], const weight_t b1b[], const weight_t b1m[], const weight_t b1v[],
    const weight_t c2w[],
    const weight_t b2g[], const weight_t b2b[], const weight_t b2m[], const weight_t b2v[],
    data_t out[CH][SZ][SZ])
{
    #pragma HLS INLINE off   

    data_t tmp[CH][SZ][SZ];  

    conv3x3<CH, CH, SZ>(x,   c1w, tmp);
    bn_relu<CH, SZ>    (tmp, b1g, b1b, b1m, b1v);

    conv3x3<CH, CH, SZ>(tmp, c2w, out);
    bn_only<CH, SZ>    (out, b2g, b2b, b2m, b2v);

    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < SZ; h++) {
            for (int w = 0; w < SZ; w++) {
                #pragma HLS PIPELINE II=1
                const data_t s = out[c][h][w] + x[c][h][w];
                out[c][h][w] = (s > (data_t)0) ? s : (data_t)0;
            }
        }
    }
}
// Residual Block — with spatial downsampling

template<int CH_IN, int CH_OUT, int SIZE_IN>
void res_down(
    data_t x  [CH_IN ][SIZE_IN  ][SIZE_IN  ],
    const weight_t c1w[],
    const weight_t b1g[], const weight_t b1b[], const weight_t b1m[], const weight_t b1v[],
    const weight_t c2w[],
    const weight_t b2g[], const weight_t b2b[], const weight_t b2m[], const weight_t b2v[],
    const weight_t skw[],
    data_t out[CH_OUT][SIZE_IN/2][SIZE_IN/2])
{
    #pragma HLS INLINE off   
    #pragma HLS ARRAY_PARTITION variable=skw cyclic factor=4 dim=1 

    const int SZ = SIZE_IN / 2;
    data_t tmp [CH_OUT][SZ][SZ];   
    data_t skip[CH_OUT][SZ][SZ];

    conv3x3_s2<CH_IN, CH_OUT, SIZE_IN>(x,   c1w, tmp);
    bn_relu<CH_OUT, SZ>                (tmp, b1g, b1b, b1m, b1v);  

    conv3x3<CH_OUT, CH_OUT, SZ>        (tmp, c2w, out);
    bn_only<CH_OUT, SZ>                (out, b2g, b2b, b2m, b2v);  

    conv1x1_s2<CH_IN, CH_OUT, SIZE_IN>(x, skw, skip);

    for (int c = 0; c < CH_OUT; c++) {
        for (int h = 0; h < SZ; h++) {
            for (int w = 0; w < SZ; w++) {
                #pragma HLS PIPELINE II=1
                const data_t s = out[c][h][w] + skip[c][h][w];
                out[c][h][w] = (s > (data_t)0) ? s : (data_t)0;
            }
        }
    }
}


// Global Average Pooling

template<int CH, int SZ>
void gap(data_t in[CH][SZ][SZ], data_t out[CH])
{
    #pragma HLS INLINE
    const data_t INV = (data_t)1 / (SZ * SZ);  

    for (int c = 0; c < CH; c++) {
        data_t acc=0;
        for (int h = 0; h < SZ; h++) {
            #pragma HLS PIPELINE II=1  
            for (int w = 0; w < SZ; w++) {
                #pragma HLS UNROLL     
                acc += in[c][h][w];
            }
        }
        out[c] = (acc * INV);  
    }
}

// Fully-Connected Layer

template<int IN_F, int OUT_F>
void dense(
    data_t         x[IN_F],
    const weight_t W[IN_F * OUT_F],
    const weight_t b[OUT_F],
    data_t         y[OUT_F])
{
    #pragma HLS ARRAY_PARTITION variable=x complete  
    #pragma HLS INLINE

    for (int o = 0; o < OUT_F; o++) {
        #pragma HLS PIPELINE II=1
        data_t acc = b[o];
        for (int i = 0; i < IN_F; i++) {
            #pragma HLS UNROLL factor=4
            acc += x[i] * W[i * OUT_F + o];
        }
        y[o] = acc;
    }
}

// Top-Level Function

void resnet20_top(
    data_t input_image[IMAGE_CHANNELS][IMAGE_SIZE][IMAGE_SIZE],
    data_t predictions[NUM_CLASSES])
{
    #pragma HLS INTERFACE m_axi port=input_image depth=3072 offset=slave bundle=gmem_in
    #pragma HLS INTERFACE m_axi port=predictions depth=10   offset=slave bundle=gmem_out

    #pragma HLS INTERFACE s_axilite port=input_image bundle=control
    #pragma HLS INTERFACE s_axilite port=predictions bundle=control
    #pragma HLS INTERFACE s_axilite port=return      bundle=control

    data_t fm16_A[16][32][32];   
    data_t fm16_B[16][32][32];

    data_t fm32_A[32][16][16];   
    data_t fm32_B[32][16][16];

    data_t fm64_A[64][8][8];     
    data_t fm64_B[64][8][8];

    data_t gap_out[64];

    conv3x3<3, 16, 32>(input_image, layer_1_weights, fm16_A);
    bn_relu<16, 32>(fm16_A, layer_2_gamma, layer_2_beta, layer_2_mean, layer_2_variance);

    res_basic<16, 32>(fm16_A,
        layer_4_weights,
        layer_5_gamma,  layer_5_beta,  layer_5_mean,  layer_5_variance,
        layer_7_weights,
        layer_8_gamma,  layer_8_beta,  layer_8_mean,  layer_8_variance,
        fm16_B);

    res_basic<16, 32>(fm16_B,
        layer_11_weights,
        layer_12_gamma, layer_12_beta, layer_12_mean, layer_12_variance,
        layer_14_weights,
        layer_15_gamma, layer_15_beta, layer_15_mean, layer_15_variance,
        fm16_A);

    res_basic<16, 32>(fm16_A,
        layer_18_weights,
        layer_19_gamma, layer_19_beta, layer_19_mean, layer_19_variance,
        layer_21_weights,
        layer_22_gamma, layer_22_beta, layer_22_mean, layer_22_variance,
        fm16_B);

    res_down<16, 32, 32>(fm16_B,
        layer_25_weights,
        layer_26_gamma, layer_26_beta, layer_26_mean, layer_26_variance,
        layer_28_weights,
        layer_30_gamma, layer_30_beta, layer_30_mean, layer_30_variance,
        layer_29_weights,
        fm32_A);

    res_basic<32, 16>(fm32_A,
        layer_33_weights,
        layer_34_gamma, layer_34_beta, layer_34_mean, layer_34_variance,
        layer_36_weights,
        layer_37_gamma, layer_37_beta, layer_37_mean, layer_37_variance,
        fm32_B);

    res_basic<32, 16>(fm32_B,
        layer_40_weights,
        layer_41_gamma, layer_41_beta, layer_41_mean, layer_41_variance,
        layer_43_weights,
        layer_44_gamma, layer_44_beta, layer_44_mean, layer_44_variance,
        fm32_A);

    res_down<32, 64, 16>(fm32_A,
        layer_47_weights,
        layer_48_gamma, layer_48_beta, layer_48_mean, layer_48_variance,
        layer_50_weights,
        layer_52_gamma, layer_52_beta, layer_52_mean, layer_52_variance,
        layer_51_weights,
        fm64_A);

    res_basic<64, 8>(fm64_A,
        layer_55_weights,
        layer_56_gamma, layer_56_beta, layer_56_mean, layer_56_variance,
        layer_58_weights,
        layer_59_gamma, layer_59_beta, layer_59_mean, layer_59_variance,
        fm64_B);

    res_basic<64, 8>(fm64_B,
        layer_62_weights,
        layer_63_gamma, layer_63_beta, layer_63_mean, layer_63_variance,
        layer_65_weights,
        layer_66_gamma, layer_66_beta, layer_66_mean, layer_66_variance,
        fm64_A);

    gap  <64, 8> (fm64_A, gap_out);
    dense<64, 10>(gap_out, layer_70_weights, layer_70_bias, predictions);
}
