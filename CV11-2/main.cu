#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <algorithm>

#include "cuda_runtime.h"
#include "pngio.h"

#define BLOCK_SIZE (16u)

#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if (err != cudaSuccess ){           \
        fprintf( stderr, "Error %s at line %d in file %s\n",  \
            cudaGetErrorString(err), __LINE__, __FILE__);    \
            exit(-1);                                          \
    }}

__device__ float rastrigin(float x1, float x2) {
    return 20 + (x1 * x1 - 10 * cos(2 * M_PI * x1) + (x2 * x2 - 10 * cos(2 * M_PI * x2)));
}

__global__ void getResults(float* output, unsigned int height, unsigned int width, float lowerBound, float upperBound) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < width && row < height) {
        float stepX1 = (upperBound - lowerBound) / (width - 1);
        float stepX2 = (upperBound - lowerBound) / (height - 1);

        float x1 = lowerBound + column * stepX1;
        float x2 = lowerBound + row * stepX2;

        output[row * width + column] = rastrigin(x1, x2);
    }
}

__global__ void visualiseResults(float* input, unsigned int height, unsigned int width, 
                                 unsigned char* redOutput, unsigned char* greenOutput, unsigned char* blueOutput,
                                 float min, float max) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < width && row < height) {
        int index = row * width + column;
        float normalizedValue = (input[index] - min) / (max - min);
        normalizedValue = fminf(1.0f, fmaxf(0.0f, normalizedValue));
        if (normalizedValue * 255 <= 64) {
            redOutput[index] = 255;
        }
        else {
            redOutput[index] = 0;
        }

        if (normalizedValue * 255 > 64 && normalizedValue * 255 <= 128) {
            greenOutput[index] = 255;
        }
        else {
            greenOutput[index] = 0;
        }

        if (normalizedValue * 255 > 128) {
            blueOutput[index] = 255;
        }
        else {
            blueOutput[index] = 0;
        }

    }
}

int main(){
    unsigned int height = 512;
    unsigned int width = 512;
    unsigned int size = width * height;

    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    float *h_values = new float[size];
    float *d_values = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_values, size * sizeof(float)));
    getResults<<<gridSize, blockSize>>>(d_values, height, width, -5.12f, 5.12f);
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(h_values, d_values, size * sizeof(float), cudaMemcpyDeviceToHost));

    float minValue = *std::min_element(h_values, h_values + size);
    float maxValue = *std::max_element(h_values, h_values + size);
    
    unsigned char *h_r = new unsigned char[size];
    unsigned char *h_g = new unsigned char[size];
    unsigned char *h_b = new unsigned char[size];

    unsigned char *d_r = NULL, *d_g = NULL, *d_b = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r, size * sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g, size * sizeof(unsigned char)));    
    CUDA_CHECK_RETURN(cudaMalloc(&d_b, size * sizeof(unsigned char)));

    visualiseResults<<<gridSize, blockSize>>>(d_values, height, width, d_r, d_g, d_b, minValue, maxValue);
    
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(h_r, d_r, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g, d_g, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b, d_b, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    png::image<png::rgb_pixel> image(width, height);
    pvg::rgb3ToPng(image, h_r, h_g, h_b);
    image.write("../rastrigin.png");    

    delete [] h_r;
    delete [] h_g;
    delete [] h_b;

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
       
}   
