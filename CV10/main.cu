#include <stdio.h>
#include <stdlib.h>
#include <png.h>

#include "cuda_runtime.h"
#include "pngio.h"

#define BLOCK_SIZE (16u)
#define FILTER_SIZE (5u)
#define TILE_SIZE (12u)

#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if (err != cudaSuccess ){           \
        fprintf( stderr, "Error %s at line %d in file %s\n",  \
            cudaGetErrorString(err), __LINE__, __FILE__);    \
            exit(-1);                                          \
    }}

__global__ void blurImage(unsigned char *input, unsigned char *output, unsigned int pitch, unsigned int width, unsigned int height) {
    int x_o = TILE_SIZE * blockIdx.x + threadIdx.x;
    int y_o = TILE_SIZE * blockIdx.y + threadIdx.y;

    int x_i = x_o - 2;
    int y_i = y_o - 2;

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    if (x_i >= 0 && x_i < width && y_i >= 0 && y_i < height) {
        sBuffer[threadIdx.y][threadIdx.x] = input[y_i * pitch + x_i];
    }
    else {
        sBuffer[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    int sum = 0;
    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
        for (int r = 0; r < FILTER_SIZE; r++) {
            for (int c = 0; c < FILTER_SIZE; c++) {
                sum += sBuffer[threadIdx.y + r][threadIdx.x + c];
            }
        }
        sum /= FILTER_SIZE * FILTER_SIZE;
        if(x_o < width && y_o < height) {
            output[y_o * width + x_o] = sum;
        }
    }
}

int main(){
    png::image<png::rgb_pixel> image("../image.png");
    unsigned int width = image.get_width();
    unsigned int height = image.get_height();

    int size = width * height;

    unsigned char *h_r = new unsigned char[size];
    unsigned char *h_g = new unsigned char[size];
    unsigned char *h_b = new unsigned char[size];

    unsigned char *h_r_n = new unsigned char[size];
    unsigned char *h_g_n = new unsigned char[size];    
    unsigned char *h_b_n = new unsigned char[size];

    pvg::pngToRgb3(h_r, h_g, h_b, image);

    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;

    size_t pitch_r;
    size_t pitch_g;
    size_t pitch_b;

    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r, &pitch_r, width * sizeof(unsigned char), height * sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g, &pitch_g, width * sizeof(unsigned char), height * sizeof(unsigned char)));    
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b, &pitch_b, width * sizeof(unsigned char), height * sizeof(unsigned char)));

    CUDA_CHECK_RETURN(cudaMemcpy2D(d_r, pitch_r, h_r, width, width * sizeof(unsigned char), height * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_g, pitch_g, h_g, width, width * sizeof(unsigned char), height * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_b, pitch_b, h_b, width, width * sizeof(unsigned char), height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    unsigned char *d_r_n = NULL;
    unsigned char *d_g_n = NULL;
    unsigned char *d_b_n = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, size * sizeof(unsigned char)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, size * sizeof(unsigned char)));    
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, size * sizeof(unsigned char)));

    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    blurImage<<<gridSize, blockSize>>>(d_r, d_r_n, pitch_r, width, height);
    blurImage<<<gridSize, blockSize>>>(d_g, d_g_n, pitch_g, width, height);
    blurImage<<<gridSize, blockSize>>>(d_b, d_b_n, pitch_b, width, height);

    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n, d_r_n, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n, d_g_n, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n, d_b_n, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    pvg::rgb3ToPng(image, h_r_n, h_g_n, h_b_n);
    image.write("../blurred.png");    

    delete [] h_r;
    delete [] h_g;
    delete [] h_b;
    delete [] h_r_n;
    delete [] h_g_n;
    delete [] h_b_n;

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_r_n);
    cudaFree(d_g_n);
    cudaFree(d_b_n);
            
}   
