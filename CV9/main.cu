#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <png.h>

#include "cuda_runtime.h"
#include "pngio.h"


#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if (err != cudaSuccess ){           \
        fprintf( stderr, "Error %s at line %d in file %s\n",  \
            cudaGetErrorString(err), __LINE__, __FILE__);    \
            exit(-1);                                          \
    }}

#define WIDTH (800u)
#define HEIGHT (600u)
#define BLOCK_SIZE (16u)

double h_diagonal = 0;
__constant__ double d_diagonal;

__global__ void createImage(unsigned char *image) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < WIDTH && y < HEIGHT) {
        unsigned int i = (y * WIDTH + x) * 3;
        image[i] = float(x) / WIDTH * 255;
        image[i + 1] = float(y) / HEIGHT * 255;
        image[i + 2] = sqrtf(powf(x, 2) + powf(y, 2)) / d_diagonal * 255;
    }
}

int main(){
    int size = WIDTH * HEIGHT * 3;

    h_diagonal = sqrtf(powf(WIDTH, 2) + powf(HEIGHT, 2));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_diagonal, &h_diagonal, sizeof(double)));

    unsigned char *h_image = new unsigned char [size];
    unsigned char *d_image = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_image, size * sizeof(unsigned char)))

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    createImage<<<grid_dim, block_dim>>>(d_image);

    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(h_image, d_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    png::image<png::rgb_pixel> image(WIDTH, HEIGHT);
    pvg::rgbToPng(image, h_image);
    image.write("../image.png");

    delete [] h_image;
    cudaFree(d_image);
}
