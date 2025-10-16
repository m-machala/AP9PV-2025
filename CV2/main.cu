#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"


#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if (err != cudaSuccess ){           \
        fprintf( stderr, "Error %s at line %d in file %s\n",  \
            cudaGetErrorString(err), __LINE__, __FILE__);    \
            exit(-1);                                          \
    }}

__global__ void getIndex(int *outputData) {
    int index = blockIdx.x * blockDim.y + threadIdx.x;
    outputData[index] = index;
}

__global__ void addToVector(int *inputData, int *outputData, int constant) {
    int index = blockIdx.x * blockDim.y + threadIdx.x;
    outputData[index] = inputData[index] + constant; 
}

int main(){
    /*printf("Hello world!\n");

    int deviceCount = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&deviceCount))
    std::cout << deviceCount << std::endl;

    cudaDeviceProp deviceProperties;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProperties, 0))
    std::cout << "Device name: " << deviceProperties.name << std::endl;
    std::cout << "Compute compatibility: " << deviceProperties.major << "." << deviceProperties.minor << std::endl; 
    std::cout << "Max block size: " << deviceProperties.maxThreadsDim[0] << " x, " << 
                                       deviceProperties.maxThreadsDim[1] << " y, " << 
                                       deviceProperties.maxThreadsDim[2] << " z" << std::endl; 

        std::cout << "Max grid size: " << deviceProperties.maxGridSize[0] << " x, " << 
                                           deviceProperties.maxGridSize[1] << " y, " << 
                                           deviceProperties.maxGridSize[2] << " z" << std::endl; */

    int threadCount = 200;
    int* inputData_h = (int*) malloc(threadCount * sizeof(int));
    int* outputData_h = (int*) malloc(threadCount * sizeof(int));

    int* outputData_d = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&outputData_d, threadCount * sizeof(int)));
    int* inputData_d = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&inputData_d, threadCount * sizeof(int)));

    //CUDA_CHECK_RETURN(cudaMemcpy(outputData_d, outputData_h, threadCount * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int gridSize = (threadCount + blockSize - 1) / blockSize;

    getIndex<<<blockSize, gridSize>>>(inputData_d);
    cudaDeviceSynchronize();

    addToVector<<<blockSize, gridSize>>>(inputData_d, outputData_d, 1000);
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(outputData_h, outputData_d, threadCount * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < threadCount; i++) {
        std::cout << outputData_h[i] << ", ";
    }

    free(outputData_h);
    CUDA_CHECK_RETURN(cudaFree(outputData_d));


}
