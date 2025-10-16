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

__global__ void addToVector(int* inputVector, int vectorSize, int value) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < vectorSize) {
        inputVector[index] += value;
    }
}


int main(){
    int vectorSize = 1000;
    //int *vector = (int*)malloc(sizeof(int) * vectorSize);
    int *managedVector = NULL;
    CUDA_CHECK_RETURN(cudaMallocManaged(&managedVector, sizeof(int) * vectorSize));

    for (int i = 0; i < vectorSize; i++) {
        managedVector[i] = i;
    }

    /*

    int *vector_d = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&vector_d, sizeof(int) * vectorSize));
    CUDA_CHECK_RETURN(cudaMemcpy(vector_d, vector, sizeof(int) * vectorSize, cudaMemcpyHostToDevice));
    */
    int blockSize = 128;
    int gridSize = (vectorSize + blockSize - 1) / blockSize;

    addToVector<<<gridSize, blockSize>>>(managedVector, vectorSize, 10000);
    cudaDeviceSynchronize();
    /*
    addToVector<<<gridSize, blockSize>>>(vector_d, vectorSize, 10000);
    cudaDeviceSynchronize();

    CUDA_CHECK_RETURN(cudaMemcpy(vector, vector_d, sizeof(int) * vectorSize, cudaMemcpyDeviceToHost));
*/
    for (int i = 0; i < vectorSize; i++) {
        std::cout << managedVector[i] << ", ";
    }
}
