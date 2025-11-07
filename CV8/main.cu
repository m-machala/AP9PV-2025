#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"

#define BLOCK_SIZE (16)

#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if (err != cudaSuccess ){           \
        fprintf( stderr, "Error %s at line %d in file %s\n",  \
            cudaGetErrorString(err), __LINE__, __FILE__);    \
            exit(-1);                                          \
    }}

    __global__ void multiplyMatrices(float* matrixA, float* matrixB, float* outputMatrix, int n, int m, int o) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
        int columnIndex = blockIdx.x * blockDim.x + threadIdx.x;

        float output = 0.0f;
        int tileCount = (o + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int tile = 0; tile < tileCount; tile++) {
            int aColumnIndex = tile * BLOCK_SIZE + threadIdx.x;
            int bRowIndex = tile * BLOCK_SIZE + threadIdx.y;

            if (rowIndex < m && aColumnIndex < o) {
                As[threadIdx.y][threadIdx.x] = matrixA[rowIndex * o + aColumnIndex];
            }
            else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (bRowIndex < o && columnIndex < n) {
                Bs[threadIdx.y][threadIdx.x] = matrixB[bRowIndex * n + columnIndex];
            }
            else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            for (int i = 0; i < BLOCK_SIZE; i++) {
                output += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }
        }
        
        if (rowIndex < m && columnIndex < n) {
            outputMatrix[rowIndex * n + columnIndex] = output;
        }
    }


int main(){
    int N = 4; // width of B and output
    int M = 4; // height of A and output
    int O = 4; // width of A, height of B

    float *matrixA, *matrixB, *outputMatrix;

    CUDA_CHECK_RETURN(cudaMallocManaged(&matrixA, M * O * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMallocManaged(&matrixB, O * N * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMallocManaged(&outputMatrix, M * N * sizeof(float)));

    for (int i = 0; i < M * O; i++) {
        matrixA[i] = i;
    }

    for (int i = 0; i < O * N; i++) {
        matrixB[i] = i;
    }
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    multiplyMatrices<<<gridSize, blockSize>>>(matrixA, matrixB, outputMatrix, N, M, O);
    cudaDeviceSynchronize();

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << outputMatrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }




}
