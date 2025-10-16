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

    __global__ void multiplyMatrices(float* matrixA, float* matrixB, float* outputMatrix, int n, int m, int o) {
        int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
        int columnIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (rowIndex < m && columnIndex < n) {
            float output = 0.0f;

            for(int i = 0; i < o; i++) {
                output += matrixA[rowIndex * o + i] * matrixB[i * n + columnIndex];
            }

            outputMatrix[rowIndex * n + columnIndex] = output;
        }
    }

// Grafika courtesy of Bc. Petr Špaňhel
//            N
//      | 0  0  0  0 |
//   O  | 0  0  0  0 |
//      | 0  0  0  0 |
//
//
//           O
//      | 0  0  0 |
//      | 0  0  0 |
//    M | 0  0  0 |
//      | 0  0  0 |
//      | 0  0  0 |

int main(){
    int N = 2; // width of B and output
    int M = 2; // height of A and output
    int O = 3; // width of A, height of B

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
    int blockSizeSingle = 16;

    dim3 blockSize(blockSizeSingle, blockSizeSingle);
    dim3 gridSize((N + blockSizeSingle - 1) / blockSizeSingle, (M + blockSizeSingle - 1) / blockSizeSingle);

    multiplyMatrices<<<gridSize, blockSize>>>(matrixA, matrixB, outputMatrix, N, M, O);
    cudaDeviceSynchronize();

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << outputMatrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }




}
