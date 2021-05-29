#include <chrono>
#include <iostream>

#include <cuda.h>
using namespace std;


void gen_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10;
    }
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        cout << matrix[i] << " ";
        if ((i + 1) % cols == 0) {
            cout << endl;
        }
    }
}

#define BLOCK_SIZE 16
#define FILTER_SIZE 3
#define IDX(row, col, len) ((row)*(len)+(col))

__global__ void laplap(double *from, double *to, double *filter, double divisor, int X, int Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0;

    if (col > 0 && row > 0 && col < X - 1 && row < Y - 1) {
        for (int i = -1; i < 2; ++i) {
            for (int j = -1; j < 2; ++j) {
                // assume that middle element in filter equals to 0
                sum += from[(row + i) * X + (col + j)] * filter[(i + 1) * FILTER_SIZE + j + 1];
            }
        }
        to[row * X + col] = sum / divisor;
    }
}


int main(int argc, char* argv[]) {
    int X = 128, Y = 128;

    double *from = new double[X * Y];

    for (int i = 0; i != X; ++i) {
        for (int j = 0; j != Y; ++j) {
            from[i * Y + j] = 0;
        }

    }

    for (int i = 0; i != X; ++i) {
        from[i * Y] = 1;
    }
    double *to = new double[X * Y];

    // 0 1 0
    // 1 0 1
    // 0 1 0
    double *filter = new double[FILTER_SIZE * FILTER_SIZE];
    for (int i = 0; i != FILTER_SIZE * FILTER_SIZE; ++i) {
        if (i % 2) {
            filter[i] = 1;
        } else {
            filter[i] = 0;
        }
    }



    print_matrix(filter, FILTER_SIZE, FILTER_SIZE);


    double *new_from, *new_to, *new_filter;
    cudaMalloc((void**) &new_from, sizeof(double) * X * Y);
    cudaMalloc((void**) &new_to, sizeof(double) * X * Y);
    cudaMalloc((void**) &new_filter, sizeof(double) * FILTER_SIZE * FILTER_SIZE);

    // copy matrix A and B from host to device memory
    cudaMemcpy(new_from, from, sizeof(double) * X * Y, cudaMemcpyHostToDevice);
    cudaMemcpy(new_filter, filter, sizeof(double) * FILTER_SIZE * FILTER_SIZE, cudaMemcpyHostToDevice);

    dim3 grid((X + BLOCK_SIZE - 1) / BLOCK_SIZE, (Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    int MAX_ITER = 8000;
    for (int it = 0; it != MAX_ITER; ++it) {
        laplap<<<grid, block>>>(new_from, new_to, new_filter, 4, X, Y);
        laplap<<<grid, block>>>(new_to, new_from, new_filter, 4, X, Y);
    }

    cudaMemcpy(to, new_from, sizeof(double) * X * Y, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    FILE *f_name;
    f_name = fopen("res.txt", "w");

    for (int i = 0; i < X; ++i)
    {
        for (int j = 0; j < Y; ++j)
        {
            fprintf(f_name, "%f\t", to[i * X + j]);
        }
         fprintf(f_name, "\n");
    }


    cudaFree(new_from);
    cudaFree(new_to);
    cudaFree(new_filter);


    delete from;
    delete to;
    delete filter;
}
