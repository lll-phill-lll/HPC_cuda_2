%%writefile hello.cu
#include <chrono>
#include <iostream>

#include <cuda.h>

using namespace std::chrono;
using namespace std;
#define RGB_COMPONENT_COLOR 255


#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}


void gen_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10;
    }
}

void print_matrix(int *matrix, int rows, int cols) {
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

__global__ void blur(int *from, int *to, int *filter, int divisor, int X, int Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

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

__global__ void blur_median(int *from, int *to, int *filter, int X, int Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int filtered[9] = {0,0,0,0,0,0,0,0,0};

    if (col > 0 && row > 0 && col < X - 1 && row < Y - 1) {
        for (int i = -1; i < 2; ++i) {
            for (int j = -1; j < 2; ++j) {
                // assume that middle element in filter equals to 0
                filtered[(i + 1) * FILTER_SIZE + j + 1] = from[(row + i) * X + (col + j)] * filter[(i + 1) * FILTER_SIZE + j + 1];
            }
        }
        for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
			for (int j = i + 1; j < FILTER_SIZE * FILTER_SIZE; ++j) {
				if (filtered[i] > filtered[j]) {
					int tmp = filtered[i];
					filtered[i] = filtered[j];
					filtered[j] = tmp;
				}
			}
		}
        to[row * X + col] = filtered[4];
    }
}


int main(int argc, char* argv[]) {
    PPMImage image;
    readPPM("dogs.ppm", image);
    // writePPM("nature_before.ppm", image);
    int X = image.x;
    int Y = image.y;

    int *from = new int[X * Y];
    int *to = new int[X * Y];

    int *filter = new int[FILTER_SIZE * FILTER_SIZE];
    for (int i = 0; i != FILTER_SIZE; ++i) {
        for (int j = 0; j != FILTER_SIZE; ++j) {
            filter[i * FILTER_SIZE + j] = 1;
        }
    }
    filter[4] = 0;
    print_matrix(filter, FILTER_SIZE, FILTER_SIZE);

    for (int channel = 0; channel != 3; ++channel) {

        for (int i = 0; i != image.all; ++i) {
            if (channel == 0) {
                from[i] = image.data[i].blue;
            } else if (channel == 1) {
                from[i] = image.data[i].red;
            } else {
                from[i] = image.data[i].green;
            }

        }

        // gen_matrix(from, X, Y);
        // print_matrix(from, X, Y);



        high_resolution_clock::time_point total_start = high_resolution_clock::now();


        int *new_from, *new_to, *new_filter;
        cudaMalloc((void**) &new_from, sizeof(int) * X * Y);
        cudaMalloc((void**) &new_to, sizeof(int) * X * Y);
        cudaMalloc((void**) &new_filter, sizeof(int) * FILTER_SIZE * FILTER_SIZE);

        // copy matrix A and B from host to device memory
        cudaMemcpy(new_from, from, sizeof(int) * X * Y, cudaMemcpyHostToDevice);
        cudaMemcpy(new_filter, filter, sizeof(int) * FILTER_SIZE * FILTER_SIZE, cudaMemcpyHostToDevice);

        dim3 grid((X + BLOCK_SIZE - 1) / BLOCK_SIZE, (Y + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        // blur<<<grid, block>>>(new_from, new_to, new_filter, 9, X, Y);
        blur_median<<<grid, block>>>(new_from, new_to, new_filter, X, Y);

        cudaMemcpy(to, new_to, sizeof(int) * X * Y, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        // Insert your code that runs C=A*B on GPU below
        cudaFree(new_from);
        cudaFree(new_to);
        cudaFree(new_filter);


        high_resolution_clock::time_point total_end = high_resolution_clock::now();
        double total_time = duration_cast<duration<double>>(total_end - total_start).count();

        cout << "Total (kernel+copy) time: " << total_time << endl;

        // print_matrix(from, X, Y);
        cout << "----------------" <<  endl;
        // print_matrix(to, X, Y);


        for (int i = 0; i != image.all; ++i) {
            if (channel == 0) {
                image.data[i].blue = to[i];
            } else if (channel == 1) {
                image.data[i].red = to[i];
            } else {
                image.data[i].green = to[i];
            }

        }
    }
    writePPM("dogs_after.ppm", image);


    delete from;
    delete to;
    delete filter;
}

