%%writefile hello.cu
#include <chrono>
#include <iostream>

#include <cuda.h>
using namespace std;

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define RGB_COMPONENT_COLOR 255

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

void PPMToGrayscale(PPMImage& img){
    const float r = 0.299F;
    const float g = 0.587F;
    const float b = 0.114F;

    for (int i = 0; i != img.all; ++i) {
        float gray = img.data[i].red * r + img.data[i].green * g + img.data[i].blue * b;
        img.data[i].red = gray;
        img.data[i].blue = gray;
        img.data[i].green = gray;
    }
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

__global__ void make_hist(int *image, int *hist, int x, int y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < x * y) {

      int value = image[tid];

      int bin = value % 256;

      atomicAdd(&hist[bin], 1);

    }
}


int main(int argc, char* argv[]) {
    PPMImage img;
    readPPM("dogs.ppm", img);
    PPMToGrayscale(img);
    writePPM("dogs_gray.ppm", img);

    int *image_for_cuda = new int[img.x * img.y];
    int *hist_for_cuda = new int[256];
    for (int i = 0; i != 256; ++i) {
        hist_for_cuda[i] = 0;
    }

    for (int i = 0; i != img.x * img.y; ++i) {
        image_for_cuda[i] = img.data[i].green;
    }

    int *image, *hist;
    cudaMalloc((void**) &image, sizeof(int) * img.x * img.y);
    cudaMalloc((void**) &hist, sizeof(int) * 256);

    // copy matrix A and B from host to device memory
    cudaMemcpy(image, image_for_cuda, sizeof(int) * img.x * img.y, cudaMemcpyHostToDevice);
    cudaMemcpy(hist, hist_for_cuda, sizeof(int) * 256, cudaMemcpyHostToDevice);

    dim3 grid((img.x * img.y + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);


    make_hist<<<grid, block>>>(image, hist, img.x, img.y);

    cudaThreadSynchronize();

    cudaMemcpy(hist_for_cuda, hist, sizeof(int) *  256, cudaMemcpyDeviceToHost);

    FILE *f_name;
    f_name = fopen("res.txt", "w");

    for (int i = 0; i < 256; ++i)
    {
        fprintf(f_name, "%d\t", hist_for_cuda[i]);

    }
    fprintf(f_name, "\n");


    cudaFree(hist);
    cudaFree(image);


    delete image_for_cuda;
    delete hist_for_cuda;
}

