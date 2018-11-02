#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
	printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
	timer.start();
            labwork.labwork5_GPU();
	printf("labwork 5 GPU nonshared ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
	    labwork.saveOutputImage("labwork5-gpu-out.jpg");
	timer.start();
            labwork.labwork5_GPU2();
	printf("labwork 5 GPU shared ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu2-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
	int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
#pragma omp parallel for schedule (dynamic)
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    	int numberGPUs = 0;
	cudaGetDeviceCount(&numberGPUs);
	printf("Number of GPUs: %d \n", numberGPUs);
	for (int i = 0; i< numberGPUs; i++){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("GPU #%d \n" , i);
		printf(" - GPU name: %s \n", prop.name);
		printf(" - Clock rate: %d \n", prop.clockRate);
		printf(" - Number of cores: %d \n", getSPcores(prop));
		printf(" - Number of multiprocessors: %d \n", prop.multiProcessorCount);
		printf(" - Warp size: %d \n", prop.warpSize);
		printf(" - Memory clock rate: %d \n", prop.memoryClockRate);
		printf(" - Memory bus width: %d \n", prop.memoryBusWidth);
	}

}

__global__ void grayscale(uchar3 *input, uchar3 *output) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned char g = (input[tid].x + input[tid].y + input[tid].z) / 3;
	output[tid].z = output[tid].y = output[tid].x = g; //store in the register -> faster
}

void Labwork::labwork3_GPU() {
	
	int pixelCount = inputImage->width * inputImage->height;	
	int blockSize = 1024;
	int numBlock = pixelCount / blockSize;
	uchar3 *devInput,*devOutput;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);
	cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
	grayscale<<<numBlock, blockSize>>>(devInput, devOutput);
	cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost);
	cudaFree(devInput);
	cudaFree(devOutput);

}

__global__ void grayscale2d(uchar3* input, uchar3* output, int width, int height) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
    int tid = tidx + tidy * width; //gridDim.x*blockDim.x != width
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x; //load global mem too much -> slow
}
void Labwork::labwork4_GPU() {
	int pixelCount = inputImage->width * inputImage->height;	
	//int blockSize = 1024;
	dim3 blockSize = dim3(1024,1);
	//int numBlock = pixelCount / blockSize;
	//dim3 gridSize = dim3((pixelCount / (blockSize.x*blockSize.y))/2, 2);
	dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);
	uchar3 *devInput,*devOutput;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);
	cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
	grayscale2d<<<gridSize, blockSize>>>(devInput, devOutput,inputImage->width, inputImage->height);
	cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost);
	cudaFree(devInput);
	cudaFree(devOutput);
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

__global__ void nonsharedblur(uchar3* input, uchar3* output, int width, int height) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width; //gridDim.x*blockDim.x != width
	/*	
	unsigned char g = input[tid].x;
	int sum = 0;
	avg blur
	sum +=input[tidx + tidy * width].x +
		input[(tidx -1) + (tidy -1) * width].x
		input[(tidx ) + (tidy -1) * width].x
		input[(tidx +1) + (tidy -1) * width].x
		input[(tidx -1) + (tidy ) * width].x
		input[(tidx +1) + (tidy ) * width].x
		input[(tidx -1) + (tidy +1) * width].x
		input[(tidx ) + (tidy +1) * width].x
		input[(tidx +1) + (tidy +1) * width].x
	sum /= 9;
	output[tidx + tidy*width].x = output[tidx + tidy*width].y = output[tidx + tidy*width].z = sum;
	*/
	int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
	 0, 3, 13, 22, 13, 3, 0,  
	 1, 13, 59, 97, 59, 13, 1,  
	 2, 22, 97, 159, 97, 22, 2,  
	 1, 13, 59, 97, 59, 13, 1,  
	 0, 3, 13, 22, 13, 3, 0,
	 0, 0, 1, 2, 1, 0, 0 };
	
	int sum = 0;
    	int c = 0;
	for (int row = -3; row<3; row++){
		for (int col = -3; col <3; col++){
		    int i = tidx + col;
		    int j = tidy + row;
		    if (i < 0) continue;
		    if (i >= width) continue;
		    if (j < 0) continue;
		    if (j >= height) continue;
		    int tid = j * width + i;
		    unsigned char g = (input[tid].x + input[tid].y + input[tid].z)/3;
		    int coefficient = kernel[(row+3) * 7 + col + 3];
		    sum = sum + g * coefficient;
		    c += coefficient;
		}
	}
	sum /= c;
    	output[tid].z = output[tid].y = output[tid].x = sum;
	//int posOut = tidy * width + tidx;
        //output[posOut * 3] = output[posOut * 3 + 1] = output[posOut * 3 + 2] = sum;
        
	
}

void Labwork::labwork5_GPU() {
    int pixelCount = inputImage->width * inputImage->height;	
	dim3 blockSize = dim3(32,32);
	dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);
	uchar3 *devInput,*devOutput;
	outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
	cudaMemcpy(devInput, inputImage->buffer, pixelCount*sizeof(uchar3), cudaMemcpyHostToDevice);
	nonsharedblur<<<gridSize, blockSize>>>(devInput, devOutput,inputImage->width, inputImage->height);
	cudaMemcpy(outputImage, devOutput, pixelCount*sizeof(uchar3), cudaMemcpyDeviceToHost);
	cudaFree(devInput);
	cudaFree(devOutput);
}


__global__ void sharedblur(uchar3* input, uchar3* output, int* kernel, int width, int height) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width; 
	/*
	int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
	 0, 3, 13, 22, 13, 3, 0,  
	 1, 13, 59, 97, 59, 13, 1,  
	 2, 22, 97, 159, 97, 22, 2,  
	 1, 13, 59, 97, 59, 13, 1,  
	 0, 3, 13, 22, 13, 3, 0,
	 0, 0, 1, 2, 1, 0, 0 };
	*/
	__shared__ int sharedkernel[49];
	for (int i=0; i<49; i++){
		sharedkernel[i] = kernel[i];
	}
	__syncthreads();
	int sum = 0;
    	int c = 0;
	for (int row = -3; row<3; row++){
		for (int col = -3; col <3; col++){
			int i = tidx + col;
		    int j = tidy + row;
		    if (i < 0) continue;
		    if (i >= width) continue;
		    if (j < 0) continue;
		    if (j >= height) continue;
		    int tid = j * width + i;
		    unsigned char g = (input[tid].x + input[tid].y + input[tid].z)/3;
		    int coefficient = sharedkernel[(row+3) * 7 + col + 3];
		    sum = sum + g * coefficient;
		    c += coefficient;
		}
	}
	sum /= c;
    	output[tid].z = output[tid].y = output[tid].x = sum;
}

void Labwork::labwork5_GPU2() {
	int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
	 0, 3, 13, 22, 13, 3, 0,  
	 1, 13, 59, 97, 59, 13, 1,  
	 2, 22, 97, 159, 97, 22, 2,  
	 1, 13, 59, 97, 59, 13, 1,  
	 0, 3, 13, 22, 13, 3, 0,
	 0, 0, 1, 2, 1, 0, 0 };
	int *share;
    int pixelCount = inputImage->width * inputImage->height;	
	dim3 blockSize = dim3(32,32);
	dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);
	uchar3 *devInput,*devOutput;
	outputImage = static_cast<char *>(malloc(pixelCount * sizeof(uchar3)));
	cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
	cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
	cudaMalloc(&share, sizeof(kernel));
	cudaMemcpy(devInput, inputImage->buffer, pixelCount*sizeof(uchar3), cudaMemcpyHostToDevice);
	cudaMemcpy(share, kernel, sizeof(kernel), cudaMemcpyHostToDevice);
	sharedblur<<<gridSize, blockSize>>>(devInput, devOutput, share, inputImage->width, inputImage->height);
	cudaMemcpy(outputImage, devOutput, pixelCount*sizeof(uchar3), cudaMemcpyDeviceToHost);
	cudaFree(devInput);
	cudaFree(devOutput);
	cudaFree(share);
}


void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
