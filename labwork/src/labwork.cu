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
	
	if (lwNum != 3 ) {
        inputFilename = std::string(argv[3]);
        labwork.loadInputImage2(inputFilename);
    	}

        timer.start();
            labwork.labwork6_GPU();
            printf("labwork 6 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
		timer.start();
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
	//inputImage2 = jpegLoader.load("../data/sky.jpg");
}

void Labwork::loadInputImage2(std::string inputFileName) {
    inputImage2 = jpegLoader.load(inputFileName);
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
    //output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
	unsigned char g = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x = g; //load global mem too much -> slow
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
	//the kernel is stored in the register -> optimized
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
	/*for (int i=0; i<49; i++){
		sharedkernel[i] = kernel[i];
	} //copy >1000 times => slow
	*/
	int localtid = threadIdx.x + threadIdx.y*blockDim.x;
	if (localtid < 49){
		sharedkernel[localtid] = kernel[localtid];
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

__global__ void threshold(uchar3* input, uchar3* output, int width, int height, int thresholdnumber) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
    int tid = tidx + tidy * width; //gridDim.x*blockDim.x != width
    unsigned char g = (input[tid].x + input[tid].y + input[tid].z) / 3;
	if (g>= thresholdnumber){
		g = 255;
	}else{
		g = 0;
	}
    output[tid].z = output[tid].y = output[tid].x = g; 
}

__global__ void brightness(uchar3* input, uchar3* output, int width, int height, int brightnessnumber) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
    int tid = tidx + tidy * width; //gridDim.x*blockDim.x != width
    unsigned char g = (input[tid].x + input[tid].y + input[tid].z) / 3;
	g += brightnessnumber;
    output[tid].z = output[tid].y = output[tid].x = g; 
    //output[tid].x = input[tid].x + brightnessnumber;
    //output[tid].y = input[tid].y + brightnessnumber;
    //output[tid].z = input[tid].z + brightnessnumber;
}

__global__ void blending(uchar3* input, uchar3* input2, uchar3* output, int width, int height, double weight) {
	
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
    int tid = tidx + tidy * width; //gridDim.x*blockDim.x != width

//rgb
    //output[tid].x = (weight * (double)input[tid].x) + ((1.0 - weight) * (double)input2[tid].x);
    //output[tid].y = (weight * (double)input[tid].y) + ((1.0 - weight) * (double)input2[tid].y);
    //output[tid].z = (weight * (double)input[tid].z) + ((1.0 - weight) * (double)input2[tid].z);

//grayscale
	unsigned char g = (input[tid].x + input[tid].y + input[tid].z) / 3;
	unsigned char g2 = (input2[tid].x + input2[tid].y + input2[tid].z) / 3;
	g = weight*g + (1.0-weight)*g2;
    output[tid].z = output[tid].y = output[tid].x = g; 
}

void Labwork::labwork6_GPU() {
/*
	int value = 0;
	printf("Enter the value: ");
	scanf("%d", &value);
*/
	int pixelCount = inputImage->width * inputImage->height;	
	dim3 blockSize = dim3(32,32);
	dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);
	uchar3 *devInput,*devOutput;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);
	//printf("%d %d\n",inputImage->width, inputImage->height); 
	//printf("%d %d\n",inputImage2->width, inputImage2->height); 

	//6c
	double weight;
	printf("Enter the weight: ");
	scanf("%ld", &weight);
	uchar3 *devInput2;
	cudaMalloc(&devInput2, pixelCount * 3);
	cudaMemcpy(devInput2, inputImage2->buffer, pixelCount*3, cudaMemcpyHostToDevice);

	cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
	//threshold<<<gridSize, blockSize>>>(devInput, devOutput,inputImage->width, inputImage->height, value);
	//brightness<<<gridSize, blockSize>>>(devInput, devOutput,inputImage->width, inputImage->height, value);
	
	blending<<<gridSize, blockSize>>>(devInput, devInput2, devOutput,inputImage->width, inputImage->height, 0.4);
	//printf("blended\n");

	cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost);
	cudaFree(devInput);
	cudaFree(devOutput);	

	//cudaFree(devInput2);

}

__global__ void grayscaletoint(uchar3* input, int* tempMin, int* tempMax, int width, int height) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid+blockDim.x >= width * height) return;
	//extern __shared__ int cacheMin[] ;
	//extern __shared__ int cacheMax[] ;
	__shared__ int cacheMin[1024] ;
	__shared__ int cacheMax[1024] ;

	unsigned int localtid = threadIdx.x;
    	
	cacheMin[localtid] = input[tid].x;
	cacheMax[localtid] = input[tid].x;
	
	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		if (localtid < s) {
			cacheMin[localtid] = min(cacheMin[localtid], cacheMin[localtid + s]);
			cacheMax[localtid] = max(cacheMax[localtid], cacheMax[localtid + s]);
		}
		__syncthreads();
	}	
	
	if (localtid == 0){ 
		tempMin[blockIdx.x] = cacheMin[0];
		tempMax[blockIdx.x] = cacheMax[0];
	}
	//int tid2 = tid *2; //apply reducing block size
	//tempMin[tid] = min(cache[tid2], cache[tid2 + blockDim.x]);
	//tempMax[tid] = max(cache[tid2], cache[tid2 + blockDim.x]);
}

__global__ void reduceMinFinal(int* in) {
	// dynamic shared memory size, allocated in host
	//extern	__shared__ int cache[];
	__shared__ int cache[1024];
	// cache the block content
	unsigned int localtid = threadIdx.x;
	unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
	//cache[localtid] = in[tid] + in[tid + blockDim.x];
	cache[localtid] = min(in[tid], in[tid + blockDim.x]);
	__syncthreads();
	// reduction in cache
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		if (localtid < s) {
			//cache[localtid] += cache[localtid + s];
			cache[localtid] = min(cache[localtid], cache[localtid + s]);
		}
	__syncthreads();
	}
	// only first thread writes back
	if (localtid == 0) in[blockIdx.x] = cache[0];
	//if (localtid == 0) minout = cache[0];
}

__global__ void reduceMaxFinal(int* in) {
	// dynamic shared memory size, allocated in host
	//extern	__shared__ int cache[];
	__shared__ int cache[1024];
	// cache the block content
	unsigned int localtid = threadIdx.x;
	unsigned int tid = threadIdx.x+blockIdx.x*2*blockDim.x;
	//cache[localtid] = in[tid] + in[tid + blockDim.x];
	cache[localtid] = max(in[tid], in[tid + blockDim.x]);
	__syncthreads();
	// reduction in cache
	for (int s = blockDim.x / 2; s > 0; s /= 2) {
		if (localtid < s) {
			//cache[localtid] += cache[localtid + s];
			cache[localtid] = max(cache[localtid], cache[localtid + s]);
		}
	__syncthreads();
	}
	// only first thread writes back
	if (localtid == 0) in[blockIdx.x] = cache[0];
	//if (localtid == 0) maxout = cache[0];
}

__global__ void stretching(uchar3* input, uchar3* output, int minValue, int maxValue)
{
    	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
/*
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
    	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
    	int tid = tidx + tidy * width;
*/
	double numerator = input[tid].x - minValue;
	double denominator = maxValue - minValue;
	unsigned char g = numerator/denominator*255;
	//unsigned char g = (double(input[tid].x - minValue) / double(maxValue - minValue)) * 255;
    output[tid].x = output[tid].y = output[tid].z = g;
}


void Labwork::labwork7_GPU() {

	//printf("Starting \n");
	int pixelCount = inputImage->width * inputImage->height;	
	//printf("%d \n", pixelCount);	
	int blockSize = 1024;
	int numBlock = (pixelCount + blockSize -1) / (blockSize);

	uchar3 *devInput, *tempOutput;
	uchar3 *devOutput;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));


	int *devMin, *devMax ;
	int * minValue = (int *) calloc(numBlock, sizeof(int)) ;
	int * maxValue = (int *) calloc(numBlock, sizeof(int)) ;

	cudaMalloc(&devMin, numBlock * sizeof(int));
	cudaMalloc(&devMax, numBlock * sizeof(int)); 

	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&tempOutput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);


	cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);

	
	grayscale<<<numBlock, blockSize>>>(devInput, tempOutput);
	grayscaletoint<<<numBlock, blockSize>>>(tempOutput, devMin, devMax, inputImage->width, inputImage->height);
	reduceMinFinal<<<numBlock, blockSize>>>(devMin);
	reduceMaxFinal<<<numBlock, blockSize>>>(devMax);
	cudaMemcpy(minValue, devMin, numBlock * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(maxValue, devMax, numBlock * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d %d \n", minValue[0], maxValue[0]);

	stretching<<<numBlock, blockSize>>>(tempOutput, devOutput, minValue[0],  maxValue[0]);
	printf("Copying \n");
	cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost);
	
	cudaFree(devInput);
	cudaFree(devOutput);
	cudaFree(tempOutput);
	cudaFree(devMin);
	cudaFree(devMax);
	
}

struct hsv{
	double *h, *s, *v;
};

__global__ void rgb2hsv(uchar3 *input, hsv output, int width, int height){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width; 

	double normR = (double)input[tid].x/255;
	double normG = (double)input[tid].y/255;
	double normB = (double)input[tid].z/255;

	double normMax = max(normR, max(normG,normB));
	double normMin = min(normR, min(normG,normB));
	double delta = normMax - normMin;

	double h = 0;
	double s = 0;
	double v = 0;

	v = normMax;

	if (normMax != 0){
		s = delta/normMax;	
		if (normMax == normR) h = 60 * fmodf(((normG-normB)/delta),6.0);
		if (normMax == normG) h = 60 * ((normB-normR)/delta +2);
		if (normMax == normB) h = 60 * ((normR-normG)/delta +4);
	}
	
	output.h[tid] = h;
	output.s[tid] = s;
	output.v[tid] = v;
	
	if(tid==0){ //print to check
		printf("R : %d, G : %d, B : %d \n", input[tid].x, input[tid].y, input[tid].z);
		printf("normR : %lf, normG : %lf, B : %lf \n", normR, normG, normB);
		printf("normMax : %lf, normMin : %lf, delta : %lf \n", normMax,normMin, delta); 
		printf("H : %lf, S : %lf V : %lf \n", output.h[tid], output.s[tid], output.v[tid]);
	}
}

__global__ void hsv2rgb(hsv input, uchar3 *output,  int width, int height){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width; 

	double h = input.h[tid];
	double s = input.s[tid];
	double v = input.v[tid];

	double d = h/60;
	int hi = (int)fmodf(d,6.0);
	double f = d - hi;
	double l = v*(1-s);
	double m = v*(1 - (f*s));
	double n = v*(1 - ((1-f)*s));

	double r,g,b;
	if(h>=0 and h<60){
		r = v;
		g = n;
		b = l;
	}
	if(h>=60 and h<120){
		r = m;
		g = v;
		b = l;
	}
	if(h>=120 and h<180){
		r = l;
		g = v;
		b = n;
	}
	if(h>=180 and h<240){
		r = l;
		g = m;
		b = v;
	}
	if(h>=240 and h<300){
		r = n;
		g = l;
		b = v;
	}
	if(h>=300 and h<360){
		r = v;
		g = l;
		b = m;
	}

	output[tid].x = (char)(r*255);
	output[tid].y = (char)(g*255);
	output[tid].z = (char)(b*255);



	if(tid==0){ //print to check
		printf("H : %lf, S : %lf V : %lf \n", h,s,v);
		printf("R : %d, G : %d, B : %d \n", output[tid].x, output[tid].y, output[tid].z);
	}
	
}


void Labwork::labwork8_GPU() {
	int pixelCount = inputImage->width * inputImage->height;	
	dim3 blockSize = dim3(32,32);
	//int numBlock = pixelCount / blockSize;
	dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);
	uchar3 *devInput,*devOutput;
	hsv devHSV;

	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);
	cudaMalloc((void**)&devHSV.h, pixelCount * sizeof(double));
    	cudaMalloc((void**)&devHSV.s, pixelCount * sizeof(double));
	cudaMalloc((void**)&devHSV.v, pixelCount * sizeof(double));

	cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
	rgb2hsv<<<gridSize, blockSize>>>(devInput, devHSV, inputImage->width, inputImage->height);
	hsv2rgb<<<gridSize, blockSize>>>(devHSV, devOutput, inputImage->width, inputImage->height);
	cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost);

	cudaFree(devInput);
	cudaFree(devOutput);
	cudaFree(devHSV.h);
    	cudaFree(devHSV.s);
	cudaFree(devHSV.v);

}


struct histogram{
	unsigned int h[256];
};

__global__ void histogramLocal(uchar3 *input, histogram* output, int width, int height){
	unsigned int tempHisto[256] = {0};
	for (int i=0; i< height; i++){
		int j= input[blockIdx.x*height + i].x; 
		 
		tempHisto[j]++;	
	}
	
	 for (int i = 0; i < 256; i++){
    		output[blockIdx.x].h[i] = tempHisto[i];
		//if(i==100) printf("%d ",tempHisto[i]);
		
	}
	
	
}

__global__ void histogramFinal(histogram * input, int *output, int width, int height){
	// dynamic shared memory size, allocated in host
	//extern	__shared__ int cache[];
	__shared__ unsigned int cache[256];
	// cache the block content
	unsigned int localtid = threadIdx.x;
	cache[localtid] = 0;
	histogram *histo = input;
	__syncthreads();
	// reduction in cache
	for (int i = 0; i < width; i++)  {
		cache[localtid] += histo[i].h[localtid];
	//__syncthreads();
	}
	__syncthreads();
	// only first thread writes back
	
	if (localtid == 0){
		for (int i = 0; i < 256; i++){
    		output[i] = cache[i];
		//printf("%d ",output[i]);
		}
	}
	
	
}

void Labwork::labwork9_GPU() {
	int pixelCount = inputImage->width * inputImage->height;	
	dim3 blockSize = dim3(32,32);
	//int numBlock = pixelCount / blockSize;
	dim3 gridSize = dim3((inputImage->width + blockSize.x -1) / blockSize.x, (inputImage->height + blockSize.y -1) / blockSize.y);
	uchar3 *devInput,*devOutput;
	uchar3 *tempOutput;
	histogram *histoLocal;
	int *histoFinal;
	int * hostHisto = (int *) calloc(256, sizeof(int)) ;

	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);
	cudaMalloc(&tempOutput, pixelCount * 3);
	cudaMalloc(&histoLocal, inputImage->width * sizeof(histogram));
	cudaMalloc(&histoFinal, 256 * sizeof(int));

	cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
	grayscale2d<<<gridSize, blockSize>>>(devInput, tempOutput,inputImage->width, inputImage->height);
	histogramLocal<<<inputImage->width,1>>>(tempOutput, histoLocal, inputImage->width, inputImage->height);
	histogramFinal<<<1,256>>>(histoLocal, histoFinal, inputImage->width, inputImage->height);
	cudaMemcpy(hostHisto, histoFinal, 256 * sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i=0;i<256;i++) {
		printf("%d ",hostHisto[i]);
		sum += hostHisto[i];
	}
	printf("%d %d \n", pixelCount, sum);
	cudaFree(devInput);
	cudaFree(devOutput);
	cudaFree(tempOutput);
	cudaFree(histoLocal);
}

void Labwork::labwork10_GPU() {

}
