#include "Estimator.cuh"
#include "../utils/Dataset.h"
#include <iostream>
#include <math.h>

__global__ void partialMoments(float *x, float* moments, int n){
	extern __shared__ float sdata[];
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x * 2,
				 total = blockDim.x  * gridDim.x,
				 thidx = threadIdx.x;

    if(index < n)
    	sdata[thidx] = x[index] + x[index + blockDim.x];
    else
    	sdata[thidx] = (powf(x[index - total], 2) +  powf(x[index + blockDim.x - total], 2));

    __syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if(thidx < s){
			sdata[thidx] += sdata[thidx + s];
			__syncthreads();
		}
	}
	if(thidx == 0)
		moments[blockIdx.x] = sdata[thidx];
}


__global__ void composeMoments(float * moments, int block, int n){
	float mean=0, var=0; int stride = block/2;
	for (int i = 0; i < stride; ++i) {
		mean += moments[i];
		var += moments[i + stride];
	}
	moments[0] = mean/n; moments[stride] = sqrtf(var/n - powf(mean/n, 2));
}

__global__ void standardNormalization(float *X, float *moments, int stride){
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	X[index] = (X[index] - moments[0])/moments[stride];
}


__device__ float gaussiankenelParallel(float distace, float sigma){
	return exp(- distace / (2 * pow(sigma, 2)));
}


__global__ void fillGramMatrix(float *X, float *GM, float sigma){

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;





	GM[index] = sigma;

}


namespace Estimator {

	unsigned int computeBlocks(int threads, int n){
		unsigned int blocks = (n + threads - 1)/threads;
		blocks = (blocks % 2 == 0) ? blocks : blocks + 1;
		return blocks;
	}

	float getSigma(int rows, int cols) {
		float penalizer = -1.0/(4 + cols);
		float scottFactor = pow(rows, penalizer);
		return sqrt(2 * cols) * scottFactor;
	}


	float computeInformationTheoryParallel(float *X, float *Y,
			int rows, int xcols, int ycols){ checkAvailableDevices();

		const unsigned int threads = 1024; size_t sfloat = sizeof(float);
		unsigned int blocks = computeBlocks(threads, rows * xcols);

		// Normalizing X

		size_t sizex = sfloat * rows * xcols; float *Xdev, *momentsx;

		cudaMalloc((void **) &Xdev, sizex);
		cudaMalloc((void **) &momentsx, sfloat * blocks);

		cudaMemcpy(Xdev, X, sizex, cudaMemcpyHostToDevice);

		/*partialMoments<<<blocks, threads, sfloat * threads>>>(
				Xdev, momentsx, rows * xcols);

		composeMoments<<<1, 1>>>(momentsx, blocks, rows * xcols);

		standardNormalization<<<blocks, threads>>>(
				Xdev, momentsx, blocks / 2);

		// Normalizing Y

		size_t sizey = sfloat * rows * ycols; float *Ydev, *momentsy;
		blocks = computeBlocks(threads, rows * ycols);

		cudaMalloc((void **) &Ydev, sizey);
		cudaMalloc((void **) &momentsy, sfloat * blocks);

		cudaMemcpy(Ydev, Y, sizey, cudaMemcpyHostToDevice);

		partialMoments<<<blocks, threads, sfloat * threads>>>(
				Ydev, momentsy, rows * ycols);

		composeMoments<<<1, 1>>>(momentsy, blocks, rows * ycols);

		standardNormalization<<<blocks, threads>>>(
				Ydev, momentsy, blocks / 2);*/


		/* Kernel Computation */

		float * gramX, * gramY; size_t sgram = sfloat * pow(rows, 2);

		cudaMalloc((void **) &gramX, sgram);
		cudaMalloc((void **) &gramY, sgram);

		blocks = computeBlocks(threads, pow(rows, 2));

		dim3 threadsGram(threads, threads, 1);
		dim3 BlockGram(blocks, blocks, 1);

		fillGramMatrix<<<BlockGram, threadsGram>>>(
				Xdev, gramX, getSigma(rows, xcols));




		cudaFree(Xdev); cudaFree(momentsx);
		cudaFree(gramX); cudaFree(gramY);
		//cudaFree(Ydev); cudaFree(momentsy);


		return 1.0;
	}

	void checkAvailableDevices(){
		int deviceCount; cudaGetDeviceCount(&deviceCount);
		if(deviceCount == 0){
			fprintf(stderr, "Error: No devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}
	}

}


