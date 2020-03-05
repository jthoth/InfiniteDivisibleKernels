#include "Estimator.cuh"
#include <iostream>
#include <math.h>


__global__ void computeStd(float* data, float* moments, int n, int pad){
	extern __shared__ float std[]; float mean = moments[pad - 1];
	unsigned int threadIndex = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * (blockDim.x * 2);
	std[threadIndex] = 0;
	if(index < n)
		std[threadIndex] = powf(data[index], 2) + powf(data[index + blockDim.x], 2);
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIndex < s) std[threadIndex] += std[threadIndex + s];
		__syncthreads();
	}
	if (index == 0) moments[index + pad]  =  sqrtf(std[0]/n - powf(mean, 2));
}


__global__ void computeMean(float* data, float* moments, int n, int pad){
	extern __shared__ float mean[];
	unsigned int threadIndex = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * (blockDim.x * 2);
	mean[threadIndex] = 0;
	if(index < n)
		mean[threadIndex] =  data[index] +  data[index + blockDim.x];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIndex < s) mean[threadIndex] += mean[threadIndex + s];
		__syncthreads();
	}
	if (index == 0) moments[index + pad] = mean[0]/n;
}


namespace Estimator {

	float computeInformationTheoryParallel(float *X, float *Y,
			int rows, int xcols, int ycols){

		availableDevices(); size_t fsize = sizeof(float);

		float *Xdev, *moments, momentsHost[4];
		cudaMalloc((void **) &moments, 4 * fsize);
	    cudaMalloc((void **) &Xdev, rows * xcols * fsize);

	    cudaMemcpy(Xdev, X, rows * xcols * fsize, cudaMemcpyHostToDevice);

		const unsigned int threads = 1024;
		unsigned int block = (rows * xcols  + threads - 1)/threads;
		computeMean<<<block, threads, threads * fsize>>>(Xdev, moments, rows * xcols, 0);
		computeStd<<<block, threads, threads * fsize>>>(Xdev, moments, rows * xcols, 1);

		float *Ydev;
	    cudaMalloc((void **) &Ydev, rows * ycols * fsize);
	    cudaMemcpy(Ydev, Y, rows * ycols * fsize, cudaMemcpyHostToDevice);

	    block = (rows * ycols  + threads - 1)/threads;
		computeMean<<<block, threads, threads * fsize>>>(Ydev, moments, rows * ycols, 2);
		computeStd<<<block, threads, threads * fsize>>>(Ydev, moments, rows * ycols, 3);

	    cudaDeviceSynchronize();


	    cudaMemcpy(momentsHost, moments, 4 * fsize, cudaMemcpyDeviceToHost);

	    std::cout << "\n\n Blocks: "  << block << '\t' << threads  << '\n' << '\n';

	    for (int i = 0; i < 4; ++i) {
			std::cout <<  momentsHost[i] << '\t';
		}

	    cudaFree(Xdev); /*cudaFree(Ydev);*/ cudaFree(moments);



		return 1.0;
	}

	void availableDevices(){
		int deviceCount; cudaGetDeviceCount(&deviceCount);
		if(deviceCount == 0){
			fprintf(stderr, "Error: No devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}
	}

}


