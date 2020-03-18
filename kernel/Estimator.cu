#include "Estimator.cuh"
#include "../utils/Dataset.h"
#include <iostream>
#include <math.h>


class RotationArgs{
	public:
	  float value = 1e-9, _cos, _sin;
	  unsigned int p, q, n;

	};

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

__global__ void fillGramMatrix(float* in, float* out,
		float sigma, int n, int m){

	__shared__ float Ys[16][16];
	__shared__ float Xs[16][16];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int yBegin = by * 16 * m;
	int xBegin = bx * 16 * m;

	int yEnd = yBegin + m - 1, y, x, k, o;

	float tmp, s = 0;

	for (y = yBegin, x = xBegin; y <= yEnd;	y += 16, x += 16){
		Ys[ty][tx] = in[y + ty * m + tx];
		Xs[tx][ty] = in[x + ty * m + tx];

		__syncthreads();

		for (k = 0; k<16; k++){
			tmp = Ys[ty][k] - Xs[k][tx];
			s += tmp * tmp;
		}
		__syncthreads();
	}
	o = by * 16 * n + ty * n + bx * 16 + tx;

	out[o] = exp(- s / (2 * pow(sigma, 2)))/n;
}

__global__ void maxValueAndArgsPerBlock(float* mat, float* target, int* idx_target) {

	__shared__ float max_vals[32];
    __shared__ unsigned int max_idxs[32];

    unsigned int width = gridDim.x;

    float cur_max = 1e-16;
    unsigned int cur_idx = 0;
    float val = 0;

    for (unsigned int i = threadIdx.x; i < blockIdx.x; i += 32) {
        val = abs(mat[blockIdx.x * width + i]);
        if (val > cur_max) {
            cur_max = val; cur_idx = i;
        }
    }
    max_vals[threadIdx.x] = cur_max;
    max_idxs[threadIdx.x] = cur_idx;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = 1e-9; cur_idx = 0;
        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max) {
                cur_max = max_vals[i];
                cur_idx = max_idxs[i];
            }
        target[blockIdx.x] = cur_max;
        idx_target[blockIdx.x] = cur_idx;
    }
}

__global__ void computeRotationArgs(float* GM, float* maxValues, int * maxArgs,
  RotationArgs* rArgs, unsigned int n){
	rArgs->value = 1e-12;
    for (size_t i = 0; i < n; i++) {
      float moment = fabsf(maxValues[i]);
      if(moment > rArgs->value){
        rArgs->value = moment;
        rArgs->p = i; rArgs->q = maxArgs[i];
      }
    }

    float phi = (GM[rArgs->q + rArgs->q * n]  - GM[rArgs->p + rArgs->p * n]);
    phi /= (2 * GM[rArgs->q + rArgs->p * n]);
    float t = phi == 0 ? 1 : (1 / (phi + (phi > 0 ? 1 : -1) * sqrt(phi * phi + 1)));
  	rArgs->_cos = 1/sqrtf(1 + t * t); rArgs->_sin = t/sqrtf(1 + t * t);
}

__global__ void jacobiRotationEigen(float *GM, RotationArgs *args, unsigned int col){

	__shared__ float s[2][128];

	unsigned int local = threadIdx.x;
	unsigned int index = local + blockIdx.x * blockDim.x;

	float _cos = args->_cos, _sin = args->_sin;

	int ip = args->p * col + index, iq = args->q * col + index;

	s[0][local] = GM[ip]; s[1][local] = GM[iq];

	__syncthreads();

	GM[ip] = s[0][local] * _cos - s[1][local] * _sin;
	GM[iq] = s[0][local] * _sin + s[1][local] * _cos;

	__syncthreads();

	ip = index * col + args->p; iq = index * col + args->q;

	s[0][local] = GM[ip]; s[1][local] = GM[iq];

	__syncthreads();

	GM[ip] = s[0][local] * _cos - s[1][local] * _sin;
	GM[iq] = s[0][local] * _sin + s[1][local] * _cos;

}

__global__ void computeEntropy(float* GM, unsigned int rows, float * entropy){
	entropy[0] = 0;
	for (int i = 0; i < rows; ++i) {
		float value = GM[i + i * rows];
		entropy[0] += value * logf(value);
	}

	entropy[0] = -entropy[0];
}


__global__ void hadamardProduct(float *GMX, float *GMY, float *GMXY){
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	GMXY[index] = GMX[index] * GMY[index];
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

		//////////////////////////// Normalizing X ////////////////////////////

		size_t sizex = sfloat * rows * xcols; float *Xdev, *momentsx;

		cudaMalloc((void **) &Xdev, sizex);
		cudaMalloc((void **) &momentsx, sfloat * blocks);

		cudaMemcpy(Xdev, X, sizex, cudaMemcpyHostToDevice);

		partialMoments<<<blocks, threads, sfloat * threads>>>(
				Xdev, momentsx, rows * xcols);

		composeMoments<<<1, 1>>>(momentsx, blocks, rows * xcols);

		standardNormalization<<<blocks, threads>>>(Xdev, momentsx, blocks / 2);


		//////////////////////////// Normalizing Y ////////////////////////////

		size_t sizey = sfloat * rows * ycols; float *Ydev, *momentsy;
		blocks = computeBlocks(threads, rows * ycols);

		cudaMalloc((void **) &Ydev, sizey);
		cudaMalloc((void **) &momentsy, sfloat * blocks);

		cudaMemcpy(Ydev, Y, sizey, cudaMemcpyHostToDevice);

		partialMoments<<<blocks, threads, sfloat * threads>>>(
				Ydev, momentsy, rows * ycols);

		composeMoments<<<1, 1>>>(momentsy, blocks, rows * ycols);

		standardNormalization<<<blocks, threads>>>(
				Ydev, momentsy, blocks / 2);

		//////////////////////////// Compute Gram Matrix ///////////////////

		float * gramX, * gramY, * gramXY; size_t sgram = sfloat * pow(rows, 2);

		cudaMalloc((void **) &gramX, sgram);
		cudaMalloc((void **) &gramY, sgram);
		cudaMalloc((void **) &gramXY, sgram);

		blocks = computeBlocks(threads, pow(rows, 2));

		dim3 block(16, 16);
		dim3 grid(ceil(rows/16), ceil(rows/16));

		fillGramMatrix<<<grid, block >>>(Xdev, gramX,
				getSigma(rows, xcols), rows, xcols);

		fillGramMatrix<<<grid, block >>>(Ydev, gramY,
				getSigma(rows, ycols), rows, ycols);

		blocks = computeBlocks(threads, rows * rows);
		hadamardProduct<<<blocks, threads>>>(gramX, gramY, gramXY);


		//////////////////////////// Compute Eigen Values & Entropy of X///////////////////

		const unsigned int MAXITER = 3 * 1e3;

		float* rowValues; int* indexes;  RotationArgs* args;
		float *entropyX;

		cudaMalloc((void **) &args, sizeof(RotationArgs));
		cudaMalloc((void **) &rowValues, rows * sfloat);
		cudaMalloc((void **) &indexes, rows * sizeof(int));
		cudaMalloc((void **) &entropyX, sfloat);

		blocks = computeBlocks(128, rows);


		for (int i = 0; i < MAXITER; ++i) {
			maxValueAndArgsPerBlock<<<rows, 32>>>(gramX, rowValues, indexes);
		    computeRotationArgs<<<1, 1>>>(gramX, rowValues, indexes, args, rows);
			jacobiRotationEigen<<<blocks, 128>>>(gramX, args, rows);
		}

		computeEntropy<<<1, 1>>>(gramX, rows, entropyX);


		//////////////////////////// Compute Eigen Values & Entropy of Y///////////////////

		float *entropyY;
		cudaMalloc((void **) &entropyY, sfloat);

		for (int i = 0; i < MAXITER; ++i) {
			maxValueAndArgsPerBlock<<<rows, 32>>>(gramY, rowValues, indexes);
		    computeRotationArgs<<<1, 1>>>(gramY, rowValues, indexes, args, rows);
			jacobiRotationEigen<<<blocks, 128>>>(gramY, args, rows);
		}

		computeEntropy<<<1, 1>>>(gramY, rows, entropyY);


		//////////////////////////// Compute Eigen Values & Entropy of XY///////////////////


		float *entropyXY;
		cudaMalloc((void **) &entropyXY, sfloat);

		for (int i = 0; i < MAXITER; ++i) {
			maxValueAndArgsPerBlock<<<rows, 32>>>(gramXY, rowValues, indexes);
		    computeRotationArgs<<<1, 1>>>(gramXY, rowValues, indexes, args, rows);
			jacobiRotationEigen<<<blocks, 128>>>(gramXY, args, rows);
		}

		computeEntropy<<<1, 1>>>(gramXY, rows, entropyXY);

	    cudaDeviceSynchronize();

	    float hx, hy, hxy;


	    cudaMemcpy(&hx, entropyX, sfloat, cudaMemcpyDeviceToHost);
	    cudaMemcpy(&hy, entropyY, sfloat, cudaMemcpyDeviceToHost);
	    cudaMemcpy(&hxy, entropyXY, sfloat, cudaMemcpyDeviceToHost);


	    cudaFree(rowValues); cudaFree(indexes);
		cudaFree(Xdev); cudaFree(momentsx);
		cudaFree(gramX); cudaFree(gramY); cudaFree(gramXY);

		cudaFree(entropyX); cudaFree(entropyY); cudaFree(entropyXY);

		cudaFree(Ydev); cudaFree(momentsy);

		std::cout << hxy << '\n';

		return hx + hy - hxy;
	}

	void checkAvailableDevices(){
		int deviceCount; cudaGetDeviceCount(&deviceCount);
		if(deviceCount == 0){
			fprintf(stderr, "Error: No devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}
	}

}


