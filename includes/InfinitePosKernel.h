/*
 * InfinitePosKernel.h
 *
 *  Created on: Feb 25, 2020
 *      Author: thoth
 */

#ifndef INFINITEPOSKERNEL_H_
#define INFINITEPOSKERNEL_H_

#include <vector>
#include <cmath>
#include "SymmetricMatrix.h"
#include "../kernel/Estimator.cuh"


float gaussiankenel(float distace, float sigma){
	return exp(- distace / (2 * pow(sigma, 2)));
}

template <class T>
class InfinitePosKernel {
private:
	bool parallel;
	void fillGramMatrix(float (*kernel)(float, float), float sigma,
			SymmetricMatrix& container, T& data);
	float getSigma(T& data);
	float computeShanon(SymmetricMatrix& gramMatrix);
	float jointShanon(SymmetricMatrix& A, SymmetricMatrix& B);
	float runSerial(T& X, T& Y);
	float runParellel(T& X, T& Y);

public:
	InfinitePosKernel(bool parallel);
	virtual ~InfinitePosKernel();
	float computeEntropy(T& data);
	float computeMutualInformation(T& X, T& Y);
};


template<class T>
inline void InfinitePosKernel<T>::fillGramMatrix(float (*kernel)(float, float),
		float sigma, SymmetricMatrix& container, T& data) {
	float accumulator, N = data.rows, cols = data.cols;
	for (int row = 0; row < N; ++row) {
		for(int dist = 0; dist < N; ++dist){ accumulator = 0;
			for (int col = 0; col < cols; ++col) {
				accumulator += pow(data(row, col) - data(dist, col), 2);
			}
			container[row][dist] = kernel(accumulator, sigma)/N;
		}
	}
}

template<class T>
inline float InfinitePosKernel<T>::getSigma(T& data) {
	float penalizer = -1.0/(4 + data.cols);
	float scottFactor = pow(data.rows, penalizer);
	return sqrt(2 * data.cols) * scottFactor;
}

template<class T>
inline InfinitePosKernel<T>::~InfinitePosKernel() {

}

template<class T>
float InfinitePosKernel<T>::computeShanon(SymmetricMatrix& gramMatrix) {
	float* eigenValues = gramMatrix.computeEigenValues(1e-4);
	float entropy = 0;
	for (int i = 0; i < gramMatrix.getSize(); ++i) {
		entropy += eigenValues[i] * log(eigenValues[i]);
	}
	delete [] eigenValues;
	return -entropy;
}

template<class T>
inline InfinitePosKernel<T>::InfinitePosKernel(bool parallel) : parallel(parallel) {
}

template<class T>
inline float InfinitePosKernel<T>::computeEntropy(T& data) {
	SymmetricMatrix gramMatrix(data.rows);
	data.standardNormalization();
	fillGramMatrix(gaussiankenel, getSigma(data), gramMatrix, data);
	return computeShanon(gramMatrix);
}

template<class T>
inline float InfinitePosKernel<T>::jointShanon(SymmetricMatrix& A,
		SymmetricMatrix& B) {
	size_t size = B.getSize(); 	SymmetricMatrix C(size);
	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			C[i][j] = A[i][j] * B[i][j] * size;
		}
	}
	return computeShanon(C);
}

template<class T>
inline float InfinitePosKernel<T>::runSerial(T& X, T& Y) {
	X.standardNormalization(); Y.standardNormalization();
	SymmetricMatrix A(X.rows); SymmetricMatrix B(Y.rows);
	fillGramMatrix(gaussiankenel, getSigma(X), A, X);
	fillGramMatrix(gaussiankenel, getSigma(Y), B, Y);
	float joint = jointShanon(A, B);
	return computeShanon(A) + computeShanon(B) - joint;
}

template<class T>
inline float InfinitePosKernel<T>::runParellel(T& X, T& Y) {
	return Estimator::computeInformationTheoryParallel(
			X.getAsOneDimArray(), Y.getAsOneDimArray(),
			X.rows, X.cols, Y.cols);
}

template<class T>
inline float InfinitePosKernel<T>::computeMutualInformation(T& X, T& Y) {
	if(parallel)
		return runParellel(X, Y);
	else
		return runSerial(X, Y);
}

#endif /* INFINITEPOSKERNEL_H_ */
