/*
 * SymmetricMatrix.cpp
 *
 *  Created on: Mar 2, 2020
 *      Author: thoth
 */

#include "SymmetricMatrix.h"
#include <cmath>

SymmetricMatrix::SymmetricMatrix():size(0), matrix(nullptr) {
}

SymmetricMatrix::SymmetricMatrix(size_t size): size(size),
		matrix(new float[(size - 1) * size / 2 + size]){

}

SymmetricMatrix::~SymmetricMatrix() {
	delete [] matrix;
}

SymmetricMatrix::Row SymmetricMatrix::operator[](size_t row)
{
    return Row(*this, row);
}

float& SymmetricMatrix::Row::operator[](size_t col)
{
    size_t r = std::max(row, col);
    size_t c = std::min(row, col);
    return matrix.matrix[(r + 1) * r / 2 + c];
}

float* SymmetricMatrix::computeEigenValues(float epsilon) {
	SymmetricMatrix* A = this;

	while(true){

		maximunArgs _max = this->computeMaxValueLower(*A);
		rotationArgs rot = computeRotationValues(*A, _max);

		if(_max.pq < epsilon)
			break;

		float* C = new float[size * size];
		rotate(*A, C, _max, rot);

		update(*A, C, _max, rot);

		delete [] C;
	}
	return A->diagonal(*A);
}

maximunArgs SymmetricMatrix::computeMaxValueLower(SymmetricMatrix& instance) {

	size_t p = 0, q = 1;  float Apq = std::abs(instance[p][q]);
    for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < i; j++){
        	if (std::abs(instance[i][j]) > Apq){
                Apq = std::abs(instance[i][j]);
                p = i; q = j;
            }
        }
    }
	return {Apq, p, q};
}

rotationArgs SymmetricMatrix::computeRotationValues(SymmetricMatrix& instance,
		maximunArgs& maxArgs) {

	float phi = (instance[maxArgs.q][maxArgs.q] - instance[maxArgs.p][maxArgs.p]);
	phi /= 2 * instance[maxArgs.p][maxArgs.q];
	float t = phi == 0 ? 1 : (1 / (phi + (phi > 0 ? 1 : -1) * sqrt(phi * phi + 1)));
	float cosine = 1 / sqrt(1 + t * t), sine = t / sqrt(1 + t * t);
	return {cosine, sine};

}

void SymmetricMatrix::rotate(SymmetricMatrix& A, float*& C,
		maximunArgs& _max, rotationArgs& rot) {
    for (size_t i = 0; i < size; i++){
    	for (size_t j = 0; j < size; j++){
            if (i == _max.p)
                C[i * size + j] = A[_max.p][j] * rot.cosine - A[_max.q][j] * rot.sine;
            else if (i == _max.q)
                C[i * size + j] = A[_max.p][j] * rot.sine + A[_max.q][j] * rot.cosine;
            else
                C[i * size + j] = A[i][j];
        }
    }
}

void SymmetricMatrix::update(SymmetricMatrix& A, float*& C,
		maximunArgs& _max, rotationArgs& rot) {
    for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j <= i; j++){
            if (j == _max.p)
                A[i][_max.p] = C[i * size + _max.p] * rot.cosine - C[i * size + _max.q] * rot.sine;
            else if (j == _max.q)
                A[i][_max.q] = C[i * size + _max.p] *  rot.sine + C[i * size + _max.q] * rot.cosine ;
            else
                A[i][j] = C[i * size + j];
        }
    }

}

void SymmetricMatrix::print(SymmetricMatrix& instance) {
	for (size_t i = 0; i < size; ++i) {  std::cout<< '\n';
		for (size_t j = 0; j < size; ++j) {
			std::cout<< instance[i][j] << '\t';
		}
	}
}

float* SymmetricMatrix::diagonal(SymmetricMatrix& instance) {
	float* diagonal = new float[size];
	for (size_t i = 0; i < size; ++i) {
		diagonal[i] = instance[i][i];
	}
	return diagonal;
}

size_t SymmetricMatrix::getSize() {
	return size;
}
