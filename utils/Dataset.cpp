/*
 * Dataset.cpp
 *
 *  Created on: Feb 23, 2020
 *      Author: thoth
 */

#include "Dataset.h"

Dataset::Dataset(int rows, int cols):
rows(rows), cols(cols){
}

Dataset::~Dataset() {
	dataset.erase(dataset.begin(), dataset.end());
}

float Dataset::operator ()(int row, int col) {
	return dataset[row][col];
}

void Dataset::print() {
	for(auto row: dataset){ std::cout<< '\n';
		for(float item:row){
			std::cout<< item << '\t';
		}
	}
}

void Dataset::standardNormalization() {
	auto moment = getMoments();
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			dataset[row][col] = (dataset[row][col] - moment.mean)/moment.std;
		}
	}
}

float* Dataset::getAsOneDimArray() {
	float* data = new float [rows * cols];
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			data[j + i * cols] = dataset[i][j];
		}
	}
	return data;
}

void Dataset::updateMoments(float& mean, float& std) {
	for (auto row : dataset) {
		for (float item : row) {
			mean += item;
			std += pow(item, 2);
		}
	}
}

moments Dataset::getMoments() {
	float mean=0, std=0, N=(rows * cols);
	this->updateMoments(mean, std);
	mean = mean/N; std = sqrt(std/N - pow(mean, 2));
	return {mean, std};
}
