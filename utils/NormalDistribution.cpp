/*
 * NormalDistribution.cpp
 *
 *  Created on: Feb 23, 2020
 *      Author: thoth
 */

#include "NormalDistribution.h"
#include <random>

NormalDistribution::NormalDistribution(int rows, int cols,
		float mean, float std): Dataset(rows, cols){
	this->generate(mean, std);
}

NormalDistribution::~NormalDistribution() {
}

void NormalDistribution::generate(float mean, float var) {
	std::default_random_engine generator;
	std::normal_distribution<float> normal(mean, var);
	for (int row = 0; row < rows; ++row) {
		std::vector<float> numbers;
		for (int col = 0; col < cols; ++col) {
			numbers.push_back(normal(generator));
		}
		dataset.push_back(numbers);
	}

}


