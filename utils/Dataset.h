/*
 * Dataset.h
 *
 *  Created on: Feb 23, 2020
 *      Author: thoth
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <numeric>
#include <iostream>
#include <vector>
#include <cmath>

struct moments{
		float mean;
		float std;
};

class Dataset {

public:
	int rows, cols;
	Dataset(int rows, int cols);
	virtual ~Dataset();
	std::vector<std::vector<float>> dataset;
	float operator()(int  row, int col);
	void print();
	moments getMoments();
	void standardNormalization();
	float* getAsOneDimArray();

private:
	void updateMoments(float& mean, float& std);
};

#endif /* DATASET_H_ */
