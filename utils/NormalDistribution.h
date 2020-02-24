/*
 * NormalDistribution.h
 *
 *  Created on: Feb 23, 2020
 *      Author: thoth
 */

#ifndef NORMALDISTRIBUTION_H_
#define NORMALDISTRIBUTION_H_

#include "Dataset.h"

class NormalDistribution : public Dataset {

public:
	NormalDistribution(int rows, int cols, float mean, float var);
	virtual ~NormalDistribution();

private:
	void generate(float mean, float std);
};

#endif /* NORMALDISTRIBUTION_H_ */
