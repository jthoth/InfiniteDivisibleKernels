/*
 * StaticData.h
 *
 *  Created on: Feb 25, 2020
 *      Author: thoth
 */

#ifndef STATICDATA_H_
#define STATICDATA_H_

#include "Dataset.h"

class StaticData: public Dataset {
public:
	StaticData();
	virtual ~StaticData();
	void fillWith(float value, int rows, int cols);
};

#endif /* STATICDATA_H_ */
