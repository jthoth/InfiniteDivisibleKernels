/*
 * StaticData.cpp
 *
 *  Created on: Feb 25, 2020
 *      Author: thoth
 */

#include "StaticData.h"

StaticData::StaticData(): Dataset(6, 5) {
	dataset = {
			{4, 4, 2, 8, 1},
			{9.7, 1, 3, 7,1},
			{6, 9 ,4.4, 6, 0},
			{9, 0, 1, 3.5, 1},
			{3, 2, 6, 5, 0.0},
			{0, 4, 20, 9,1},
			  };

}

StaticData::~StaticData() {

}

