#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

namespace Estimator {

	float computeInformationTheoryParallel(float *X, float *Y,
			int rows, int xcols, int ycols);

	void checkAvailableDevices();
	unsigned int computeBlocks(int threads, int n);


}
