#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Estimator {
	void compute(bool* data, bool* gram, int x, int y);
}