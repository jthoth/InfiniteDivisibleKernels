#include <iostream>

#include "utils/NormalDistribution.h"
#include "utils/MnistLoader.h"
#include "utils/StaticData.h"


#include "kernel/Estimator.cuh"
#include "includes/InfinitePosKernel.h"


int main() {

	/*MnistLoader mnist("./data/train-images-idx3-ubyte", 20);
	InfinitePosKernel<MnistLoader> estimator;*/




	NormalDistribution A(200, 20, 5, 3);
	NormalDistribution B(200, 15, 4, 3);

	InfinitePosKernel<NormalDistribution> estimator;
	std::cout << estimator.computeMutualInformation(A, B);


	return 0;
}
