#include <iostream>

#include "utils/NormalDistribution.h"
#include "utils/MnistLoader.h"
#include "utils/StaticData.h"

#include <iomanip>
#include "includes/InfinitePosKernel.h"


int main() {

	MnistLoader mnistA("./data/train-images-idx3-ubyte", 256);
	MnistLoader mnistB("./data/train-images-idx3-ubyte", 256);

	InfinitePosKernel<MnistLoader> estimator(true);

	std::cout << estimator.computeMutualInformation(mnistA, mnistA);

	return 0;
}
