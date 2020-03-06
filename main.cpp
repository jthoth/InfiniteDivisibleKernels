#include <iostream>

#include "utils/NormalDistribution.h"
#include "utils/MnistLoader.h"
#include "utils/StaticData.h"


#include "includes/InfinitePosKernel.h"


int main() {

	MnistLoader mnistA("./data/train-images-idx3-ubyte", 1308);
	MnistLoader mnistB("./data/train-images-idx3-ubyte", 1308);

	InfinitePosKernel<MnistLoader> estimator(true);
	estimator.computeMutualInformation(mnistA, mnistA);



	/*NormalDistribution A(1000, 10, 5, 3);
	NormalDistribution B(1000, 12, 4, 1);

	InfinitePosKernel<NormalDistribution> estimator(true);
	estimator.computeMutualInformation(A, B);*/

/*	StaticData A;
	StaticData B;


	InfinitePosKernel<StaticData> estimator(true);
	estimator.computeMutualInformation(A, B);*/

	auto mome = mnistA.getMoments();
	std::cout << "\n\n A \t Media : " << mome.mean << "\t Std : " << mome.std ;
	mome = mnistB.getMoments();
	std::cout << "\n\n B \t Media : " << mome.mean << "\t Std : " << mome.std ;

	return 0;
}
