#include <iostream>

#include "utils/NormalDistribution.h"
#include "utils/MnistLoader.h"
#include "utils/StaticData.h"


#include "includes/InfinitePosKernel.h"


int main() {

	/*MnistLoader mnist("./data/train-images-idx3-ubyte", 20);
	InfinitePosKernel<MnistLoader> estimator;*/




	NormalDistribution A(100, 15, 5, 3);
	NormalDistribution B(100, 13, 4, 1);

	InfinitePosKernel<NormalDistribution> estimator(true);
	estimator.computeMutualInformation(A, B);

/*	StaticData A;
	StaticData B;


	InfinitePosKernel<StaticData> estimator(true);
	estimator.computeMutualInformation(A, B);*/

	auto mome = A.getMoments();
	std::cout << "\n\n A \t Media : " << mome.mean << "\t Std : " << mome.std ;
	mome = B.getMoments();
	std::cout << "\n\n B \t Media : " << mome.mean << "\t Std : " << mome.std ;

	return 0;
}
