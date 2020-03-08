#include <iostream>

#include "utils/NormalDistribution.h"
#include "utils/MnistLoader.h"
#include "utils/StaticData.h"

#include <iomanip>
#include "includes/InfinitePosKernel.h"


int main() {

	/*MnistLoader mnistA("./data/train-images-idx3-ubyte", 1308);
	MnistLoader mnistB("./data/train-images-idx3-ubyte", 1308);*/

	/*InfinitePosKernel<MnistLoader> estimator(true);
	estimator.computeMutualInformation(mnistA, mnistA);*/


    std::cout << std::fixed;
    std::cout << std::setprecision(3);

	NormalDistribution A(32, 32, 5, 3);
	NormalDistribution B(32, 32, 4, 1);

	InfinitePosKernel<NormalDistribution> estimator(true);
	estimator.computeMutualInformation(A, B);

	/*StaticData A;
	StaticData B;

	A.fillWith(1.0, 32, 32);


	InfinitePosKernel<StaticData> estimator(true);
	estimator.computeMutualInformation(A, B);*/

	auto mome = A.getMoments();
	std::cout << "\n\n A \t Media : " << mome.mean << "\t Std : " << mome.std ;
	mome = A.getMoments();
	std::cout << "\n\n B \t Media : " << mome.mean << "\t Std : " << mome.std ;

	return 0;
}
