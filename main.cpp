#include <iostream>
#include "utils/NormalDistribution.h"


int main() {


	NormalDistribution dataset(70000, 100, 3, 4);
	auto moments = dataset.getMoments();
	std::cout << '\n' << moments.mean << ' ' << moments.std;

	dataset.standardNormalization();
	moments = dataset.getMoments();
	std::cout << '\n' << moments.mean << ' ' << moments.std;

	return 0;
}
