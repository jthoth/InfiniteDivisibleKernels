#pragma once
#include <fstream>

class CpuVsGpu {
private:
	std::fstream connection;

public:
	explicit CpuVsGpu(const std::string& fileOutput);
	void operator()(int xCells, int yCells);

	~CpuVsGpu();
};



