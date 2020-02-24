#include <string>
#include "CpuVsGpu.h"
#include <iostream>
#include <sstream>
#include <chrono>

CpuVsGpu::CpuVsGpu(const std::string& file) {
	this->connection.open(file.c_str(), std::ios::out);
	this->connection << "device,x,y,seconds\n";
}

CpuVsGpu::~CpuVsGpu(){
}

