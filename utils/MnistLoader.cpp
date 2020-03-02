#include <fstream>
#include <assert.h>
#include "MnistLoader.h"


MnistLoader::MnistLoader(std::string file, int num) : Dataset(0, 0)
{
	load(file, num);
}

MnistLoader::MnistLoader(std::string file): Dataset(0, 0) {
	load(file, 0);
}

MnistLoader::~MnistLoader()
{
	// empty
}

int MnistLoader::reverseInt (int i)
{
	unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void MnistLoader::extractGlobalInformation(std::ifstream& file) {
	file.read((char*) (&rows), sizeof(rows)), rows = reverseInt(rows);
	file.read((char*) (&image_rows), sizeof(image_rows)), image_rows =
			reverseInt(image_rows);
	file.read((char*) (&image_cols), sizeof(image_cols)), image_cols =
			reverseInt(image_cols);
}

void MnistLoader::fillDataset(std::ifstream& file) {
	unsigned char* buffer = new unsigned char[cols];
	for (int i = 0; i < rows; ++i) {
		file.read((char*) (buffer), cols);
		std::vector<float> values(buffer, buffer + cols);
		dataset.push_back(values);
	}
	file.close();
	delete[] buffer;
}

void MnistLoader::load_images(std::ifstream& file, int num)
{	int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if(magic_number != 2051)
    	throw std::runtime_error("Invalid MNIST image file!");
	extractGlobalInformation(file);
	cols = image_rows * image_cols;
	updateRows(num);
	fillDataset(file);
}


void MnistLoader::load(std::string image_file, int num)
{
	std::ifstream file(image_file.c_str(), std::ios::binary);
	if(file.is_open()){
		load_images(file, num);

	}else{
		throw std::runtime_error("Cannot open file `" + image_file + "`!");
	}
}

void MnistLoader::updateRows(int num) {
	if(num != 0){
		rows = num;
	}
}
