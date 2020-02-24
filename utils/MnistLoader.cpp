#include <fstream>
#include <assert.h>
#include "MnistLoader.h"


MnistLoader::MnistLoader(std::string image_file,
	std::string label_file,	int num) : Dataset(0, 0)
{
	m_size = 0;
	load_images(image_file, num);
	load_labels(label_file, num);
}

MnistLoader::MnistLoader(std::string image_file,
	std::string label_file) :
	MnistLoader(image_file, label_file, 0)
{
	// empty
}

MnistLoader::~MnistLoader()
{
	// empty
}

int MnistLoader::to_int(char* p)
{
	return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
		((p[2] & 0xff) << 8) | ((p[3] & 0xff) << 0);
}


void MnistLoader::load_images(std::string image_file, int num)
{	
	std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

	bool a = ifs.is_open();

	char p[4];

	ifs.read(p, 4);
	int magic_number = to_int(p);

	ifs.read(p, 4);
	m_size = to_int(p);

	if (num != 0 && num < m_size) m_size = num;

	ifs.read(p, 4);
	rows = to_int(p);

	ifs.read(p, 4);
	cols = to_int(p);

	char* q = new char[rows * cols];

	for (int i = 0; i < m_size; ++i) {
		ifs.read(q, rows * cols);
		std::vector<float> image(rows * cols);
		for (int j = 0; j < rows * rows; ++j) {
			image[j] = q[j] / 255.0;
		}
		dataset.push_back(image);
	}
	delete[] q;

	ifs.close();
}

void MnistLoader::load_labels(std::string label_file, int num)
{
	std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
	char p[4];

	ifs.read(p, 4);
	int magic_number = to_int(p);

	ifs.read(p, 4);
	int size = to_int(p);
	// limit
	if (num != 0 && num < m_size) size = num;

	for (int i = 0; i < size; ++i) {
		ifs.read(p, 1);
		int label = p[0];
		m_labels.push_back(label);
	}

	ifs.close();
}
