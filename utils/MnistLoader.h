/*

Based on: Arpaka implementation
Source: https://github.com/arpaka/mnist-loader

*/

#include <string>
#include "Dataset.h"

class MnistLoader : public Dataset {

private:

	std::vector<int> m_labels;
	int m_size;

	void load_images(std::string file, int num = 0);
	void load_labels(std::string file, int num = 0);
	int  to_int(char* p);

public:
	MnistLoader(std::string image_file, std::string label_file, int num);
	MnistLoader(std::string image_file, std::string label_file);
	~MnistLoader();


	int size() { return m_size; }

	std::vector<float> images(int id) { return dataset[id]; }
	int labels(int id) { return m_labels[id]; }
};
