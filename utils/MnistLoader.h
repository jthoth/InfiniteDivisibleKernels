#include <string>
#include "Dataset.h"

class MnistLoader : public Dataset {

private:
	int image_rows, image_cols;
	void load(std::string file, int num = 0);
	void load_images(std::ifstream& file, int num = 0);
	int reverseInt (int i);
	void extractGlobalInformation(std::ifstream& file);
	void fillDataset(std::ifstream& file);
	void updateRows(int num);

public:
	MnistLoader(std::string image_file, int num);
	MnistLoader(std::string image_file);
	~MnistLoader();
	int size() { return rows; }
	std::vector<float> images(int id) { return dataset[id]; }
};
