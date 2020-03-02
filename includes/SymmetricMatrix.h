#ifndef SYMMETRICMATRIX_H_
#define SYMMETRICMATRIX_H_

#include <iostream>

struct maximunArgs{
	float pq;
	size_t p;
	size_t q;
};

struct rotationArgs {
	float cosine;
	float sine;
};

class SymmetricMatrix {
public:
	SymmetricMatrix();
	virtual ~SymmetricMatrix();
    SymmetricMatrix(size_t size);

    class Row
    {
        friend class SymmetricMatrix;
        private:
            SymmetricMatrix& matrix;
            size_t row;

            Row(SymmetricMatrix& mat, size_t row):
            	matrix(mat), row(row) {}

        public:
            float& operator[](size_t index);
    };

    Row operator[](size_t index);

    float* computeEigenValues(float epsilon);
    size_t getSize();
    void print(SymmetricMatrix& instance);

private:
    const size_t size; float* matrix;
    maximunArgs computeMaxValueLower(SymmetricMatrix& instance);
    rotationArgs computeRotationValues(SymmetricMatrix& instance,
    		maximunArgs& maxArgs);
    void rotate(SymmetricMatrix& instance, float*& C, maximunArgs& maxArgs, rotationArgs& rot);
    void update(SymmetricMatrix& instance, float*& C, maximunArgs& maxArgs, rotationArgs& rot);
    float* diagonal(SymmetricMatrix& instance);
};

#endif /* SYMMETRICMATRIX_H_ */
