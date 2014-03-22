#include <stdio.h>
#include <stdint.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
#define BLOCK_SIZE 8

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    uint64_t* elements;
} Matrix;




// Forward declaration of the matrix multiplication kernel
__global__ void MatVecMulKernel(Matrix, uint64_t* , uint64_t*);
__global__ void transposeNaive(Matrix, Matrix, int, int);


// Matrix-Vector multiplication - Host code

void MatVecMul(Matrix A, uint64_t* B, uint64_t* C, int vecsize)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(uint64_t);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    uint64_t *d_B;
    size = vecsize * sizeof(uint64_t);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    uint64_t *d_C;
    size = vecsize * sizeof(uint64_t);
    cudaMalloc(&d_C, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((1 + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);

    MatVecMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C , d_C, size,
               cudaMemcpyDeviceToHost);
    /*
    for(int i = 0; i < A.height; i++){
    	     printf("%llu ", C[i]);
    	    printf("\n");
    	  }
    	  printf("\n");
    */
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B);
    cudaFree(d_C);
}

// MV multiplication kernel called by MatMul()
__global__ void MatVecMulKernel(Matrix A, uint64_t* B, uint64_t*  C)
{

    uint64_t Cvalue = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < A.width; ++e){
        B[e+col] = (B[e+col] == 0) ? 0x00 : 0xffffffffffffffff;
    }



    if(row > A.height || col > A.width) return;
    for (int e = 0; e < A.width; ++e)
        Cvalue = Cvalue ^ (A.elements[row * A.width + e] & B[e + col]);
    C[row  + col] = Cvalue;
}

void Trans(Matrix A, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(uint64_t);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
           cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    //size = C.width * C.height * sizeof(uint64_t);
    size = C.width * C.height * sizeof(uint64_t);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimGrid(A.width/TILE_DIM, A.height/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    //transposeNaive<<<dimGrid, dimBlock>>>(d_C, d_A, A.width, A.height );

    // Read C from device memory
    cudaMemcpy(C.elements , d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
}

__global__ void transposeNaive(Matrix odata, Matrix idata, int width, int height)
{
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

   if (xIndex < width && yIndex < height)
   {
       unsigned int index_in  = xIndex + width * yIndex;
       unsigned int index_out = yIndex + height * xIndex;
       odata.elements[index_out] = idata.elements[index_in];
   }
}


int main() {

	Matrix A;


	uint64_t *B;
	uint64_t *C;

	int w, h;
	w = 384;
	h = 1040;

	uint64_t mat[1040][384];

	FILE *myfile;

	uint64_t myvariable;

	int i;
	int j;

	//Read in G.m example matrix from repo
	myfile=fopen("./src/mat.txt", "r");

	for(i = 0; i < 1040; i++){
	  for (j = 0 ; j < 384; j++)
	  {
		 fscanf(myfile,"%llu",&myvariable);
		 mat[i][j] = myvariable;
		 //printf ("%llu", mat[i][j]);
      }
	  //printf ("\n");
    }
    fclose(myfile);


	A.height = h;
	A.width = w;
	A.elements = &mat[0][0]; //(uint64_t*)malloc(A.width * A.height * sizeof(uint64_t));

	B = (uint64_t*) malloc(h * sizeof(uint64_t));
	for(int i = 0; i < h; i++){
	      B[i] = 2;//rand() % 2;
	}

	C = (uint64_t*) malloc(h * sizeof(uint64_t));
    /*
	D.height = w;
	D.width = h;
	D.elements =  (uint64_t*)malloc(D.width * D.height * sizeof(uint64_t));

	We do not need to transpose because we have index access to columns Trans(A,D);
    */

	MatVecMul(A,B,C, h);
      /*
	  for(int i = 0; i < h; i++){
	    for(int j = 0; j < w; j++)
	      printf("%llu ", A.elements[i*A.width + j]);
	    printf("\n");
	  }

	  printf("\n");

	  for(int i = 0; i < h; i++){
	     printf("%llu ", B[i]);
	    printf("\n");
	  }
	  printf("\n");
      */

	for(int i = 0; i < A.height; i++){
	   printf("%llu ", C[i]);
	   printf("\n");
	}
	printf("\n");

	return 0;
}

