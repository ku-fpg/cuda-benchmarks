#include <stdio.h>
#include <stdint.h>

const int TILE_DIM = 4;
const int BLOCK_ROWS = 4;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    uint64_t* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 4

__global__ void MatVecMulKernelV2(Matrix, Matrix, uint64_t* , uint64_t*);

void MatVecMulV2(Matrix A, uint64_t* B, uint64_t* C, int vecsize)
{
	Matrix At;
    At.height = A.height;
    At.width = A.width;
    At.elements =  (uint64_t*)malloc(At.width * At.height * sizeof(uint64_t));

    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(uint64_t);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);

    Matrix d_At;
	d_At.width = At.width; d_At.height = At.height;
	cudaMalloc(&d_At.elements, size);
	cudaMemcpy(d_At.elements, At.elements, size,
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
    dim3 dimGrid(vecsize / dimBlock.x, A.height / dimBlock.y);
    MatVecMulKernelV2<<<dimGrid, dimBlock>>>(d_A, d_At, d_B, d_C);



    // Read C from device memory
    cudaMemcpy(C , d_C, size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(At.elements, d_At.elements, size,
    		   cudaMemcpyDeviceToHost);

    for(int i = 0; i < A.height; i++){
    	for(int j = 0; j < A.width; j++)
    		printf("%llu ", A.elements[i*A.width + j]);
        printf("\n");
     }

     printf("\n");

     for(int i = 0; i < At.height; i++){
		 for(int j = 0; j < At.width; j++)
			 printf("%llu ", At.elements[i*At.width + j]);
		 printf("\n");
     }

     printf("\n");

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_At.elements);
    cudaFree(d_B);
    cudaFree(d_C);
}


__global__ void MatVecMulKernelV2(Matrix A, Matrix At, uint64_t* B, uint64_t*  C)
{

    uint64_t Cvalue = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;


    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS){
    	At.elements[x*width + (y+j)] = A.elements[(y+j)*width + x];
    }

    for (int e = 0; e < At.width; ++e)
        B[e+col] = (B[e+col] == 0) ? 0x00 : 0xffffffffffffffff;

    for (int e = 0; e < At.width; ++e)
        Cvalue = Cvalue xor ( At.elements[row * At.width + e] & B[e + col]);
    C[row + col] = Cvalue;
}

int main(){
  Matrix A;

  uint64_t *B;
  uint64_t *C;

  int w, h;
  w = 4;
  h = 4;

  uint64_t mat[4][4] =  {1,0,0,0,
                         1,0,0,0,
                         1,0,0,0,
                         1,0,0,0};

  A.height = h;
  A.width = w;
  A.elements =  &mat[0][0]; //(uint64_t*)malloc(A.width * A.height * sizeof(uint64_t));

  B = (uint64_t*)malloc(h * sizeof(uint64_t));

  C = (uint64_t*)malloc(h * sizeof(uint64_t));


  for(int i = 0; i < h; i++)
      B[i] = 1;

  MatVecMulV2(A,B,C, h);


  for(int i = 0; i < h; i++){
     printf("%llu ", B[i]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < h; i++){
     printf("%llu ", C[i]);
    printf("\n");
  }
  printf("\n");
  return 0;
}

