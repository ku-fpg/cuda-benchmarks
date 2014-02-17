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

// Forward declaration of the matrix multiplication kernel
__global__ void MatVecMulKernel(Matrix, uint64_t* , uint64_t*);
__global__ void transposeNaive(Matrix, Matrix);


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
    dim3 dimGrid(vecsize / dimBlock.x, A.height / dimBlock.y);
    MatVecMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C , d_C, size,
               cudaMemcpyDeviceToHost);

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

    for (int e = 0; e < A.width; ++e)
        B[e+col] = (B[e+col] == 0) ? 0x00 : 0xffffffffffffffff;     

    for (int e = 0; e < A.width; ++e)
        Cvalue = Cvalue xor ( A.elements[row * A.width + e] & B[e + col]);
    C[row + col] = Cvalue;
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
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);
    transposeNaive<<<dimGrid, dimBlock>>>(d_C, d_A);

    // Read C from device memory
    cudaMemcpy(C.elements , d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
}

//Modified Cuda-code sample from https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
__global__ void transposeNaive(Matrix odata, Matrix idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata.elements[x*width + (y+j)] = idata.elements[(y+j)*width + x];
}

int main(){
  Matrix A;
  Matrix D;
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

  D.height = h;
  D.width = w;
  D.elements =  (uint64_t*)malloc(D.width * D.height * sizeof(uint64_t));  

  for(int i = 0; i < h; i++)
      B[i] = 1;

  Trans(A,D);  

  MatVecMul(D,B,C, h);
  
  for(int i = 0; i < A.height; i++){
    for(int j = 0; j < A.width; j++)
      printf("%llu ", A.elements[i*A.width + j]);
    printf("\n");
  }
  
  printf("\n");

  for(int i = 0; i < D.height; i++){
    for(int j = 0; j < D.width; j++)
      printf("%llu ", D.elements[i*D.width + j]);
    printf("\n");
  }

  printf("\n");
  
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
