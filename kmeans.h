#define malloc2D(name, xDim, yDim) do {               \
    name = (float **)malloc(xDim * sizeof(float *));          \
    name[0] = (float *)malloc(xDim * yDim * sizeof(float));   \
    for (int i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        printf("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}

int cuda_kmeans( float ** dimObjects, int numCoords, int numObjs, int numClusters, float threshold,
                          int * membership, float ** clusters, int * loop_iterations);
double  wtime(void);

inline void allocate_2D_memory(float *** arr, int m, int n){
    *arr = (float**)malloc(m*sizeof(float*));
  for(int i=0; i<m; i++)
    (*arr)[i] = (float*)malloc(n*sizeof(float));
}

