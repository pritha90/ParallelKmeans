#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#define MAX_ITERS 1000

void demalloc2D(float *** arr, int m){
    for (int i = 0; i < m; i++)
        free((*arr)[i]);
    free(*arr);
}

__host__ __device__ inline static
float compute_euclidean_distance_without_sqrt(int    dDimensions,
                    int    NInputs,
                    int    kClusters,
                    float *inputData,     
                    float *finalClusters,    
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    for (i = 0; i < dDimensions; i++) {
        ans += (inputData[NInputs * i + objectId] - finalClusters[kClusters * i + clusterId]) *
               (inputData[NInputs * i + objectId] - finalClusters[kClusters * i + clusterId]);
    }

    return(ans);
}

__global__ static
void find_nearest_cluster(int dDimensions,
                          int NInputs,
                          int kClusters,
                          float *inputData,       
                          float *deviceClusters,    
                          int *membership,          
			  int *membershipChanged)
{
    float *finalClusters = deviceClusters;

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < NInputs) {
        int   index, i;
        float dist, min_dist;

        index    = 0;
        min_dist = compute_euclidean_distance_without_sqrt(dDimensions, NInputs, kClusters,
                                 inputData, finalClusters, objectId, 0);

        for (i=1; i<kClusters; i++) {
            dist = compute_euclidean_distance_without_sqrt(dDimensions, NInputs, kClusters,
                                 inputData, finalClusters, objectId, i);
            if (dist < min_dist) {
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[objectId] = 1;
        }
        else  membershipChanged[objectId] = 0;

        membership[objectId] = index;

        __syncthreads();    

    }
}

int cuda_kmeans(float **inputData,      
                   int     dDimensions,    
                   int     NInputs,      
                   int     kClusters,  
                   float   threshold,    
                   int    *membership,   
		   float **finalClusters,
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize, *hostMembershipChanged; 
    float    delta;          
    float  **transposedInputData;
    float  **transposedClusters;
    float  **newClusters;   

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership, *deviceMembershipChanged;

    malloc2D(transposedInputData, dDimensions, NInputs);
    for (i = 0; i < dDimensions; i++) {
        for (j = 0; j < NInputs; j++) {
            transposedInputData[i][j] = inputData[j][i];
        }
    }

    malloc2D(transposedClusters, dDimensions, kClusters);
    for (i = 0; i < dDimensions; i++) {
        for (j = 0; j < kClusters; j++) {
            transposedClusters[i][j] = transposedInputData[i][j];
        }
    }

    for (i=0; i<NInputs; i++) membership[i] = -1;

    newClusterSize = (int*) calloc(kClusters, sizeof(int));
    hostMembershipChanged =  (int*) calloc(NInputs, sizeof(int));

    malloc2D(newClusters, dDimensions, kClusters);
    for (i = 0; i < dDimensions; i++) {
        for (j = 0; j < kClusters; j++) {
		newClusters[i][j] = 0.0;
    }
    }
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (NInputs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;


    checkCuda(cudaMalloc(&deviceObjects, NInputs*dDimensions*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, kClusters*dDimensions*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, NInputs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceMembershipChanged, NInputs*sizeof(int)));

    checkCuda(cudaMemcpy(deviceObjects, transposedInputData[0],
              NInputs*dDimensions*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              NInputs*sizeof(int), cudaMemcpyHostToDevice));

    do {
        checkCuda(cudaMemcpy(deviceClusters, transposedClusters[0],
                  kClusters*dDimensions*sizeof(float), cudaMemcpyHostToDevice));

        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock >>>
            (dDimensions, NInputs, kClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceMembershipChanged);

        cudaDeviceSynchronize(); checkLastCudaError();

	checkCuda(cudaMemcpy(hostMembershipChanged, deviceMembershipChanged,
                  sizeof(int)*NInputs, cudaMemcpyDeviceToHost));

        checkCuda(cudaMemcpy(membership, deviceMembership,
                  NInputs*sizeof(int), cudaMemcpyDeviceToHost));
	
	int myd = 0;
        for (i=0; i<NInputs; i++) {
            index = membership[i];
	    if(hostMembershipChanged[i] == 1)  myd++;	
            newClusterSize[index]++;
            for (j=0; j<dDimensions; j++)
                newClusters[j][index] += inputData[i][j];
        }
        delta = (float)myd;

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<kClusters; i++) {
            for (j=0; j<dDimensions; j++) {
                if (newClusterSize[i] > 0)
                    transposedClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= NInputs;
    } while (delta > threshold && loop++ < MAX_ITERS);

    *loop_iterations = loop + 1;

    for (i = 0; i < kClusters; i++) {
        for (j = 0; j < dDimensions; j++) {
            finalClusters[i][j] = transposedClusters[j][i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));

    free(transposedInputData[0]);
    free(transposedInputData);
    free(transposedClusters[0]);
    free(transposedClusters);
    free(newClusters[0]);
    free(newClusters);


/*    demalloc2D(&transposedInputData, dDimensions);
    demalloc2D(&transposedClusters, dDimensions);
    demalloc2D(&newClusters,dDimensions );*/
    free(newClusterSize);

    return 1;
}

