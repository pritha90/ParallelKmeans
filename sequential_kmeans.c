#include <stdio.h>
#include <stdlib.h>

#define MAX_ITERS 1000

float compute_euclidean_distance_without_sqrt(int    numDimensions,  
                    float *firstVector,   
                    float *secVector)   
{
    int i;
    float ans=0.0;

    for (i=0; i<numDimensions; i++)
        ans += (firstVector[i]-secVector[i]) * (firstVector[i]-secVector[i]);

    return(ans);
}

int find_nearest_cluster(int     numClusters, 
                         int     numCoords,
                         float  *inputDataPoint,
                         float **clusterCenters)
{
    int   index, i;
    float dist, min_dist;

    index    = 0;
    min_dist = compute_euclidean_distance_without_sqrt(numCoords, inputDataPoint, clusterCenters[0]);

    for (i=1; i<numClusters; i++) {
        dist = compute_euclidean_distance_without_sqrt(numCoords, inputDataPoint, clusterCenters[i]);
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

int sequential_kmeans(float **inputData,
               int     numCoords,
               int     numObjs,
               int     numClusters,
               float   threshold,
               int    *belongsToCluster, float **clusterCenters, int *loop_iterations)
{
    int      i, j, index, iterations=0;
    int     *numPointsInEachCluster;
    
    float    delta;
    float  **newClusterCenters;

    for (i=0; i<numObjs; i++) belongsToCluster[i] = -1;

    for (i=0; i<numClusters; i++)
            for (j=0; j<numCoords; j++)
                clusterCenters[i][j] = inputData[i][j];

    numPointsInEachCluster = (int*) calloc(numClusters, sizeof(int));

    newClusterCenters    = (float**) malloc(numClusters * sizeof(float*));
    newClusterCenters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    for (i=1; i<numClusters; i++)
        newClusterCenters[i] = newClusterCenters[i-1] + numCoords;

    do {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            index = find_nearest_cluster(numClusters, numCoords, inputData[i],
                                         clusterCenters);

            if (belongsToCluster[i] != index) delta += 1.0;

            belongsToCluster[i] = index;

            numPointsInEachCluster[index]++;
            for (j=0; j<numCoords; j++)
                newClusterCenters[index][j] += inputData[i][j]; //updating the cluster center of the "index" cluster
        }
        //calulate new cluster centers and reset newClusterCenters
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (numPointsInEachCluster[i] > 0)
                    clusterCenters[i][j] = newClusterCenters[i][j] / numPointsInEachCluster[i];
                newClusterCenters[i][j] = 0.0;   
            }
            numPointsInEachCluster[i] = 0;   
        }
            
        delta /= numObjs;
    } while (delta > threshold && iterations++ < MAX_ITERS); 
    *loop_iterations = iterations + 1;

    free(newClusterCenters[0]);
    free(newClusterCenters);
    free(numPointsInEachCluster);

    return 1;
}


