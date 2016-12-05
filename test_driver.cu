#include <stdio.h>
#include <stdlib.h>
#include "sequential_kmeans.c"
#include "kmeans.h"

void random_data(float ** array, int m, int n) {
    for(int i = 0; i < m ; i++) {
	for(int j = 0; j < n; j++){
        array[i][j] = (float)rand()/(float)RAND_MAX;
    }
}
}

void allocate_random_cluster_centers_atfirst(float ** objects, float ** clusters, int N, int dimension, int k){
   
   /*for (i=1; i<k; i++)
        clusters[i] = clusters[i-1] + dimension;   */

}

void print_timing(char * type, double clustering_timing, int loop_iterations){
	printf("Type : %s\n", type);
	printf("Computation timing = %10.4f sec\n", clustering_timing);
        printf("Loop iterations    = %d\n", loop_iterations);
}

int main() {

int k = 128, dimension = 1000, N = 51200;
           int    *membership;    /* [N] */
           float **objects;       /* [N][dimension] data objects */
           float **clusters;      /* [k][dimension] cluster center */
           float   threshold;
           double  timing, clustering_timing;
           int     loop_iterations, i ,j;

    threshold        = 0.001;
    printf("N       = %d\n", N);
    printf("dimension     = %d\n", dimension);
    printf("k   = %d\n", k);
    printf("threshold     = %.4f\n", threshold);
    allocate_2D_memory(&objects, N, dimension);
    random_data(objects, N, dimension);

    timing            = wtime();
    clustering_timing = timing;

    membership = (int*) malloc(N * sizeof(int));
    allocate_2D_memory(&clusters, k, dimension);
    //sequential_kmeans(objects, dimension, N, k, threshold, membership, clusters, &loop_iterations);

    timing            = wtime();
    clustering_timing = timing - clustering_timing;
    print_timing("Sequential", clustering_timing, loop_iterations);

    free(membership);
    membership = (int*) malloc(N * sizeof(int));
    loop_iterations = 0;

    timing            = wtime();
    clustering_timing = timing;

    cuda_kmeans(objects, dimension, N, k, threshold,
                          membership, clusters, &loop_iterations);

    timing            = wtime();
    clustering_timing = timing - clustering_timing;
    print_timing("Parallel", clustering_timing, loop_iterations);

    free(objects[0]);
    free(objects);
    free(membership);
    free(clusters[0]);
    free(clusters);
return 0;
}
