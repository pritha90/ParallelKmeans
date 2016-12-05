all:  
	nvcc -g -pg  -I. -DBLOCK_SHARED_MEM_OPTIMIZATION=0 --ptxas-options=-v -o timer.o -c timer.cu
	nvcc -g -pg  -I. -DBLOCK_SHARED_MEM_OPTIMIZATION=0 --ptxas-options=-v -o cuda_kmeans.o -c cuda_kmeans.cu
	nvcc -g -pg  -I. -DBLOCK_SHARED_MEM_OPTIMIZATION=0 --ptxas-options=-v -o test_driver.o -c test_driver.cu
	nvcc -g -pg -o test_driver test_driver.o timer.o cuda_kmeans.o	

