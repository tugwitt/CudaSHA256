// cd /home/hork/cuda-workspace/CudaSHA256/Debug/files
// time ~/Dropbox/FIIT/APS/Projekt/CpuSHA256/a.out -f ../file-list
// time ../CudaSHA256 -f ../file-list


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>
#include <ctype.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <string>
#include <cassert>
#include <cstring>


__global__ void sha256_cuda_new(JOB * job) {

	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, job->data, job->size);
	sha256_final(&ctx, job->digest);
}

__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}


void print_hash(unsigned char* sha256) {
	for (uint8_t i = 0; i < 32; ++i) {
		std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(sha256[i]);
	}
	std::cout << std::dec << std::endl;
}

long getMicrotime(){
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}


void pre_sha256() {
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


JOB * JOB_init(BYTE * data, long size) {
	JOB * j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	return j;
}

void runJobs(JOB ** jobs, int n){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}


void run_sha(unsigned char test[], int n, char* string) {

	JOB ** jobs;
	BYTE * buffer = 0;
	unsigned long fsize = strlen((char*)test);

	checkCudaErrors(cudaMallocManaged(&buffer, (fsize+1)*sizeof(char)));
	checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));
	memcpy(buffer, test, fsize); 

	for (int i = 0; i < n;i++){
		jobs[i] = JOB_init(buffer, fsize);
	}
	
	runJobs(jobs, n);

	unsigned char* found_hash;
	memcpy(found_hash, jobs[0]->digest, 32);
	print_hash(found_hash);

}




int main() {

	cudaDeviceSynchronize();

	unsigned char test[] = "test\n";

	char string[65];

	long start = getMicrotime();

	pre_sha256();

	run_sha(test, 1, string);

	long diff = getMicrotime() - start;

	printf("%ld", diff);

	cudaDeviceReset();


	return 0;
}
