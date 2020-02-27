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


__global__ void sha256_cuda(JOB * job) {

	SHA256_CTX ctx;
	sha256_init(&ctx);
	sha256_update(&ctx, job->data, job->size);
	sha256_final(&ctx, job->digest);
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

void run_sha(unsigned char test[]) {

	JOB * job;
	BYTE * buffer = 0;
	unsigned long fsize = strlen((char*)test);

	checkCudaErrors(cudaMallocManaged(&buffer, (fsize+1)*sizeof(char)));
	
	memcpy(buffer, test, fsize);  
	job = JOB_init(buffer, fsize);

	pre_sha256();

	int blockSize = 4;
	int numBlocks = (1 + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (job);

	cudaDeviceSynchronize();
	printf("%s\n", hash_to_string(job->digest));
	cudaDeviceReset();

}




int main() {

	unsigned char test[] = "test\n";

	run_sha(test);

	return 0;
}
