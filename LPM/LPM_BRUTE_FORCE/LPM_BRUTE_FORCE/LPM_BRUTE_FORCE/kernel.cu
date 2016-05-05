
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <limits>


#include <string.h>
#include <time.h>



void read_data(char * path, char** comp, char** prov, unsigned char* subnet_b1, unsigned char * subnet_b2, unsigned char * subnet_b3, unsigned char * subnet_b4, unsigned char * subnet_pref, int total);
void read_data(char *);
void save_test_results(char * device, char * methodname, int * blks, int * threads, int* inputlen, int * searchedIp, float * buildtime, float * exectime, int size);

__device__ int isIpMatchingSubnet_1(int subnetmasks_b1, int subnetmasks_b2, int subnetmasks_b3, int subnetmasks_b4, int prefixLength, int ip_b1, int ip_b2, int ip_b3, int ip_b4)
{
	if (prefixLength >= 8)
	{
		if (prefixLength == 8 && subnetmasks_b1 == ip_b1)
			return 0;
		if (subnetmasks_b1 > ip_b1)
			return -1;
		if (subnetmasks_b1 < ip_b1)
			return 1;
		if (prefixLength >= 16)
		{
			if (prefixLength == 16 && subnetmasks_b2 == ip_b2)
				return 0;
			if (subnetmasks_b2 > ip_b2)
				return -1;
			if (subnetmasks_b2 < ip_b2)
				return 1;
			if (prefixLength >= 24)
			{
				if (prefixLength == 24 && subnetmasks_b3 == ip_b3)
					return 0;
				if (subnetmasks_b3 > ip_b3)
					return -1;
				if (subnetmasks_b3 < ip_b2)
					return 1;

				if (prefixLength == 32 && subnetmasks_b4 == ip_b4)
					return 0;
				else if (subnetmasks_b4 <= ip_b4)
					return 0;
				else
					return -1;
			}
			else if (subnetmasks_b3 <= ip_b3)
				return 0;
			else
				return -1;
		}
		else if (subnetmasks_b2 <= ip_b2)
			return 0;
		else
			return -1;
	}
	else if (subnetmasks_b1 <= ip_b1)
		return 0;
	else
		return -1;
}

__device__ int isIpMatchingSubnet(unsigned char subnetmasks_b1, unsigned char subnetmasks_b2, unsigned char subnetmasks_b3, unsigned char subnetmasks_b4, unsigned char prefixLength, unsigned char ip_b1, unsigned char ip_b2, unsigned char ip_b3, unsigned char ip_b4)
{
	unsigned char maskValues[8] = { 255, 254, 252, 248, 240, 224, 192, 128 };
	int diff = 0;
	if (prefixLength > 8)
	{
		if (subnetmasks_b1 != ip_b1)
			return -1;
		
		if (prefixLength > 16)
		{
			if (subnetmasks_b2 != ip_b2)
				return -1;
			
			if (prefixLength > 24)
			{
				if (subnetmasks_b3 != ip_b3)
					return -1;
				
				diff = 32 - prefixLength;
				unsigned char res = ip_b4 & maskValues[diff];
				if (res == subnetmasks_b4)
					return 0;
				return -1;
				
			}
			else
			{
				diff = 24 - prefixLength;
				unsigned char res = ip_b3 & maskValues[diff];
				if (res == subnetmasks_b3)
					return 0;
				return -1;
			}
			
		}
		else
		{
			diff = 16 - prefixLength;
			unsigned char res = ip_b2 & maskValues[diff];
			if (res == subnetmasks_b2)
				return 0;
			return -1;
		}
	}
	else
	{
		diff = 8 - prefixLength;
		unsigned char res = ip_b1 & maskValues[diff];
		if (res == subnetmasks_b1)
			return 0;
		return -1;
	}
}


__global__ void findMatches(unsigned char * dev_subnetmasks_b1, unsigned char* dev_subnetmasks_b2, unsigned char * dev_subnetmasks_b3, unsigned char * dev_subnetmasks_b4, unsigned char * dev_prefix_length, unsigned char * ip_b1, unsigned char * ip_b2, unsigned char * ip_b3, unsigned char * ip_b4, int * dev_matchingIndexes, int shift, int totSize, int ipsize)
{
	//int threadId = threadIdx.x;
	unsigned int curthread = (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x);
	//if (blockIdx.x == 143)
		//printf("%d\n", curthread);

	int threadId = curthread % totSize;
	int ipindx = curthread / totSize;
	
	if (ipindx >= ipsize )
		return;
	
	
	
	//if (ipindx > 665)
	//if (threadId == 0 || threadId == totSize - 1)
		//printf("ipind: %d, thread : %d  ------ %d - %d - %d - %d - %d - %d\n", ipindx, threadId, threadIdx.x , blockIdx.x ,blockDim.x , blockIdx.y , gridDim.x, curthread);
	//printf("%d, %d, %d, %d, %d, %d, %d, %d, %d\n", (dev_subnetmasks_b1[threadId + shift], dev_subnetmasks_b2[threadId + shift], dev_subnetmasks_b3[threadId + shift], dev_subnetmasks_b4[threadId + shift], dev_prefix_length[threadId + shift], ip_b1[ipindx], ip_b2[ipindx], ip_b3[ipindx], ip_b4[ipindx]));
	if (0 == isIpMatchingSubnet(dev_subnetmasks_b1[threadId + shift], dev_subnetmasks_b2[threadId + shift], dev_subnetmasks_b3[threadId + shift], dev_subnetmasks_b4[threadId + shift], dev_prefix_length[threadId + shift], ip_b1[ipindx], ip_b2[ipindx], ip_b3[ipindx], ip_b4[ipindx]))
	{
		//printf(" indx: %d ", (32 * ipindx) + dev_prefix_length[threadId + shift]);
		dev_matchingIndexes[ (32 * ipindx) + dev_prefix_length[threadId + shift]] = threadId + shift;
	}
	//printf("thread ID: %d, ipindx: %d after value: %d\n", threadId, ipindx, dev_matchingIndexes[dev_prefix_length[threadId + shift]]);

}

__global__ void findProviderName(int * dev_matchingIndexes, int * dev_finalres, int searchedIpsSize )
{
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread >= searchedIpsSize)
		return;
	
	for (int i = 31; i >= 0; i--)
	{
		
		
		if (dev_matchingIndexes[32 * thread + i] != -1)
		{
			dev_finalres[thread] = dev_matchingIndexes[32 * thread + i];
			//printf("thread: %d, indx: %d, val: %d\n", thread, 32 * thread + i, dev_matchingIndexes[32 * thread + i]);
			return;
		}
	}
	
	dev_finalres[thread] = -1;
			
}

__global__ void resetIpsIndxes(int * dev_matchingIndexes, int searchedIpsSize)
{
	//int threadId = threadIdx.x;
	int threadId = (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.y);
	if (threadId >= searchedIpsSize)
		return;
	dev_matchingIndexes[threadId] = -1;
}


cudaError_t startBFPLPM(unsigned char * subnetmasks_b1, unsigned char * subnetmasks_b2, unsigned char * subnetmasks_b3, unsigned char* subnetmasks_b4, unsigned char * prefixLength, unsigned char * ip_b1, unsigned char * ip_b2, unsigned char * ip_b3, unsigned char * ip_b4, int size, int searchedIpsSize);

const int size = 5000000;//5822030; //3328690;
const int searchedIpsSize = 65000;
int main()
{
	//data_prepration("./InputData/as-schemes.csv");
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Device name: %s\n", prop.name);
	char * methodName = "LPM-BRUTE FORCE";

	
	
	char ** comp = new char*[size];
	char ** prov = new char*[size];

	unsigned char * subnetmasks_b1 = new unsigned char[size];
	unsigned char * subnetmasks_b2 = new unsigned char[size];
	unsigned char * subnetmasks_b3 = new unsigned char[size];
	unsigned char * subnetmasks_b4 = new unsigned char[size];
	unsigned char * prefixLength = new unsigned char[size];
	
	unsigned char * ip_b1 = new unsigned char[searchedIpsSize];
	unsigned char * ip_b2 = new unsigned char[searchedIpsSize];
	unsigned char * ip_b3 = new unsigned char[searchedIpsSize];
	unsigned char * ip_b4 = new unsigned char[searchedIpsSize];

	clock_t start, end;

	start = clock();
	read_data("./InputData/subnets10000000.txt", comp, prov, subnetmasks_b1, subnetmasks_b2, subnetmasks_b3, subnetmasks_b4, prefixLength, size);
	end = clock();

	clock_t elapsed = (end - start) * 1000.0 / CLOCKS_PER_SEC;
	printf("readimg time: %ld\n", elapsed );
	
	/* Intializes random number generator */
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < searchedIpsSize; i++)
	{
		int pos = rand() % 100000;
		ip_b1[i] = subnetmasks_b1[pos];
		ip_b2[i] = subnetmasks_b2[pos];
		ip_b3[i] = subnetmasks_b3[pos];
		ip_b4[i] = subnetmasks_b4[pos];
	}
	//for (int counter = 0; counter < 40; counter++)
		//printf("%d.%d.%d.%d/%d\n", /*comp[counter], prov[counter]*/ subnetmasks_b1[counter], subnetmasks_b2[counter], subnetmasks_b3[counter], subnetmasks_b4[counter], prefixLength[counter]);


	

	
	cudaError_t cudaStatus = startBFPLPM(subnetmasks_b1, subnetmasks_b2, subnetmasks_b3, subnetmasks_b4, prefixLength, ip_b1, ip_b2, ip_b3, ip_b4, size, searchedIpsSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "startPLPM failed!");
		return 1;
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


cudaError_t startBFPLPM(unsigned char * subnetmasks_b1, unsigned char * subnetmasks_b2, unsigned char * subnetmasks_b3, unsigned char* subnetmasks_b4, unsigned char * prefixLength, unsigned char * ip_b1, unsigned char * ip_b2, unsigned char * ip_b3, unsigned char * ip_b4, int size, int searchedIpsSize)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	
	unsigned char * dev_subnetmasks_b1 = 0;
	unsigned char * dev_subnetmasks_b2 = 0;
	unsigned char * dev_subnetmasks_b3 = 0;
	unsigned char * dev_subnetmasks_b4 = 0;
	unsigned char * dev_prefix_length = 0;

	unsigned char * dev_ip_b1 = 0;
	unsigned char * dev_ip_b2 = 0;
	unsigned char * dev_ip_b3 = 0;
	unsigned char * dev_ip_b4 = 0;

	int * dev_matchedIndexes = 0;
	int * dev_final_results = 0;

	// Allocate GPU buffers for subnetmasks vector.
	cudaStatus = cudaMalloc((void**)&dev_subnetmasks_b1, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_subnetmasks_b2, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_subnetmasks_b3, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_subnetmasks_b4, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_prefix_length, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_ip_b1, searchedIpsSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_ip_b2, searchedIpsSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_ip_b3, searchedIpsSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_ip_b4, searchedIpsSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_final_results, searchedIpsSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_matchedIndexes, 32 *searchedIpsSize* sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_subnetmasks_b1, subnetmasks_b1, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_subnetmasks_b2, subnetmasks_b2, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_subnetmasks_b3, subnetmasks_b3, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_subnetmasks_b4, subnetmasks_b4, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_prefix_length, prefixLength, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_ip_b1, ip_b1, searchedIpsSize* sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_ip_b2, ip_b2, searchedIpsSize* sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_ip_b3, ip_b3, searchedIpsSize* sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_ip_b4, ip_b4, searchedIpsSize* sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	int * minusOneArray = new int [32 * searchedIpsSize];
	for (int i = 0; i < 32 * searchedIpsSize; i++)
		minusOneArray[i] = -1;

	cudaStatus = cudaMemcpy(dev_matchedIndexes, minusOneArray, 32 * searchedIpsSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// prepare test data
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Device name: %s\n", prop.name);
	char * methodName = "LPM-BRUTE FORCE";

	int testsCount = 25;
	const int saving_list_length = 10002;
	int blks[saving_list_length];
	int threads[saving_list_length];
	float timebuilding[saving_list_length];
	float execTime[saving_list_length];
	int searchedIPs[saving_list_length];
	int datasize[saving_list_length];

	int totalTests = 0;
 	int data_size_start = 200000;
	int data_size_step = 100000;
	int threads_start = 96;
	int threads_step = 32;
	int searchedIp_start = 1;
	int searched_ip_step = 150;
	int data_size_counter = 0;
	int max_threads_in_block = 1025;
	clock_t start;
	int threadspreBlk = 0;
	int blocks = 0;
	int shift = 0;
	int threadspreBlkPro = 0;
	int blocksPro = 0;
	int * matchedIndxes;
	int * fres;
	clock_t end;
	clock_t elapsed = 0;
	for (data_size_counter = data_size_start; data_size_counter < size; data_size_counter += data_size_step)
	{
		for (int thread_counter = threads_start; thread_counter < max_threads_in_block; thread_counter += threads_step)
		{
			int searched_ips_size_max = (thread_counter * 65000 * 2) / data_size_counter;
			for (int searchedIPCounter = searchedIp_start; searchedIPCounter < searched_ips_size_max; searchedIPCounter += searched_ip_step)
			{
				for (int testcounter = 0; testcounter < testsCount; testcounter++)
				{
					if (true)
					{
						unsigned int all_needed_threads = searchedIPCounter * data_size_counter / thread_counter;
						int blocks_y = 1;
						if (all_needed_threads  > 65000)
						{
							blocks = 65000;
							blocks_y = all_needed_threads / 65000;
							if (all_needed_threads % 65000 > 0)
								blocks_y++;
						}
						else
						{
							blocks_y = 1;
							blocks = all_needed_threads;
							if (data_size_counter * searchedIPCounter % thread_counter > 0)
								blocks++;
						}

						dim3 numBlocks(blocks, blocks_y);
						shift = 0; // change it later
						printf("TEST NO: %d:%d\n", totalTests, testcounter);
						printf("SEARCHED IPs : %d\n", searchedIPCounter);
						printf("DATA SIZE : %d\n", data_size_counter);
						printf("THREADS: %d\n", thread_counter);
						printf("BLOCKS: %d\n", blocks);
						printf("BLOCKS - Y: %d\n", blocks_y);
						printf("------------------------------------------------------------------\n");
						start = clock();

						findMatches << <numBlocks, thread_counter >> >(dev_subnetmasks_b1, dev_subnetmasks_b2, dev_subnetmasks_b3, dev_subnetmasks_b4, dev_prefix_length, dev_ip_b1, dev_ip_b2, dev_ip_b3, dev_ip_b4, dev_matchedIndexes, shift, data_size_counter, searchedIPCounter);

						// Check for any errors launching the kernel
						/*cudaStatus = cudaGetLastError();
						if (cudaStatus != cudaSuccess) {
							fprintf(stderr, "findMatches launch failed: %s\n", cudaGetErrorString(cudaStatus));
							goto Error;
						}

						// cudaDeviceSynchronize waits for the kernel to finish, and returns
						// any errors encountered during the launch.
						cudaStatus = cudaDeviceSynchronize();
						if (cudaStatus != cudaSuccess) {
							fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findMatches!\n", cudaStatus);
							goto Error;
						}*/
						threadspreBlkPro = searchedIPCounter;
						blocksPro = 1;
						if (threadspreBlkPro > 1024)
						{
							threadspreBlkPro = 1024;
							blocksPro = searchedIPCounter / threadspreBlkPro;
							if (data_size_counter % threadspreBlkPro != 0)
								blocksPro++;
						}

						findProviderName << <blocksPro, threadspreBlkPro >> >(dev_matchedIndexes, dev_final_results, searchedIPCounter);
						// Check for any errors launching the kernel
						cudaStatus = cudaGetLastError();
						if (cudaStatus != cudaSuccess) {
							fprintf(stderr, "findMatches launch failed: %s\n", cudaGetErrorString(cudaStatus));
							goto Error;
						}

						// cudaDeviceSynchronize waits for the kernel to finish, and returns
						// any errors encountered during the launch.
						cudaStatus = cudaDeviceSynchronize();
						if (cudaStatus != cudaSuccess) {
							fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findMatches!\n", cudaStatus);
							goto Error;
						}
						matchedIndxes = new int[32 * searchedIPCounter];
						cudaStatus = cudaMemcpy(matchedIndxes, dev_matchedIndexes, 32 * searchedIPCounter * sizeof(int), cudaMemcpyDeviceToHost);
						if (cudaStatus != cudaSuccess) {
							fprintf(stderr, "cudaMemcpy failed!");
							goto Error;
						}

						fres = new int[searchedIPCounter];
						cudaStatus = cudaMemcpy(fres, dev_final_results, searchedIPCounter * sizeof(int), cudaMemcpyDeviceToHost);
						if (cudaStatus != cudaSuccess) {
							fprintf(stderr, "cudaMemcpy failed!");
							goto Error;
						}
						end = clock();
						elapsed = (end - start) * 1000 / CLOCKS_PER_SEC;
						//printf("searching time:  %ld\n", elapsed);

						//for (int lk = 0; lk < searchedIPCounter; lk++)
						//printf("%d  ------> %d:  IP: %d.%d.%d.%d    ->   Subnet: %d.%d.%d.%d-%d\n", lk, fres[lk], ip_b1[lk], ip_b2[lk], ip_b3[lk], ip_b4[lk], subnetmasks_b1[fres[lk]], subnetmasks_b2[fres[lk]], subnetmasks_b3[fres[lk]], subnetmasks_b4[fres[lk]], prefixLength[fres[lk]]);
						blks[totalTests] = blocks;
						threads[totalTests] = thread_counter;
						timebuilding[totalTests] = 0;
						execTime[totalTests] = elapsed; // ((float)elapsed) / testsCount;
						searchedIPs[totalTests] = searchedIPCounter;
						datasize[totalTests] = data_size_counter;
						totalTests++;
						elapsed = 0;
						if (totalTests == 1000)
						{
							save_test_results(prop.name, methodName, blks, threads, datasize, searchedIPs, timebuilding, execTime, totalTests);
							totalTests = 0;

						}
						// reset MatchedIPS:
						int resetThreads = searchedIPCounter;
						int resetBlocks = 1;
						if (searchedIPCounter > 1024)
						{
							resetBlocks = 1024;
							resetBlocks = searchedIPCounter / 1024;
							if (searchedIPCounter % 1024 > 0)
								resetBlocks++;
						}
						resetIpsIndxes << <resetBlocks, resetThreads >> > (dev_matchedIndexes, searchedIPCounter);

						//printf("Original Array \n");
						//for (int i = 0; i < size; i++)
						//printf("%d  ------> %d.%d.%d.%d - %d\n", i, subnetmasks_b1[i], subnetmasks_b2[i], subnetmasks_b3[i], subnetmasks_b4[i], prefixLength[i]);
						/*

						for (int k = 0; k < searchedIPCounter; k++)
						{
						printf("Matched Values for IP: %d.%d.%d.%d are:\n", ip_b1[k], ip_b2[k], ip_b3[k], ip_b4[k]);
						for (int i = 31; i >= 0; i--)
						{
						if (matchedIndxes[i + k * 32] != -1)
						{
						printf("Prefix: %d    , Index: %d  \n", i, matchedIndxes[i + k * 32]);
						printf("%d  ------> %d.%d.%d.%d - %d\n", i, subnetmasks_b1[matchedIndxes[i + k * 32]], subnetmasks_b2[matchedIndxes[i + k * 32]], subnetmasks_b3[matchedIndxes[i + k * 32]], subnetmasks_b4[matchedIndxes[i + k * 32]], prefixLength[matchedIndxes[i + k * 32]]);
						}
						}
						printf("-----------------------------------------------------------\n");
						}
						*/

					}
					
					
				}
				
			}
			
		}


	}

	if (totalTests > 0)
		save_test_results(prop.name, methodName, blks, threads, datasize, searchedIPs, timebuilding, execTime, totalTests);

			


Error:
	cudaFree(dev_subnetmasks_b1);
	cudaFree(dev_subnetmasks_b2);
	cudaFree(dev_subnetmasks_b3);
	cudaFree(dev_subnetmasks_b4);
	cudaFree(dev_matchedIndexes);
	cudaFree(dev_final_results);
	cudaFree(dev_ip_b1);
	cudaFree(dev_ip_b2);
	cudaFree(dev_ip_b3);
	cudaFree(dev_ip_b4);
	cudaFree(dev_prefix_length);
	return cudaStatus;
}

void save_test_results(char * device, char * methodname, int * blks, int * threads, int * inputlen, int * searchedIp, float * buildtime, float * exectime, int size)
{
	FILE * fout = fopen("./TestResults/res.txt", "a");
	if (fout == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	for (int i = 0; i < size; i++)
	{
		fprintf(fout, "%s,%s,%d,%d,%d,%d,%0.3lf,%0.3lf\n", device, methodname, blks[i], threads[i], inputlen[i], searchedIp[i], buildtime[i], exectime[i]);
	}
	fclose(fout);
}

void data_prepration(char * path)
{
	//int x = std::numeric_limits<int>::max();
	const int buffer_size = 1000000; // 2147483647;
	char s[buffer_size];
	FILE* f = fopen(path, "r");
	if (!f)
		return;
	FILE * fout = fopen("./InputData/as-schemes-final.csv", "w");
	if (fout == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	char * comp;
	char * provider;
	char * subnetmasks;
	char * subnetmask;
	char * prefix_length;
	int counter = 0;
	while (fgets(s, buffer_size, f) != NULL)
	{
		counter++;
		comp = strtok(s, ",");
		provider = strtok(NULL, ",");
		subnetmasks = strtok(NULL, ",");
		printf(" %s\n", comp);
		printf(" %s\n", provider);
		printf(" %s\n", subnetmasks);

		// spliting subnet masks
		char * subnetmask_f = strtok(subnetmasks, " ");
		while (subnetmask_f != NULL && strcmp(subnetmask_f, "\n") != 0)
		{
			int slash_indx = (int)strchr(subnetmask_f, '/');
			subnetmask = new char[slash_indx - (int)subnetmask_f];
			prefix_length = new char[strlen(subnetmask_f) - slash_indx + (int)subnetmask_f];
			strncpy(subnetmask, subnetmask_f, slash_indx - (int)subnetmask_f);
			subnetmask[slash_indx - (int)subnetmask_f] = '\0';
			strcpy(prefix_length, &subnetmask_f[slash_indx - (int)subnetmask_f + 1]);

			printf(" %s\n", subnetmask_f);
			printf(" %s\n", subnetmask);
			printf(" %s\n", prefix_length);

			fprintf(fout, "%s", comp);
			fprintf(fout, "%s", ";");
			fprintf(fout, "%s", provider);
			fprintf(fout, "%s", ";");
			fprintf(fout, "%s", subnetmask);
			fprintf(fout, "%s", ";");
			fprintf(fout, "%s\n", prefix_length);

			subnetmask_f = strtok(NULL, " ");
		}

	}
	fclose(fout);
	fclose(f);

}

void read_data(char * path, char** comp, char** prov, unsigned char* subnet_b1, unsigned char * subnet_b2, unsigned char * subnet_b3, unsigned char * subnet_b4, unsigned char * subnet_pref, int total)
{
	const int buffer_size = 1000000; // 2147483647;
	char s[buffer_size];
	FILE* f = fopen(path, "r");
	if (!f)
		return;
	char * subnetmask;
	int counter = 0;
	while (fgets(s, buffer_size, f) != NULL)
	{
		comp[counter] = strtok(s, ";");
		prov[counter] = strtok(NULL, ";");
		subnetmask = strtok(NULL, ";");
		subnet_pref[counter] = (unsigned char)atoi(strtok(NULL, ";"));

		subnet_b1[counter] = (unsigned char)atoi(strtok(subnetmask, "."));
		subnet_b2[counter] = (unsigned char)atoi(strtok(NULL, "."));
		subnet_b3[counter] = (unsigned char)atoi(strtok(NULL, "."));
		subnet_b4[counter] = (unsigned char)atoi(strtok(NULL, "."));
		//printf("%s - %s - %d - %d - %d - %d - %d\n", comp[counter], prov[counter], subnet_b1[counter], subnet_b2[counter], subnet_b3[counter], subnet_b4[counter], subnet_pref[counter]);
		counter++;
		if (counter == total)
			break;
	}
	fclose(f);

}