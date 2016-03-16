
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <limits>
#include <string.h>
#include <time.h>


__device__ void copy_array(int * dest, int * src, int size)
{
	for (int i = 0; i < size; i++)
		dest[i] = src[i];
}

cudaError_t startPLPM(int * input_array, int size);

void calculateAccumilativeCountForBucket(int * bucketsItems, int * accumBucketsItems, int size)
{
	accumBucketsItems[0] = 0;
	for (int i = 1; i < size; i++)
		accumBucketsItems[i] = accumBucketsItems[i - 1] + bucketsItems[i - 1];
}

__device__ int compareElements(int elm1, int elm2)
{
	/*
	if equal return 0
	if item1 > item 2 return +1
	ir item1 < item 2 return -1
	*/

	if (elm1 > elm2)
		return 1;
	if (elm1 < elm2)
		return -1;
	return 0;


}

__device__ int findBucketforItem(int* dev_splitters, int item, int splittersSize)
{
	int start = 0;
	int end = splittersSize - 1;
	int res = 0;
	if (compareElements(item, dev_splitters[0]) < 0)
		return 0;
	if (compareElements(item, dev_splitters[end])  > 0)
		return end + 1;
	int curr_indx = 0;
	while (abs(start - end) > 1)
	{
		curr_indx = (start + end) / 2;
		res = compareElements(item, dev_splitters[curr_indx]);
		if (res == 0)
			return curr_indx;
		if (res < 0)
		{
			curr_indx -= 1;
			res = compareElements(item, dev_splitters[curr_indx]);
			if (res > 0)
				return curr_indx + 1;
			end = curr_indx + 1;
		}
		else
		{
			curr_indx += 1;
			res = compareElements(item, dev_splitters[curr_indx]);
			if (res <= 0)
				return curr_indx;
			start = curr_indx - 1;
		}
	}

	return 0;
}

__global__ void findSplitters(int * dev_input_array, int * dev_splitters, int splitterSize, int totalSize)
{
	int threadId = threadIdx.x;
	int newLocation = 0;
	int jumpStep = totalSize / splitterSize;
	int item_indx = threadId * jumpStep;
	int res = 0;
	//printf("%d - %d - %d - %d\n ", threadId, item_indx, jumpStep, dev_subnetmasks_b4[item_indx]);
	for (int i = 0; i < splitterSize; i++)
	{
		res = compareElements(dev_input_array[item_indx], dev_input_array[i * jumpStep]);

		if (res > 0)
			newLocation += 1;
		else if ((res == 0) && (i * jumpStep < item_indx))
			newLocation += 1;
	}
	dev_splitters[newLocation] = dev_input_array[item_indx];
	
	//printf("ZZZZZ%d - %d - %d - %d\n ", threadId, item_indx, newLocation, dev_subnetmasks_b4[item_indx]);
}

__global__ void fillBucket(int * dev_input_array, int * dev_splitters, int* bucket_id, int* itemsCountInEachBucketForEachThread, int * serial_id, int rangeSize, int rangesCount, int splittersSize, int totalSize)
{
	int threadId = threadIdx.x;
	int start = threadId * rangeSize;
	int end = threadId * rangeSize + rangeSize;
	if (threadId == splittersSize - 1)
		end = totalSize;
	//printf("%d - %d - %d - %d\n ", threadId, start, end, rangeSize);
	for (int i = start; i < end; i++)
	{
		int bucket = findBucketforItem(dev_splitters,dev_input_array[i], splittersSize);
		bucket_id[i] = bucket;
		serial_id[i] = itemsCountInEachBucketForEachThread[threadId * (splittersSize + 1) + bucket];
		itemsCountInEachBucketForEachThread[threadId * (splittersSize + 1) + bucket] += 1;
	}
}

__global__ void calculateAccumItemsRangeBucket(int* itemsCountInEachBucketForEachThread, int * accumItemsCountInEachBucketForEachThread, int splittersSize, int rangesCount)
{
	int threadId = threadIdx.x;
	accumItemsCountInEachBucketForEachThread[threadId] = 0;
	for (int i = 1; i < rangesCount; i++)
		accumItemsCountInEachBucketForEachThread[threadId + (splittersSize + 1) * i] = accumItemsCountInEachBucketForEachThread[(splittersSize + 1) * (i - 1) + threadId] + itemsCountInEachBucketForEachThread[(splittersSize + 1) * (i - 1) + threadId];
}

__global__ void countBucketItems(int* itemsCountInEachBucketForEachThread, int * bucketsItems, int splittersSize, int rangesCount)
{
	int threadId = threadIdx.x;
	bucketsItems[threadId] = 0;
	for (int i = 0; i < rangesCount; i++)
		bucketsItems[threadId] += itemsCountInEachBucketForEachThread[(splittersSize + 1) * i + threadId];
}

__global__ void mergeBuckets(int * dev_input_array, int* bucket_id, int* itemsCountInEachBucketForEachThread, int * serial_id, int * accum_bucket, int* accItemsCountInEachBucketForEachThread, int* dev_input_array_temp, int rangeSize, int splittersSize, int totalSize)
{
	int threadId = threadIdx.x;
	int start = threadId * rangeSize;
	int end = threadId * rangeSize + rangeSize;
	//printf("%d - %d - %d \n", threadId, start, end);
	if (threadId == splittersSize - 1)
		end = totalSize;
	for (int i = start; i < end; i++)
	{
		//printf("%d - %d --> %d \n", threadId, i, accum_bucket[bucket_id[i]] + serial_id[i]);
		dev_input_array_temp[accum_bucket[bucket_id[i]] + serial_id[i] + accItemsCountInEachBucketForEachThread[threadId * (splittersSize + 1) + bucket_id[i]]] = dev_input_array[i];
	}
}

__device__ void mergeSort(int * dev_bucket, int * dev_bucket_temp, int * dev_accum_bucket, int rangeL, int rangeR, int start)
{
	int leftTrav = 0;
	int rightTrav = 0;
	while ((leftTrav < rangeL) && (rightTrav < rangeR))
	{
		if (dev_bucket[start + leftTrav] < dev_bucket[start + rangeL + rightTrav])
		{
			dev_bucket_temp[start + leftTrav + rightTrav] = dev_bucket[start + leftTrav];
			leftTrav++;
		}
		else
		{
			dev_bucket_temp[start + leftTrav + rightTrav] = dev_bucket[start + rangeL + rightTrav];
			rightTrav++;
		}

	}
	if (leftTrav < rangeL)
	for (leftTrav; leftTrav < rangeL; leftTrav++)
		dev_bucket_temp[start + leftTrav + rightTrav] = dev_bucket[start + leftTrav];
	else
	for (rightTrav; rightTrav < rangeR; rightTrav++)
		dev_bucket_temp[start + leftTrav + rightTrav] = dev_bucket[start + rangeL + rightTrav];

}

__global__ void startSeqMergeSort(int * dev_input_array_temp, int * dev_accum_bucket)
{
	int threadId = threadIdx.x;
	int start = dev_accum_bucket[threadId];
	int end = dev_accum_bucket[threadId + 1];
	int bucketSize = end - start;
	//printf("%d - %d - %d \n", threadId, start, end);
	if (bucketSize == 1)
		return;

	int * bucket_temp = new int[bucketSize];
	//memcpy(bucket_temp, dev_subnetmasks_b4_temp + start, bucketSize * 4);
	copy_array(bucket_temp, dev_input_array_temp + start, bucketSize);
	int * temp_array = new int[bucketSize];

	for (int i = 1; i < bucketSize; i *= 2)
	{
		int rangeL = i;
		int rangeR = i;

		for (int first = 0; first < bucketSize; first += rangeL * 2)
		{
			if (first + rangeL > bucketSize)
				rangeL = bucketSize - first;

			if (bucketSize - (2 * rangeL + first) < 0)
				rangeR = bucketSize - (rangeL + first);
			mergeSort(bucket_temp, temp_array, dev_accum_bucket, rangeL, rangeR, first);
		}
		//free(bucket_temp);
		bucket_temp = temp_array;
		temp_array = new int[bucketSize];

	}
	//memcpy(dev_subnetmasks_b4_temp + start, bucket_temp, (bucketSize)* 4);
	copy_array(dev_input_array_temp + start, bucket_temp, bucketSize);

}

const int splitters_size = 11;
const int size = 200;
const int sizee = 200;
int main()
{
	//data_prepration("./InputData/as-schemes.csv");
	//return;

	int * input_array = new int[size];
	srand(time(NULL));
	printf("Original Array \n ");
	for (int i = 0; i < size; i++)
	{
		input_array[i] = rand() % 2000;
		printf("%d-", input_array[i]);
	}

	printf("\n \n ");
	cudaError_t cudaStatus = startPLPM(input_array, size);
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


cudaError_t startPLPM(int * input_array, int size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int threadsCount = splitters_size;

	
	int splitters[splitters_size];
	
	int * dev_input_array = 0;
	
	int * dev_input_array_temp = 0;
	
	int * dev_splitters= 0;
	

	int * dev_bucket_id = 0;
	int * dev_serial_id = 0;
	int * dev_itemsCountInEachBucketForEachThread = 0;
	int * dev_accumItemsCountInEachBucketForEachThread = 0;
	int * dev_accum_bucket = 0,
	int * dev_bucketsItems = 0;

	// Allocate GPU buffers for subnetmasks vector.
	cudaStatus = cudaMalloc((void**)&dev_input_array, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_splitters, splitters_size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_bucket_id, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_serial_id, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_itemsCountInEachBucketForEachThread, threadsCount * (splitters_size + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_accumItemsCountInEachBucketForEachThread, threadsCount * (splitters_size + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_accum_bucket, (splitters_size + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bucketsItems, (splitters_size + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_input_array_temp, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input_array, input_array, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int zeroArray[splitters_size * (splitters_size + 1)] = { 0 };
	cudaStatus = cudaMemcpy(dev_itemsCountInEachBucketForEachThread, zeroArray, threadsCount * (splitters_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}




	// Launch a kernel on the GPU with one thread for each element.
	findSplitters << <1, threadsCount >> >(dev_input_array, dev_splitters, splitters_size, size);

	int spl[splitters_size];
	cudaStatus = cudaMemcpy(spl, dev_splitters, splitters_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int rangeSize = size / threadsCount;
	int rangesCount = threadsCount;

	fillBucket << <1, threadsCount >> >(dev_input_array, dev_splitters, dev_bucket_id, dev_itemsCountInEachBucketForEachThread, dev_serial_id, rangeSize, rangesCount, splitters_size, size);

	int buk[sizee];
	cudaStatus = cudaMemcpy(buk, dev_bucket_id, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	int itCB[splitters_size * (splitters_size + 1)];
	cudaStatus = cudaMemcpy(itCB, dev_itemsCountInEachBucketForEachThread, splitters_size * (splitters_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	int serial[sizee];
	cudaStatus = cudaMemcpy(serial, dev_serial_id, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	countBucketItems << <1, splitters_size + 1 >> >(dev_itemsCountInEachBucketForEachThread, dev_bucketsItems, splitters_size, rangesCount);

	int item_count[splitters_size + 1];
	cudaStatus = cudaMemcpy(item_count, dev_bucketsItems, (splitters_size + 1)* sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	int accumBucketsItems[splitters_size + 2];
	int bucketItems[splitters_size + 1];
	cudaStatus = cudaMemcpy(bucketItems, dev_bucketsItems, (splitters_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	calculateAccumilativeCountForBucket(bucketItems, accumBucketsItems, splitters_size + 2);

	cudaStatus = cudaMalloc((void**)&dev_accum_bucket, (splitters_size + 2)* sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_accum_bucket, accumBucketsItems, (splitters_size + 2) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	calculateAccumItemsRangeBucket << <1, splitters_size + 1 >> >(dev_itemsCountInEachBucketForEachThread, dev_accumItemsCountInEachBucketForEachThread, splitters_size, rangesCount);

	int accitemsCountInEachBucketForEachThread[splitters_size * (splitters_size + 1)];
	cudaStatus = cudaMemcpy(accitemsCountInEachBucketForEachThread, dev_accumItemsCountInEachBucketForEachThread, threadsCount * (splitters_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	mergeBuckets << <1, threadsCount >> >(dev_input_array, dev_bucket_id, dev_itemsCountInEachBucketForEachThread, dev_serial_id, dev_accum_bucket, dev_accumItemsCountInEachBucketForEachThread, dev_input_array_temp, rangeSize, splitters_size, size);

	startSeqMergeSort << <1, splitters_size + 1 >> >(dev_input_array_temp, dev_accum_bucket);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "findSplitters launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching !\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	int * out = new int[size];
	cudaStatus = cudaMemcpy(out, dev_input_array_temp, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	for (int i = 0; i < size; i++)
	for (int j = 0; j < i; j++)
	if (out[j] > out[i])
		printf(" wrong sorting %d - with- %d -- Values %d - %d \n", j, i, out[j], out[i]);

	printf("Sorted Array \n ");
	for (int i = 0; i < size; i++)
		printf("%d--", out[i]);



Error:
	cudaFree(dev_input_array);
	cudaFree(dev_splitters);
	cudaFree(dev_input_array_temp);
	cudaFree(dev_bucket_id);
	cudaFree(dev_serial_id);
	cudaFree(dev_itemsCountInEachBucketForEachThread);
	cudaFree(dev_accumItemsCountInEachBucketForEachThread);
	cudaFree(dev_accum_bucket);
	cudaFree(dev_bucketsItems);
	return cudaStatus;
}

