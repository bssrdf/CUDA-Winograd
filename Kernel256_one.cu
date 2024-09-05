#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>
#include <immintrin.h>

//#include "cudnn.h"
#include "util.h"
#include "Kernel256_one.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}



__global__ void kernel_1024_one_256(float *A, float *B, float *bnBias, float *bnScale, float *C) {
	int tile = blockIdx.x, in_channel = threadIdx.x, line = threadIdx.y;
	int ind = line*256 + in_channel;

	extern __shared__ float shared_[];
	float *weights = shared_ + 1024*4, *output = weights + 256*16, *input = shared_;
	float *bias = output + 4*256, *scale = bias + 256;

	for (int i = 0; i < 4; i++)
		input[ind + i*1024] = A[tile*4096 + i*1024 + ind];
	bias[in_channel] = bnBias[in_channel];
	scale[in_channel] = bnScale[in_channel];
	output[ind] = 0.0f;
	__syncthreads();

	for (int k = 0; k < 1024; k += 16) {
		float *B_start = B + k*256;
		for (int i = 0; i < 4; i++)
			weights[ind + i*1024] = B_start[i*1024 + ind];
		__syncthreads();

		float *A_start = input + k;
		for (int p = 0; p < 16; p++) {
			output[ind] += A_start[line*1024 + p] * weights[in_channel + p*256];
		}
		__syncthreads();
	}

	float *C_start = C + tile*1024, res = scale[in_channel] * output[ind] + bias[in_channel];
	C_start[ind] = res > 0 ? res : 0;
}


int kernel_256_1_in() {
	float *input = get_parameter(inputName256one, 14*14*1024);
	float *weight = get_parameter(weightName256one, 256*1024);

	float *bnBias = get_parameter(bnBiasName256one, 256);
	float *bnScale = get_parameter(bnScaleName256one, 256);
	float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 256);
	float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 256);
	float *eMeanName = get_parameter(eMeanName256one, 256);
	float *eVarName = get_parameter(eVarName256one, 256);

	float *input_, *output_, *weight_, *bnBias_, *bnScale_, *eMeanName_, *eVarName_;

	int nInput = 14*14*1024, nOutput = 14*14*256, nWeights = 256*1024;
	float tmp[nOutput], tmp_cudnn[nOutput];

	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;
	cudaError_t s;

	/////////////////////////////////

	// My Kernel

	/////////////////////////////////

	/*  1. Data preparation  */
	cudaMalloc((void **) &input_, nInput<<3);
	cudaMalloc((void **) &output_, nOutput<<2);
	cudaMalloc((void **) &weight_, nWeights<<2);
	cudaMalloc((void **) &bnBias_, 256<<2);
	cudaMalloc((void **) &bnScale_, 256<<2);

	cudaMemcpy(input_, input, nInput<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnBias_, bnBias_myKernel, 256<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnScale_, bnScale_myKernel, 256<<2, cudaMemcpyHostToDevice);


	/*  2. Computing  */
	nT1 = getTimeMicroseconds64();

	kernel_1024_one_256 <<<dim3(49), dim3(256, 4), (4*1024 + 16*256 + 4*256 + 2*256)<<2 >>> (input_, weight_, bnBias_, bnScale_, output_);

	//cudaCheckError();
	cudaDeviceSynchronize();

	nT2 = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2-nT1);


	/*  3. Copy back and free  */
	s = cudaMemcpy(tmp, output_, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	cudaCheckError();

	free(bnBias_myKernel);
	free(bnScale_myKernel);

	
//	output_checker(tmp, tmp_cudnn, 14, 256, 0);

	return ((nT2-nT1) << 16);
}



__global__ void kernel_256_one_1024(float *A, float *B, float *bnBias, float *bnScale, float *C) {
	int tile = blockIdx.x, part = blockIdx.y, in_channel = threadIdx.x, line = threadIdx.y;
	int ind = line*256 + in_channel;

	extern __shared__ float shared_[];
	float *weights = shared_ + 256*4, *output = weights + 256*32, *input = shared_;
	float *bias = output + 4*256, *scale = bias + 256;

	input[ind] = A[tile * 1024 + ind];
	bias[in_channel] = bnBias[part*256 + in_channel];
	scale[in_channel] = bnScale[part*256+ in_channel];
	output[ind] = 0.0f;
	__syncthreads();

	for (int k = 0; k < 256; k += 32) {
		for (int i = 0; i < 8; i++)
			weights[ind + 1024*i] = B[(k + i*4 + line)*1024 + part*256 + in_channel];
		__syncthreads();

		float *A_start = input + k;
		for (int p = 0; p < 32; p++) {
			output[ind] += A_start[line*256 + p] * weights[in_channel + p*256];
		}
		__syncthreads();
	}

	float *C_start = C + tile*4096 + part*256;
	C_start[line * 1024 + in_channel] = scale[in_channel] * output[ind] + bias[in_channel];
}


int kernel_256_1_out() {
	float *input = get_parameter(inputName256one, 14*14*256);
	float *weight = get_parameter(weightName256one, 256*1024);

	float *bnBias = get_parameter(bnBiasName256one, 1024);
	float *bnScale = get_parameter(bnScaleName256one, 1024);
	float *bnBias_myKernel = get_parameter(bnBias_myKernel_Name256one, 1024);
	float *bnScale_myKernel = get_parameter(bnScale_myKernel_Name256one, 1024);
	float *eMeanName = get_parameter(eMeanName256one, 1024);
	float *eVarName = get_parameter(eVarName256one, 1024);

	float *input_, *output_, *weight_, *bnBias_, *bnScale_, *eMeanName_, *eVarName_;

	int nInput = 14*14*256, nOutput = 14*14*1024, nWeights = 256*1024;
	float tmp[nOutput], tmp_cudnn[nOutput];

	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;
	cudaError_t s;

	/////////////////////////////////

	// My Kernel

	/////////////////////////////////

	/*  1. Data preparation  */
	cudaMalloc((void **) &input_, nInput<<3);
	cudaMalloc((void **) &output_, nOutput<<2);
	cudaMalloc((void **) &weight_, nWeights<<2);
	cudaMalloc((void **) &bnBias_, 1024<<2);
	cudaMalloc((void **) &bnScale_, 1024<<2);

	cudaMemcpy(input_, input, nInput<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_, weight, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnBias_, bnBias_myKernel, 1024<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(bnScale_, bnScale_myKernel, 1024<<2, cudaMemcpyHostToDevice);


	/*  2. Computing  */
	nT1 = getTimeMicroseconds64();

	kernel_256_one_1024 <<<dim3(49, 4), dim3(256, 4), (4*256 + 32*256 + 4*256 + 2*256)<<2 >>> (input_, weight_, bnBias_, bnScale_, output_);

	cudaCheckError();
	cudaDeviceSynchronize();

	nT2 = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2-nT1);


	/*  3. Copy back and free  */
	s = cudaMemcpy(tmp, output_, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	cudaCheckError();

	free(bnBias_myKernel);
	free(bnScale_myKernel);


//	output_checker(tmp, tmp_cudnn, 14, 1024, 0);

	return ((nT2-nT1) << 16);
}
