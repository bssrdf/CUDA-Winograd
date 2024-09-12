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
#include "Kernel160_winograd.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

#define MY_KERNEL 0

#define d(input, i, j, Inz) ( input[Inz + i*960 + (j*160)] )
__global__ void kernel_160_winograd_BtdB(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Iny1 = threadIdx.y, Inz = threadIdx.x;
	int Iny = Iny0+Iny1, stride_r = 2560, stride_c = 160; // 2560 = 16*160
	int c_glb_start = Inx*stride_r + Iny*stride_c + Inz, c_input = Iny1*160 + Inz;

	extern __shared__ float input[];

	// int stride_768[6] = {0, 768, 1536, 2304, 3072, 3840}; // 960 = 6*160
	int stride_960[6] = {0, 960, 1920, 2880, 3840, 4800}; // 960 = 6*160
	for (int i = 0; i < 6; i++) {
		input[c_input + stride_960[i]] = pInputs[c_glb_start + i*stride_r];
	}
	__syncthreads();

	float BTd[6];
	switch(Iny1) {
		case 0:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 0, j, Inz)*4 - d(input, 2, j, Inz)*5 + d(input, 4, j, Inz);
			}
			break;
		case 1:
			for (int j = 0; j < 6; j++) {
				BTd[j] = -d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 + d(input, 3, j, Inz) + d(input, 4, j, Inz);
			}
			break;
		case 2:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 - d(input, 3, j, Inz) + d(input, 4, j, Inz);
			}
			break;
		case 3:
			for (int j = 0; j < 6; j++) {
				BTd[j] = -d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) + d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
			}
			break;
		case 4:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) - d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
			}
			break;
		case 5:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*4 - d(input, 3, j, Inz)*5 + d(input, 5, j, Inz);
			}
			break;
	}
	__syncthreads();

	int tmp_offset = Iny1*960+Inz;
	for (int i = 0; i < 6; i++) {
		input[tmp_offset + i*160] = BTd[i];
	}
	__syncthreads();

	float BTdB[6];
	switch(Iny1) {
		case 0:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 0, Inz) - 5*d(input, i, 2, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 1:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = -4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) + d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 2:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) - d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 3:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = -2*d(input, i, 1, Inz) - d(input, i, 2, Inz) + 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 4:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 2*d(input, i, 1, Inz) - d(input, i, 2, Inz) - 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 5:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 1, Inz) - 5*d(input, i, 3, Inz) + d(input, i, 5, Inz);
			}
			break;
	}
	__syncthreads();

	for (int i = 0; i < 6; i++) {
		pOutputs[(Iny1 + i*6)*2560 + ((blockIdx.x<<2)+blockIdx.y)*160 + Inz] = BTdB[i];
	}
}

__global__ void kernel_160_winograd_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
	int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
	int c_input = Inx*6 + Iny;

	__shared__ float bias, scale;
	extern __shared__ float input[];

	input[c_input] = pInputs[c_input*16*160 + (Tilex*4+Tiley)*160 + kz];
	bias = pBiases[kz];
	scale = pScales[kz];
	__syncthreads();

	float tmp = 0;
	switch(Inx) {
		case 0:
			tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
			break;
		case 1:
			tmp = input[6+Iny] - input[12+Iny] + 2*input[18+Iny] - 2*input[24+Iny];
			break;
		case 2:
			tmp = input[6+Iny] + input[12+Iny] + 4*input[18+Iny] + 4*input[24+Iny];
			break;
		case 3:
			tmp = input[6+Iny] - input[12+Iny] + 8*input[18+Iny] - 8*input[24+Iny] + input[30+Iny];
			break;
	}
	__syncthreads();

	input[c_input] = tmp;
	__syncthreads();

	if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;
	
	int x;
	float o;
	switch(Iny) {
		case 0:
			x = Inx*6;
			// o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]) + bias;
			o = (input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*160 + kz] = o > 0 ? o : 0;
			break;
		case 1:
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
			o = (input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*160 + kz] = o > 0 ? o : 0;
			break;
		case 2:
			if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
			o = (input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*160 + kz] = o > 0 ? o : 0;
			break;
		case 3:
			if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
			o = (input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*160 + kz] = o > 0 ? o : 0;
			break;
	}
}

__global__ void kernel_160_OuterProduct_160(float *A, float *B, float *C) {
	int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
	int c_input = tY*160 + tX, c_kernel = c_input, 
	    // T_offset = (Tile<<11) + (Part<<10) + c_input, 
		T_offset = Tile*2560 + Part*640 + c_input, // 2560 = 16*160, 640 = 160*4
	    B_offset = Tile*25600 + c_kernel; //25600 = 160*160

	
	extern __shared__ float input[];
	float *kernel = input + 640, *out = kernel + 3200; //3200 = 160*4*5
	// int B_stride[32]    = {0, 160, 320, 480, 640, 800, 960, 1120, 1280, 1440, 1600, 1760, 1920, 2080, 2240, 2400, 2560, 2720, 2880, 3040, 3200, 3360, 3520, 3680, 3840, 4000, 4160, 4320, 4480, 4640, 4800, 4960};
	int B_stride[20]    = {0, 160, 320, 480, 640, 800, 960, 1120, 1280, 1440, 1600, 1760, 1920, 2080, 2240, 2400, 2560, 2720, 2880, 3040};
	out[c_input] = 0.0f;

	input[c_input] = A[T_offset]; // 36*4 blocks, each block loads 160*4 values with 160*4 threads (1 value/t) 
    // we need to compute 160 products and sum them up. 
	// the loop below iterates 8 times and each step computes 20 products and accumulates 
	// the partial sum in shared memory
	// each thread block will process 160*160 kernel values
	// 36 blocks in each row (4 rows in total) will use the same 160*160 kernel values
	for (int k = 0; k < 8; k++) {
		int B_start = B_offset + k*3200; // 160*20
		// each thread loads 5 kernel values
		// the whole thread block can load 160*4*5 = 160*20 = 3200 values
		// incrementing k by 1 will shift by  3200 = 160*20 values 
		kernel[c_kernel] = B[B_start]; 
		kernel[c_kernel+640] = B[B_start+640];
		kernel[c_kernel+1280] = B[B_start+1280]; 
		kernel[c_kernel+1920] = B[B_start+1920];
		kernel[c_kernel+2560] = B[B_start+2560];
		__syncthreads();

        // after sync, all 20 kernel values are ready to be used? How?
		// 3200 / 160 = 20, each thread gets 20 values to use
		// the threads with the same tY share the same 20 values, but how?   
		float sum = 0;
		// int y_tmp = (tY<<7)+(k<<5);
		int y_tmp = tY*160 + k*20;
		for (int j = 0; j < 20; j++) {
			sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
		}
		out[tY*160 + tX] += sum;
		__syncthreads();
	}

	C[T_offset] = out[c_input];
	
}

int kernel_160() {
	float *input_ = get_parameter(inputName160, 16*16*160);
	float *bias = get_parameter(biasName160, 160);
	float *W = get_parameter(weight_NCHW_Name160, 3*3*160*160);
	float *input, *output, *l_weights, *l_bias;
	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;
	cudaError_t s;



	/////////////////////////////////

	// My Kernel

	/////////////////////////////////
	float *kernel = get_parameter(weight_winograd_Name160, 36*160*160), *t_input, *ip;
	int nInput = 16*16*160, nOutput = 16*16*160, nWeights = 36*160*160, nBias = 160, 
	    nTransInput = 16*6*6*160, nInnerProd = 16*6*6*160;
	float *l_bnBias, *l_bnScale, *bnBias, *bnScale;

	cudaMalloc((void **) &input, nInput<<2);
	cudaMalloc((void **) &output, nOutput<<2);
	cudaMalloc((void **) &l_weights, nWeights<<2);
	cudaMalloc((void **) &l_bias, nBias<<2);
	cudaMalloc((void **) &t_input, nTransInput<<2);
	cudaMalloc((void **) &ip, nInnerProd<<2);

	cudaMemset((void *) input, 0, nInput<<2);
	cudaMemset((void *) output, 0, nOutput<<2);
	cudaMemset((void *) t_input, 0, nTransInput<<2);
	cudaMemset((void *) l_weights, 0, nWeights<<2);
	cudaMemset((void *) ip, 0, nInnerProd<<2);

	cudaMemcpy(input, input_, nInput<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bias, bias, nBias<<2, cudaMemcpyHostToDevice);

	bnBias = get_parameter(bnBias_winograd_Name160, 160);
	bnScale = get_parameter(bnScale_winograd_Name160, 160);
	cudaMalloc((void **) &l_bnBias, nBias<<2);
	cudaMalloc((void **) &l_bnScale, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);

	float *tmp = (float*)malloc(nOutput*4);

	nT1 = getTimeMicroseconds64();

	kernel_160_winograd_BtdB <<<dim3(4, 4), dim3(160, 6), (6*6*160)<<2 >>> (input, t_input);
	kernel_160_OuterProduct_160<<<dim3(36, 4), dim3(160, 4), (4*160 + 5*4*160 + 4*160)<<2 >>> (t_input, l_weights, ip);
	kernel_160_winograd_AtIA <<<dim3(4, 4, 160), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
	//cudaCheckError();
	cudaDeviceSynchronize();
	
	nT2 = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2-nT1); 

	s = cudaMemcpy(tmp, output, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	//cudaCheckError();

	cudaFree(t_input);
	cudaFree(output);
	cudaFree(l_weights);
	cudaFree(l_bias);
	cudaFree(ip);

	free(kernel);
	free(bnScale);
	free(bnBias);
	

	float *conv_cpu =  (float*)malloc(14*14*160*4);

    nT1_cudnn = getTimeMicroseconds64();
	compute_cpu(input_, W, conv_cpu, 16, 160, 1);
    nT2_cudnn = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);  
	
	free(input_);
	free(W);


//	output_checker(tmp, tmp_cudnn, 14, 160, 1);
	output_checker(tmp, conv_cpu, 14, 160, 1);
	free(conv_cpu);
	free(tmp);

	return ((nT2-nT1) << 16);
}