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
#include "Kernel320_winograd.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

#define MY_KERNEL 0

#define d(input, i, j, Inz) ( input[Inz + i*960 + (j*160)] )
__global__ void kernel_320_winograd_BtdB(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Part = blockIdx.z, Iny1 = threadIdx.y, Inz = threadIdx.x;
	int Iny = Iny0+Iny1, stride_r = 5120, stride_c = 320; // 5120 = 16*320
	int c_glb_start = Inx*stride_r + Iny*stride_c + Inz + Part*160, c_input = Iny1*160 + Inz;

	extern __shared__ float input[];

	int stride_960[6] = {0, 960, 1920, 2880, 3840, 4800}; // 960 = 6*160
	for (int i = 0; i < 6; i++) {
		// if(blockIdx.x == 3 && blockIdx.y == 0 && Part == 0 && Inz == 0 && Iny1 == 0)
		//    printf("%d: %d, %d \n", i, c_input + stride_960[i], c_glb_start + i*stride_r);
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
		pOutputs[(Iny1 + i*6)*5120 + ((blockIdx.x<<2)+blockIdx.y)*320 + Inz + Part*160] = BTdB[i];
	}
}

__global__ void kernel_320_winograd_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
	int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
	int c_input = Inx*6 + Iny;

	__shared__ float bias, scale;
	extern __shared__ float input[];

	input[c_input] = pInputs[c_input*16*320 + (Tilex*4+Tiley)*320 + kz];
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

	// if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;
	if (Inx > 3) return;
	
	int x;
	float o;
	switch(Iny) {
		case 0:
			x = Inx*6;
			// o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]) + bias;
			o = (input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*320 + kz] = o;
			break;
		case 1:
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
			o = (input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*320 + kz] = o;
			break;
		case 2:
			// if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
			o = (input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*320 + kz] = o;
			break;
		case 3:
			// if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
			o = (input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*320 + kz] = o;
			break;
	}
}

__global__ void kernel_320_OuterProduct_320(float *A, float *B, float *C) {
	int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
	int c_input = tY*320 + tX, c_kernel = c_input, 
	    // T_offset = (Tile<<11) + (Part<<10) + c_input, 
		T_offset = Tile*5120 + Part*640 + c_input, // 5120 = 16*320, 640 = 320*2
	    B_offset = Tile*102400 + c_kernel; // 102400 = 320*320

	
	extern __shared__ float input[];
	float *kernel = input + 640, *out = kernel + 6400; //3200 = 320*2*10
	int B_stride[20] = {0, 320, 640, 960, 1280, 1600, 1920, 2240, 2560, 2880, 3200, 3520, 3840, 4160, 4480, 4800, 5120, 5440, 5760, 6080};
	out[c_input] = 0.0f;

	input[c_input] = A[T_offset]; // 36*8 blocks, each block loads 320*2 values with 320*2 threads (1 value/t) 
    // we need to compute 320 products and sum them up. 
	// the loop below iterates 16 times and each step computes 20 products and accumulates 
	// the partial sum in shared memory
	// each thread block will process 320*320 kernel values
	// 36 blocks in each row (16 rows in total) will use the same 320*320 kernel values
	for (int k = 0; k < 16; k++) {
		int B_start = B_offset + k*6400; // 320*20 = 6400 
		// each thread loads 10 kernel values
		// the whole thread block can load 320*2*10 = 6400 values
		// incrementing k by 1 will shift by 6400 = 320*20 values 
		kernel[c_kernel] = B[B_start]; 
		kernel[c_kernel+640] = B[B_start+640];
		kernel[c_kernel+1280] = B[B_start+1280]; 
		kernel[c_kernel+1920] = B[B_start+1920];
		kernel[c_kernel+2560] = B[B_start+2560];		
		kernel[c_kernel+3200] = B[B_start+3200];
		kernel[c_kernel+3840] = B[B_start+3840];
		kernel[c_kernel+4480] = B[B_start+4480];
		kernel[c_kernel+5120] = B[B_start+5120];
		kernel[c_kernel+5760] = B[B_start+5760];
		__syncthreads();

        // after sync, all 20 kernel values are ready to be used? How?
		// 6400 / 320 = 20, each thread gets 20 values to use
		// the threads with the same tY share the same 20 values, but how?   
		float sum = 0;
		// int y_tmp = (tY<<7)+(k<<5);
		int y_tmp = tY*320 + k*20;
		for (int j = 0; j < 20; j++) {
			sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
		}
		out[tY*320 + tX] += sum;
		__syncthreads();
	}

	C[T_offset] = out[c_input];
	
}

int kernel_320() {
	float *input_ = get_parameter(inputName320, 16*16*320);
	float *bias = get_parameter(biasName320, 320);
	float *W = get_parameter(weight_NCHW_Name320, 3*3*320*320);
	float *input, *output, *l_weights, *l_bias;
	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;

	uint64_t st1 = 0, st2 = 0, st3 = 0;
	uint64_t et1 = 0, et2 = 0, et3 = 0;

	cudaError_t s;



	/////////////////////////////////

	// My Kernel

	/////////////////////////////////
	float *kernel = get_parameter(weight_winograd_Name320, 36*320*320), *t_input, *ip;
	int nInput = 16*16*320, nOutput = 16*16*320, nWeights = 36*320*320, nBias = 320, 
	    nTransInput = 16*6*6*320, nInnerProd = 16*6*6*320;
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

	bnBias = get_parameter(bnBias_winograd_Name320, 320);
	bnScale = get_parameter(bnScale_winograd_Name320, 320);
	cudaMalloc((void **) &l_bnBias, nBias<<2);
	cudaMalloc((void **) &l_bnScale, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);

	float *tmp = (float*)malloc(nOutput*4);

	nT1 = getTimeMicroseconds64();

	kernel_320_winograd_BtdB <<<dim3(4, 4, 2), dim3(160, 6), (6*6*160)<<2 >>> (input, t_input);
	kernel_320_OuterProduct_320<<<dim3(36, 8), dim3(320, 2), (2*320 + 10*2*320 + 2*320)<<2 >>> (t_input, l_weights, ip);
	kernel_320_winograd_AtIA <<<dim3(4, 4, 320), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
	// cudaCheckError();
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
	cudaFree(l_bnBias);
	cudaFree(l_bnScale);
	cudaFree(ip);
	cudaFree(input);


	free(kernel);
	free(bnScale);
	free(bnBias);
	free(bias);
	

	float *conv_cpu =  (float*)malloc(14*14*320*4);

    nT1_cudnn = getTimeMicroseconds64();
	compute_cpu(input_, W, conv_cpu, 16, 320, 320, 1);
    nT2_cudnn = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);  
	
	free(input_);
	free(W);


//	output_checker(tmp, tmp_cudnn, 14, 320, 1);
	output_checker(tmp, conv_cpu, 14, 320, 1);
	free(conv_cpu);
	free(tmp);

	return ((nT2-nT1) << 16);
}