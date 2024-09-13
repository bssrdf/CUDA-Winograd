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
#include "Kernel2560_winograd.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

#define MY_KERNEL 0

#define d(input, i, j, Inz) ( input[Inz + i*960 + (j*160)] )
__global__ void kernel_2560_winograd_BtdB(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Part = blockIdx.z, Iny1 = threadIdx.y, Inz = threadIdx.x;
	int Iny = Iny0+Iny1, stride_r = 40960, stride_c = 2560; //  40960 = 16*2560
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
		pOutputs[(Iny1 + i*6)*40960 + ((blockIdx.x<<2)+blockIdx.y)*2560 + Inz + Part*160] = BTdB[i];
	}
}

__global__ void kernel_2560_winograd_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
	int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
	int c_input = Inx*6 + Iny;

	__shared__ float bias, scale;
	extern __shared__ float input[];

	input[c_input] = pInputs[c_input*16*2560 + (Tilex*4+Tiley)*2560 + kz];
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
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*2560 + kz] = o > 0 ? o : 0;
			break;
		case 1:
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
			o = (input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*2560 + kz] = o > 0 ? o : 0;
			break;
		case 2:
			// if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
			o = (input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*2560 + kz] = o > 0 ? o : 0;
			break;
		case 3:
			// if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
			o = (input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]);
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*2560 + kz] = o > 0 ? o : 0;
			break;
	}
}

__global__ void kernel_2560_OuterProduct_2560(float *A, float *B, float *C) {
	int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
	u_int64_t c_input = tY*2560 + tX, 
	    c_kernel = tY*512 + tX,  
		T_offset = Tile*40960 + Part*2560*2 + c_input, // 40960 = 16*2560, 1024 = 512*2
	    B_offset = Tile*6553600 + c_input; // 6553600 = 2560*2560
    
	extern __shared__ float input[];
	int B_stride[32] = {0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872};

    // we need to compute 2560 products and sum them up. 
	// the loop below iterates 5 times and each step computes 512 products and accumulates 

	for (int l = 0; l < 5; l++){		
		u_int64_t B_start_l = B_offset + l*512;
		u_int64_t C_offset = T_offset + l*512;
	for (int i = 0; i < 5; i++){		
		float *kernel = input + 1024, *out = kernel + 16384; // 10240 = 512*2*16	
		out[c_kernel] = 0.0f;
        u_int64_t offset = T_offset + (i<<9); // *512
		// if(tX == 0 && tY == 0 && Tile == 0 && Part == 0){
		// 	printf("%d, %d, %ld \n", l, i, offset);
		// }
		input[c_kernel] = A[offset]; // 36*8 blocks, each block loads 512*2 values with 512*2 threads (1 value/t) 
		
		// the loop below iterates  times and each step computes 32 products and accumulates 
		// the partial sum in shared memory
		// each thread block will process 2560*2560 kernel values
		// 36 blocks in each row (8 rows in total) will use the same 2560*2560 kernel values
		u_int64_t B_start_i = B_start_l + i*512*2560;
		for (int k = 0; k < 16; k++) {
			u_int64_t B_start = B_start_i + k*32*2560 ; // 2560*20 = 6400 
			// each thread loads 16 kernel values
			// the whole thread block can load 512*2*16 = 16384 values
			// incrementing k by 1 will shift by 16384 = 512*2*16 values 

			// if we have 2 threads in y direction, the two with the same tY 
			// alternating get two lines of portion of 2560 weights
			// because the layout of weights has out_dim as the first (fast) dimension,
			// to get weights along in_dim, we have to iterating over rows		
			kernel[c_kernel] = B[B_start]; 			
			kernel[c_kernel+1024] = B[B_start+5120]; // 5120 = 2560*2
			kernel[c_kernel+2048] = B[B_start+10240];
			kernel[c_kernel+3072] = B[B_start+15360];
			kernel[c_kernel+4096] = B[B_start+20480];
			kernel[c_kernel+5120] = B[B_start+25600];
			kernel[c_kernel+6144] = B[B_start+30720];
			kernel[c_kernel+7168] = B[B_start+35840];
			kernel[c_kernel+8192] = B[B_start+40960];
			kernel[c_kernel+9216] = B[B_start+46080];
			kernel[c_kernel+10240] = B[B_start+51200];
			kernel[c_kernel+11264] = B[B_start+56320];
			kernel[c_kernel+12288] = B[B_start+61440];
			kernel[c_kernel+13312] = B[B_start+66560];
			kernel[c_kernel+14336] = B[B_start+71680];
			kernel[c_kernel+15360] = B[B_start+76800];

			__syncthreads();

			// after sync, all 32 kernel values are ready to be used? How?
			// 16384 / 512 = 32, each thread gets 32 values to use
			// the threads with the same tY share the same 32 values, but how?   
			float sum = 0;
			// int y_tmp = (tY<<7)+(k<<5);
			int y_tmp = tY*512 + (k<<5); //*32
			for (int j = 0; j < 32; j++) {
				sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
			}
			out[c_kernel] += sum;
			__syncthreads();
		}
        // assumes C[T_offset] is initialized with 0.f    
		
		C[C_offset] += out[c_kernel];
		// if(tX == 0 && tY == 1 && Tile == 0 && Part == 0){
		// 	printf("%d, %d, %ld, %f, %f \n", l, i, C_offset, C[C_offset],  out[c_kernel]);
		// }
		
	}
	}
	
}

int kernel_2560() {
	float *input_ = get_parameter(inputName2560, 16*16*2560);
	float *bias = get_parameter(biasName2560, 2560);
	float *W = get_parameter(weight_NCHW_Name2560, 3*3*2560*2560);
	float *input, *output, *l_weights, *l_bias;
	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;

	uint64_t st1 = 0, st2 = 0, st3 = 0;
	uint64_t et1 = 0, et2 = 0, et3 = 0;

	cudaError_t s;



	/////////////////////////////////

	// My Kernel

	/////////////////////////////////
	float *kernel = get_parameter(weight_winograd_Name2560, 36*2560*2560), *t_input, *ip;
	int nInput = 16*16*2560, nOutput = 16*16*2560, nWeights = 36*2560*2560, nBias = 2560, 
	    nTransInput = 16*6*6*2560, nInnerProd = 16*6*6*2560;
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

	bnBias = get_parameter(bnBias_winograd_Name2560, 2560);
	bnScale = get_parameter(bnScale_winograd_Name2560, 2560);
	cudaMalloc((void **) &l_bnBias, nBias<<2);
	cudaMalloc((void **) &l_bnScale, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);

	float *tmp = (float*)malloc(nOutput*4);

	nT1 = getTimeMicroseconds64();

	kernel_2560_winograd_BtdB <<<dim3(4, 4, 16), dim3(160, 6), (6*6*160)<<2 >>> (input, t_input);
	// cudaCheckError();
	int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(kernel_2560_OuterProduct_2560, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
	kernel_2560_OuterProduct_2560<<<dim3(36, 8), dim3(512, 2), (2*512 + 16*2*512 + 2*512)<<2 >>> (t_input, l_weights, ip);
	// cudaCheckError();
	kernel_2560_winograd_AtIA <<<dim3(4, 4, 2560), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
	// cudaCheckError();
	cudaDeviceSynchronize();

	nT2 = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2-nT1); 

	s = cudaMemcpy(tmp, output, nOutput<<2, cudaMemcpyDeviceToHost);
	// printf("A. %s\n", cudaGetErrorName(s));
	cudaCheckError();

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
	

	// float *conv_cpu =  (float*)malloc(14*14*2560*4);

    // nT1_cudnn = getTimeMicroseconds64();
	// compute_cpu(input_, W, conv_cpu, 16, 2560, 1);
    // nT2_cudnn = getTimeMicroseconds64();
	// printf("TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);  
	
	free(input_);
	free(W);


	// output_checker(tmp, conv_cpu, 14, 2560, 1);
	// free(conv_cpu);
	free(tmp);

	return ((nT2-nT1) << 16);
}