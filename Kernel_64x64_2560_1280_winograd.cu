#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudnn.h"
#include <xmmintrin.h>
#include <immintrin.h>

//#include "cudnn.h"
#include "util.h"
#include "Kernel_64x64_2560_1280_winograd.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

#define MY_KERNEL 0

#define d(input, i, j, Inz) ( input[Inz + i*960 + (j*160)] )
__global__ void kernel_2560_1280_64_winograd_BtdB(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Part = blockIdx.z, Iny1 = threadIdx.y, Inz = threadIdx.x;
	int Iny = Iny0+Iny1, stride_r = 163840, stride_c = 2560; //  163840 = 64*2560
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
        // 655360 = 16*16*2560; 
		pOutputs[(Iny1 + i*6)*655360 + ((blockIdx.x<<2)+blockIdx.y)*2560 + Inz + Part*160] = BTdB[i];
		// if((Iny1 + i*6)*655360 + ((blockIdx.x<<4)+blockIdx.y)*2560 + Inz + Part*160 == 649639){
		// 	printf(" AA, %d, %d, %d, %d, %d, %d \n", blockIdx.x, blockIdx.y, blockIdx.z, Inz, Iny1, i);
		// }
	}
}

__global__ void kernel_2560_1280_64_winograd_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
	int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
	int c_input = Inx*6 + Iny;

	__shared__ float bias, scale;
	extern __shared__ float input[];

	input[c_input] = pInputs[c_input*16*16*1280 + (Tilex*16+Tiley)*1280 + kz];
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
			// pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*1280 + kz] = o > 0 ? o : 0;
			// when Tilex=Tiley=kz=Inx=Iny=0, the first index to POutputs is (64+1)*1280,
			// which is to skip all channels over the first row plus the first channel on the 
			// 2nd row, because they are padded 
			pOutputs[(((Tilex<<2)+1+Inx)*64 + (Tiley<<2)+1)*1280 + kz] = o;
			break;
		case 1:
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
			o = (input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*64 + (Tiley<<2)+2)*1280 + kz] = o;
			break;
		case 2:
			// if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
			o = (input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]);
			pOutputs[(((Tilex<<2)+1+Inx)*64 + (Tiley<<2)+3)*1280 + kz] = o;
			break;
		case 3:
			// if (Tiley == 3) break;
			x = Inx*6;
			// o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
			o = (input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]);
			pOutputs[(((Tilex<<2)+1+Inx)*64 + (Tiley<<2)+4)*1280 + kz] = o;
			break;
	}
}

__global__ void kernel_2560_1280_64_OuterProduct_2560_1280(float *A, float *B, float *C) {
	int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
	u_int64_t c_input = tY*1280 + tX, 
	    c_kernel = tY*256 + tX,  
		c_tr_in = tY*2560 + tX, 
		T_offset = Tile*655360 + Part*2560*4 + c_tr_in, // 655360 = 16*16*2560, 1024 = 512*2
		C_start = Tile*327680 + Part*1280*4 + c_input, // 327680 = 16*16*1280, 1024 = 512*2
	    B_offset = Tile*3276800 + c_input; // 3276800 = 1280*2560
    
	extern __shared__ float input[];
	// int B_stride[32] = {0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872};
    int B_stride[32] = {0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936};
    // we need to compute 2560 products and sum them up. 
	// the loop below iterates 5 times and each step computes 512 products and accumulates 
    float *kernel = input + 1024, *out = kernel + 8192; // 8192 = 256*4*8	

	for (int l = 0; l < 5; l++){		
		u_int64_t B_start_l = B_offset + l*256;
		u_int64_t C_offset = C_start + l*256;
		out[c_kernel] = 0.0f;

		for (int i = 0; i < 10; i++){

			u_int64_t offset = T_offset + (i<<8); // *256
			// if(tX == 0 && tY == 0 && Tile == 0 && Part == 0){
			// 	printf("%d, %d, %ld \n", l, i, offset);
			// }
			input[c_kernel] = A[offset]; // 36*4 blocks, each block loads 256*4 values with 256*4 threads (1 value/t) 
			
			// the loop below iterates  times and each step computes 32 products and accumulates 
			// the partial sum in shared memory
			// each thread block will process 2560*1280 kernel values
			// 36 blocks in each row (8 rows in total) will use the same 2560_1280*2560_1280 kernel values
			u_int64_t B_start_i = B_start_l + i*256*1280;
			for (int k = 0; k < 8; k++) {
				u_int64_t B_start = B_start_i + k*32*1280 ; // 
				// each thread loads 8 kernel values
				// the whole thread block can load 256*4*8 = 16384 values
				// incrementing k by 1 will shift by 16384 = 256*4*8 values 

				// if we have 4 threads in y direction, these 4 with the same tY 
				// alternating get 4 lines of portion of 1280 weights
				// because the layout of weights has out_dim as the first (fast) dimension,
				// to get weights along in_dim, we have to iterating over rows		
				kernel[c_kernel] = B[B_start];
				kernel[c_kernel+1024] = B[B_start+5120]; // 5120 = 1280*4 (out_dim * number of threads in y dir)
				kernel[c_kernel+2048] = B[B_start+10240];
				kernel[c_kernel+3072] = B[B_start+15360];
				kernel[c_kernel+4096] = B[B_start+20480];
				kernel[c_kernel+5120] = B[B_start+25600];
				kernel[c_kernel+6144] = B[B_start+30720];
				kernel[c_kernel+7168] = B[B_start+35840];

				__syncthreads();

				// after sync, all 32 kernel values are ready to be used? How?
				// 16384 / 512 = 32, each thread gets 32 values to use
				// the threads with the same tY share the same 32 values, but how?   
				float sum = 0;
				// int y_tmp = (tY<<7)+(k<<5);
				int y_tmp = tY*256 + (k<<5); //*32
				for (int j = 0; j < 32; j++) {
					sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
				}
				out[c_kernel] += sum;
				__syncthreads();
			}
			// assumes C[T_offset] is initialized with 0.f    
		}
		C[C_offset] = out[c_kernel];
		// if(tX == 0 && tY == 1 && Tile == 0 && Part == 0){
		// 	printf("%d, %d, %ld, %f, %f \n", l, i, C_offset, C[C_offset],  out[c_kernel]);
		// }
		
	}
	
}

int kernel_2560_1280_64() {
	float *input_ = get_parameter(inputName2560_1280_64, 64*64*2560);
	float *bias = get_parameter(biasName2560_1280_64, 2560);
	float *W = get_parameter(weight_NCHW_Name2560_1280_64, 3*3*2560*1280);
	float *input, *output, *l_weights, *l_bias;
	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;

	uint64_t st1 = 0, st2 = 0, st3 = 0;
	uint64_t et1 = 0, et2 = 0, et3 = 0;

	cudaError_t s;



	/////////////////////////////////

	// My Kernel

	/////////////////////////////////
	float *kernel = get_parameter(weight_winograd_Name2560_1280_64, 36*2560*1280), *t_input, *ip;
	int nInput = 64*64*2560, nOutput = 64*64*1280, nWeights = 36*2560*1280, nBias = 2560, 
	    nTransInput = 16*16*6*6*2560, nInnerProd = 16*16*6*6*1280;
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

	bnBias = get_parameter(bnBias_winograd_Name2560_1280_64, 2560);
	bnScale = get_parameter(bnScale_winograd_Name2560_1280_64, 2560);
	cudaMalloc((void **) &l_bnBias, nBias<<2);
	cudaMalloc((void **) &l_bnScale, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);

	float *tmp = (float*)malloc(nOutput*4);

	float *tmps = (float*)malloc(nTransInput<<2);

	int iterations = 10;
	int ll = 649639;

	float mi, mx;
	int mi_i, mx_i;
    
    CUevent hStart, hStop;
	float ms, avg;
	cudaEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC); // CU_EVENT_DEFAULT
	cudaEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC);

	int maxbytes = 98304; // 96 KB
    cudaFuncSetAttribute(kernel_2560_1280_64_OuterProduct_2560_1280, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    // warm up
	kernel_2560_1280_64_winograd_BtdB <<<dim3(16, 16, 16), dim3(160, 6), (6*6*160)<<2 >>> (input, t_input);
	s = cudaMemcpy(tmps, t_input, nTransInput<<2, cudaMemcpyDeviceToHost);
	find_minmax(tmps, nTransInput, &mi, &mx, &mi_i, &mx_i);
	// for (int l = 0; l < nTransInput; l++){
	// 	if(tmps[l] != 0.0){
	// 		printf("at %d, non zero \n ", l);
	// 		break;
	// 	}		   
	// }
	printf("t_input: %s, %f(%d), %f (%d), %f \n", cudaGetErrorName(s), mi, mi_i, mx, mx_i, tmps[ll]);
	kernel_2560_1280_64_OuterProduct_2560_1280<<<dim3(36, 64), dim3(256, 4), (4*256 + 8*4*256 + 4*256)<<2 >>> (t_input, l_weights, ip);
	kernel_2560_1280_64_winograd_AtIA <<<dim3(16, 16, 1280), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
    cudaDeviceSynchronize();

    cudaEventRecord( hStart, NULL ) ;
    for(int iter=0; iter<iterations; iter++){ 
	kernel_2560_1280_64_winograd_BtdB <<<dim3(16, 16, 16), dim3(160, 6), (6*6*160)<<2 >>> (input, t_input);
	s = cudaMemcpy(tmps, t_input, nTransInput<<2, cudaMemcpyDeviceToHost);
	find_minmax(tmps, nTransInput, &mi, &mx, &mi_i, &mx_i);
	printf("t_input: %s, %d, %f (%d), %f (%d), %f \n", cudaGetErrorName(s), iter, mi, mi_i, mx, mx_i, tmps[ll]);
	kernel_2560_1280_64_OuterProduct_2560_1280<<<dim3(36, 64), dim3(256, 4), (4*256 + 8*4*256 + 4*256)<<2 >>> (t_input, l_weights, ip);
	kernel_2560_1280_64_winograd_AtIA <<<dim3(16, 16, 1280), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
	}
	// cudaDeviceSynchronize();
	cudaEventRecord( hStop, NULL );
    cudaEventSynchronize( hStop ) ;
    cudaEventElapsedTime( &ms, hStart, hStop ) ;
    avg = ms / iterations;

	printf("MyKernal: TotalTime = %.1f ms and avg = %.3f ms\n", ms, avg); 

	s = cudaMemcpy(tmp, output, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("A. %s, %f \n", cudaGetErrorName(s), tmp[57183]);
	cudaCheckError();

	cudaFree(t_input);
	cudaFree(output);
	cudaFree(l_weights);
	cudaFree(l_bias);
	// cudaFree(l_bnBias);
	// cudaFree(l_bnScale);
	cudaFree(ip);
	// cudaFree(input);


	// free(kernel);
	free(bnScale);
	free(bnBias);
	// free(bias);
	

    bnBias = get_parameter(bnBiasName2560_1280_64, 2560);
	bnScale = get_parameter(bnScaleName2560_1280_64, 2560);
	float* eMean = get_parameter(eMeanName2560_1280_64, 2560);
	float* eVar = get_parameter(eVarName2560_1280_64, 2560);
	float *l_eMean, *l_eVar;

    nInput = 64*64*2560, nOutput = 62*62*1280, nWeights = 3*3*2560*1280, nBias = 2560;

	cudaMalloc((void **) &output, nOutput<<2);
	cudaMalloc((void **) &l_weights, nWeights<<2);
	cudaMalloc((void **) &l_bias, nBias<<2);
	// cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bias, bias, nBias<<2, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &l_eMean, nBias<<2);
	cudaMalloc((void **) &l_eVar, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_eMean, eMean, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_eVar, eVar, nBias<<2, cudaMemcpyHostToDevice);

	cudaMemset((void *) output, 0, nOutput<<2);	
	
	
	cudaMemcpy(l_weights, W, nWeights<<2, cudaMemcpyHostToDevice);

	float *tmp_cudnn =  (float*)malloc(nOutput<<2);

	cudnnStatus_t status;
	float one = 1.0, zero = 0.0;
	int size;

	cudnnHandle_t handle;
	status = cudnnCreate(&handle);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");

	cudnnTensorDescriptor_t xdesc, ydesc, bdesc;
	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	status = cudnnCreateTensorDescriptor(&xdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
	status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 2560, 64, 64);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
	status = cudnnCreateTensorDescriptor(&ydesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
	status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 1280, 62, 62);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
	status = cudnnCreateFilterDescriptor(&wdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
	status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1280, 2560, 3, 3);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");
	status = cudnnCreateTensorDescriptor(&bdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed8\n");
	status = cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 2560, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed9\n");
	cudnnConvolutionDescriptor_t conv_desc;
	status = cudnnCreateConvolutionDescriptor(&conv_desc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
	status = cudnnSetConvolution2dDescriptor(conv_desc, 0,0, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
	if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");

	cudnnActivationDescriptor_t act_desc;
	status = cudnnCreateActivationDescriptor(&act_desc);  
	if (status != CUDNN_STATUS_SUCCESS) printf("failed12\n");
	status = cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed13\n");

	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
	status = cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed14\n");
	status = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2560, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed15\n");

	// cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)6;
	// cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
	// cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	// cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;

	status = cudnnGetConvolutionForwardWorkspaceSize(handle,
	   xdesc,
	   wdesc,
	   conv_desc,
	   ydesc,
	   algo,
	   (size_t *)&(size));

	float *extra;
	cudaMalloc((void **) &extra, size);


	status = cudnnConvolutionForward(handle, &one,
			xdesc, input, wdesc, l_weights, 
			conv_desc, algo, 
			extra, size, &zero,
			ydesc, output);
	if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed1\n");
	cudaDeviceSynchronize();
	// nT1_cudnn = getTimeMicroseconds64();
    cudaEventRecord( hStart, NULL ) ;
    for(int iter=0; iter<iterations; iter++){  
		status = cudnnConvolutionForward(handle, &one,
			xdesc, input, wdesc, l_weights, 
			conv_desc, algo, 
			extra, size, &zero,
			ydesc, output);
	}
	cudaEventRecord( hStop, NULL );
    cudaEventSynchronize( hStop ) ;
    cudaEventElapsedTime( &ms, hStart, hStop ) ;
    avg = ms / iterations;
	printf("Cudnn: TotalTime = %.1f ms and avg = %.3f ms\n", ms, avg); 
	// if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed1\n");

	// status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
	// 	&one, &zero, 
	// 	ydesc, output, ydesc, output,
	// 	bnScaleBiasMeanVarDesc, l_bnScale, l_bnBias, l_eMean, l_eVar, CUDNN_BN_MIN_EPSILON);
	// if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed2\n");

	// status = cudnnActivationForward(handle, act_desc, &one,
	// 	ydesc, output, &zero,
	// 	ydesc, output);
	// if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed3\n");

	// cudaDeviceSynchronize();
	// nT2_cudnn = getTimeMicroseconds64();
	// printf("cuDNN TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);
	
	s = cudaMemcpy(tmp_cudnn, output, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));



    cudaFree(extra);
	

	// cudaFree(t_input);
	cudaFree(output);
	cudaFree(l_weights);
	cudaFree(l_bias);
	cudaFree(l_bnBias);
	cudaFree(l_bnScale);
	// cudaFree(ip);
	cudaFree(input);


	free(kernel);
	free(bnScale);
	free(bnBias);
	free(bias);

	
	

	// float *conv_cpu =  (float*)malloc(14*14*1280*4);

    // nT1_cudnn = getTimeMicroseconds64();
	// compute_cpu(input_, W, conv_cpu, 16, 2560, 1280, 1);
    // nT2_cudnn = getTimeMicroseconds64();
	// printf("TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);  
	
	free(input_);
	free(W);
    

	output_checker(tmp, tmp_cudnn, 62, 1280, 1);
	// free(conv_cpu);
	free(tmp);
    free(tmp_cudnn);
	return ((nT2-nT1) << 16);
}