#include "util.h"
#include <time.h>
#include "math.h"

uint64_t getTimeMicroseconds64() {
  uint64_t nTime;
  struct timespec tSpec;

  clock_gettime(CLOCK_REALTIME, &tSpec);

  nTime = (uint64_t)tSpec.tv_sec * 1000000 + (uint64_t)tSpec.tv_nsec / 1000;
  return nTime;
}

float* transpose(float* weight, int h, int w) {
  float* new_weight = (float*)malloc(w * h * 4);
  int i, j;
  for (i = 0; i < w; ++i) {
    for (j = 0; j < h; ++j) {
      new_weight[j * w + i] = weight[i * h + j];
    }
  }

  free(weight);
  return new_weight;
}

float* get_parameter(const char* filename, int size) {
  float* parameter = (float*)malloc(size * 4);
  if (!parameter) {
    printf("Bad Malloc\n");
    exit(0);
  }
  FILE* ptr = fopen(filename, "rb");

  if (!ptr) {
    printf("Bad file path: %p, %s\n", ptr, strerror(errno));
    exit(0);
  }
  fread(parameter, size * 4, 1, ptr);

  fclose(ptr);
  return parameter;
}

float output_checker(float* A, float* B, int len, int channel, int shift) {
  int error_cnt = 0, i, j, k;
  float max_error = 0;
  int ie=-1, je=-1, ke = -1;
  float r_win = 0.f, r_cpu = 0.f;
  for (i = 0; i < len; i++) {
    for (j = 0; j < len; j++) {
      for (k = 0; k < channel; k++) {
        float diff = fabs(
            A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k] -
            B[(i * len + j) * channel + k]);
        // if (k == 0)
        // if(i == 4 && j == 0)
          //  printf(" i, j, k, wi, cpu:  %d, %d, %d, %f, %f \n", i, j, k,
          //       A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k],
          //         B[(i * len + j) * channel + k]);
        if (diff > 1e-5){
          error_cnt++;           
        }
        // if (diff > 1.e-4){
        //    printf(" i, j, k, wi, cpu:  %d, %d, %d, %f, %f \n", i, j, k,
        //         A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k],
        //           B[(i * len + j) * channel + k]);
        //    return 0.f;
        // }

        // else{
        //       printf(" i, j, wi, cpu:  %d, %d, %d, %f, %f \n", i, j, k,
        //         A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k],
        //           B[(i * len + j) * channel + k]);
        // }        
        if (diff > max_error){
          max_error = diff;
          r_win = A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k];
          r_cpu =  B[(i * len + j) * channel + k];
          ie = i; je = j; ke = k;
        }
      }
    }
  }
  printf("[max_error: %f][error_cnt: %d]\n", max_error, error_cnt);
  printf("[max_error at (i,j,k) : (%d, %d, %d) \n", ie, je, ke);
  printf("[wino: %f][cpu: %f]\n", r_win, r_cpu);
}


void compute_cpu(float* A, float* W,  float *C, int len, int channel, int shift){
    int i, j, k;
    int stride = len*channel;
    
      for (j = 1; j < len - 1; j++){
        for (i = 1; i < len - 1; i++){

          for (int l = 0; l < channel; l++){
          float sum = 0.f;
          for (k = 0; k < channel; k++){
            for(int ii=-1; ii < 2; ii++){ 
              for(int jj=-1; jj < 2; jj++){ 
                 int x = i+ii, y = j+jj; 
                //  int idx1 = x+k*len+y*stride;
                //  int idx2 = l*channel*channel+k*channel+(ii+1)*3+jj+1;
                //  printf("%d, %d \n", idx1, idx2);
                //  sum += A[x+k*len+y*stride]*W[l*channel*3*3+k*3*3+(ii+1)*3+jj+1];
                 sum += A[k + (x * len + y)*channel]*W[l*channel*3*3+k*3*3+(ii+1)*3+jj+1];
                //  if(fabs(sum) > 1.e10)
                //     printf("wrong: %f, %f \n", A[x+k*len+y*stride],W[l*channel*3*3+k*3*3+(ii+1)*3+jj+1]);
              }
            }
          }
          // int idx3 = i+l*(len-2)+j*(len-2)*channel;
          // printf("idx3 = %d \n", idx3);
          C[((i-1) * (len-2) + j-1) * channel + l] = sum > 0? sum : 0.f;    
      }
    }
    }

      

}
