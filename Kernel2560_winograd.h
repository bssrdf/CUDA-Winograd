#ifndef __KERNEL2560_WINOGRAD_H__
#define __KERNEL2560_WINOGRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName2560[] = "data/input_14_1_2560.bin";
const char biasName2560[] = "data/bias_2560.bin";
const char weight_winograd_Name2560[] = "data/weight_winograd_2560_2560.bin";
const char weight_NCHW_Name2560[] = "data/weight_NCHW_2560_2560.bin";

const char bnBiasName2560[] = "data/bnBias_2560.bin";
const char bnScaleName2560[] = "data/bnScale_2560.bin";
const char bnBias_winograd_Name2560[] = "data/bnBias_winograd_2560.bin";
const char bnScale_winograd_Name2560[] = "data/bnScale_winograd_2560.bin";
const char eMeanName2560[] = "data/eMean_2560.bin";
const char eVarName2560[] = "data/eVar_2560.bin";

int kernel_2560();

#ifdef __cplusplus
}
#endif

#endif
