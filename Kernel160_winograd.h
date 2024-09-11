#ifndef __KERNEL160_WINOGRAD_H__
#define __KERNEL160_WINOGRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName160[] = "data/input_14_1_160.bin";
const char biasName160[] = "data/bias_160.bin";
const char weight_winograd_Name160[] = "data/weight_winograd_160_160.bin";
const char weight_NCHW_Name160[] = "data/weight_NCHW_160_160.bin";

const char bnBiasName160[] = "data/bnBias_160.bin";
const char bnScaleName160[] = "data/bnScale_160.bin";
const char bnBias_winograd_Name160[] = "data/bnBias_winograd_160.bin";
const char bnScale_winograd_Name160[] = "data/bnScale_winograd_160.bin";
const char eMeanName160[] = "data/eMean_160.bin";
const char eVarName160[] = "data/eVar_160.bin";

int kernel_160();

#ifdef __cplusplus
}
#endif

#endif
