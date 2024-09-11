#ifndef __KERNEL320_WINOGRAD_H__
#define __KERNEL320_WINOGRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName320[] = "data/input_14_1_320.bin";
const char biasName320[] = "data/bias_320.bin";
const char weight_winograd_Name320[] = "data/weight_winograd_320_320.bin";
const char weight_NCHW_Name320[] = "data/weight_NCHW_320_320.bin";

const char bnBiasName320[] = "data/bnBias_320.bin";
const char bnScaleName320[] = "data/bnScale_320.bin";
const char bnBias_winograd_Name320[] = "data/bnBias_winograd_320.bin";
const char bnScale_winograd_Name320[] = "data/bnScale_winograd_320.bin";
const char eMeanName320[] = "data/eMean_320.bin";
const char eVarName320[] = "data/eVar_320.bin";

int kernel_320();

#ifdef __cplusplus
}
#endif

#endif
