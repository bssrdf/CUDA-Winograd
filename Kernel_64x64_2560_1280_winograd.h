#ifndef __KERNEL2560_1280_64_WINOGRAD_H__
#define __KERNEL2560_1280_64_WINOGRAD_H__

#ifdef __cplusplus
extern "C" {
#endif

const char inputName2560_1280_64[] = "data/input_62_1_2560.bin";
const char biasName2560_1280_64[] = "data/bias_2560.bin";
const char weight_winograd_Name2560_1280_64[] = "data/weight_winograd_2560_1280.bin";
const char weight_NCHW_Name2560_1280_64[] = "data/weight_NCHW_2560_1280.bin";

const char bnBiasName2560_1280_64[] = "data/bnBias_2560.bin";
const char bnScaleName2560_1280_64[] = "data/bnScale_2560.bin";
const char bnBias_winograd_Name2560_1280_64[] = "data/bnBias_winograd_2560.bin";
const char bnScale_winograd_Name2560_1280_64[] = "data/bnScale_winograd_2560.bin";
const char eMeanName2560_1280_64[] = "data/eMean_2560.bin";
const char eVarName2560_1280_64[] = "data/eVar_2560.bin";

int kernel_2560_1280_64();

#ifdef __cplusplus
}
#endif

#endif
