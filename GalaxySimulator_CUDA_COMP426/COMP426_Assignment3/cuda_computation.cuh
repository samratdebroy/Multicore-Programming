#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cuda_helper\helper_math.h>

cudaError_t integrate_with_cuda(float dt, float2* pos, float2* vel, const float2* acc, unsigned int size);