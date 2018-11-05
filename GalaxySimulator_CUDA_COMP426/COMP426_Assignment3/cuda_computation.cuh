#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cuda_helper\helper_math.h>

cudaError_t reset_quadtree_with_cuda(float2* pos, float* mass, int* child);
cudaError_t compute_forces_and_integrate_with_cuda(float dt, float2* pos, float2* vel, const float2* acc, float* mass, int* child, float4* min_max_extents);
