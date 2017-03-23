#ifndef KERNEL_H
#define KERNEL_H

struct uchar4;
struct int3;
struct float2;
struct float3;
struct float4;

void kernelLauncher(uchar4 *d_out, float2 *d_vol, int w, int h,
  int3 volSize, int method, int zs, float theta, float threshold,
  float dist, int id);
 void volumeKernelLauncher(float2 *d_vol, float3 *d_coords, int3 volSize, 
 	float3 voxDim, float3 boxMin, int atomCount, float *d_charge);

#endif