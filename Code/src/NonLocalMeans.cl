#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif


#define imgH 480
#define imgW 640
#define h 1
#define patchW 3



__kernel void NonLocalMeansFilter(__global float* img, __global float* imgTemp, __global float* C)
{ 
  
	int j = get_global_id(1);
	int i = get_global_id(0);

 for(int k=i; k<imgH - patchW + 1; k++)
  {
    for(int l=0; l<imgW - patchW + 1; l++)
    {

      if(l != j)
      {
        float v = 0;

        for(int p=k; p<k+patchW; p++)
        {
          for(int q=l; q<l+patchW; q++)
          {
            v += (img[(i+p-k)*imgW + j+q-l] - img[p*imgW + q]);
            v = v*v;
          }
        }

          float w = exp(-v/(h*h));

		 imgTemp[i*imgW + j] += w * img[k*imgW + l];
	     C[i*imgW + j] += w;
	     imgTemp[k*imgW + l] += w * img[i*imgW + j];
	     C[k*imgW + l] += w;
        }
      }
  }
} 
