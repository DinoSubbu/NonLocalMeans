#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void NonLocalMeans(float* img, float* imgTemp, float* C)

 { 
  
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
}