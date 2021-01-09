#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


float calculateDiff(__read_only image2d_t imageInput,
		int2 pos1,
		int2 pos2,
		int filterSize,
		int iteration) {
	int FILTER_SIZE = 1;
    float res = 0.0f;
    for(int offset_x = -FILTER_SIZE; offset_x <= FILTER_SIZE; ++offset_x) {
        for(int offset_y = -FILTER_SIZE; offset_y <= FILTER_SIZE; ++offset_y) {
            int2 offset = (int2)(offset_x, offset_y);
            float diff = read_imagef(imageInput, sampler, pos1 + offset).x/255.0f - read_imagef(imageInput, sampler, pos2 + offset).x/255.0f;
            diff = diff*diff;
            res += diff;
        }
    }
    return res;
}

__kernel void nonLocalMeans(
        __read_only image2d_t imageInput,
		__write_only image2d_t imageOutput,
        __private int searchSize,
        __private int filterSize,
        __private float parameterH,
        __private int iteration) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int j = get_global_id(0);
    int i = get_global_id(1);
    float sumBottom = 0.0f;
    float sumTop = 0.0f;
    int SEARCH_SIZE = 11;

    // Loop over search region
    for(int searchOffsetX = -SEARCH_SIZE; searchOffsetX <= SEARCH_SIZE; ++searchOffsetX) {
        for(int searchOffsetY = -SEARCH_SIZE; searchOffsetY <= SEARCH_SIZE; ++searchOffsetY) {
            int2 searchOffsetAtScale = (int2)(searchOffsetX*(iteration+1), searchOffsetY*(iteration+1));
            float diff = calculateDiff(imageInput, pos, pos + searchOffsetAtScale, filterSize, iteration);
            diff = native_exp(-diff/(2.0f*parameterH*parameterH));// / (2.0f*parameterH*parameterH);
            sumBottom += diff;
            sumTop += diff*read_imagef(imageInput, sampler, pos + searchOffsetAtScale).x/255.0f;
        }
    }

    //output[i+j] = (sumTop/sumBottom)*255.0f;
    write_imagef(imageOutput, pos, (uchar)clamp((sumTop/sumBottom)*255.0f, 0.0f, 255.0f ));
}
