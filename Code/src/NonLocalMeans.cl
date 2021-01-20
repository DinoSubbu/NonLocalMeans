#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void nonLocalMeansFilter(
        __read_only image2d_t imageInput,
        __write_only image2d_t imageOutput,
        __private int searchSize,
        __private int filterSize,
        __private float parameterH,
        __private int iteration
        ) { 
    const int2 pos = {get_global_id(0), get_global_id(1)};
    float sumBottom = 0.0f;
    float sumTop = 0.0f;

    // Loop over search region
    for(int searchOffsetX = -SEARCH_SIZE; searchOffsetX <= SEARCH_SIZE; ++searchOffsetX) {
        for(int searchOffsetY = -SEARCH_SIZE; searchOffsetY <= SEARCH_SIZE; ++searchOffsetY) {
            int2 searchOffsetAtScale = {searchOffsetX*(iteration+1), searchOffsetY*(iteration+1)};
            float diff = calculateDiff(imageInput, pos, pos + searchOffsetAtScale, filterSize, iteration);
            diff = native_exp(-diff/(2.0f*parameterH*parameterH));// / (2.0f*parameterH*parameterH);
            sumBottom += diff;
            sumTop += diff*read_imageuf(imageInput, sampler, pos + searchOffsetAtScale).x/255.0f;
        }
    }

    write_imageui(imageOutput, pos, (uchar)clamp((sumTop/sumBottom)*255.0f, 0.0f, 255.0f ));
}