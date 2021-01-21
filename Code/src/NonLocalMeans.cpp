//////////////////////////////////////////////////////////////////////////////
// Non Local Means
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <boost/lexical_cast.hpp>

using namespace std;

// TODO: Remove constants later
int imgWidth = 640;
int imgHeight = 480;

//////////////////////////////////////////////////////////////////////////////
// CPU implementation of NonLocal Means Algorithm
//////////////////////////////////////////////////////////////////////////////
void nlmHost(float h_input[][640],
			float** h_outputCpu,
			const float h,
			const int patchW,
			const int width,
			const int height)
{

	cout<<"Starting to run nlm algorithm on cpu"<<endl;

	float h_temp[height][width], C[height][width];

	for(int i=0; i<height - patchW + 1; i++)
	{
		std::cout<<"Debug:: I VALUE: "<<i<<std::endl;
	   for(int j=0; j<width - patchW + 1; j++)
	   {
	      for(int k=i; k<height - patchW + 1; k++)
	      {
	         for(int l=0; l<width - patchW + 1; l++)
	         {
                if(l != j)
	            {
	              float v = 0;
                  for(int p=k; p<k+patchW; p++)
	              {
	                 for(int q=l; q<l+patchW; q++)
	                 {
	                    // Euclidean distance distance calculation
	                	 v += (h_input[i+p-k][j+q-l] - h_input[p][q]) * (h_input[i+p-k][j+q-l] - h_input[p][q]);
	                 }
	              }

                  // Weight matrix calculation => exp(-v/h^²)
	              float w = exp(-v/(h*h));

	              // Multiply pixels with weight matrix and add them up!
				  h_temp[i][j] += w * h_input[k][l];
				  C[i][j] += w;
				  h_temp[k][l] += w * h_input[i][j];
				  C[k][l] += w;
	           }
	         }
	     }
	   }
	}
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context	
	//cl::Context context(CL_DEVICE_TYPE_GPU);

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId] (), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT (deviceNr > 0);
	ASSERT ((size_t) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "src/NonLocalMeans.cl");
	
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Declare some values
	// todo use this variables later
/* 	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes */

	// Use an image (Valve.pgm) as input data
	std::size_t inputWidth, inputHeight;
	std::vector<float> inputData;
	
	// Allocate space for output data from CPU and GPU on the host, @ dinesh 
	float h_input[imgHeight][imgWidth] = {0}, h_outputGpu[imgHeight][imgWidth] = {0};

	// Allocate space for input and output data on the device
	cl::Buffer d_Img(context, CL_MEM_READ_WRITE, inputWidth * inputHeight * sizeof(float) );
    cl::Buffer d_ImgTemp(context, CL_MEM_READ_WRITE,inputWidth * inputHeight * sizeof(float) );
    cl::Buffer d_C(context, CL_MEM_READ_WRITE, inputWidth * inputHeight * sizeof(float) );

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
    memset(h_input, 255, inputWidth * inputHeight * sizeof(float));
    memset(h_outputGpu, 255, inputWidth * inputHeight * sizeof(float));
    memset(h_outputCpu, 255, inputWidth * inputHeight * sizeof(float));
	
	//TODO: GPU
    queue.enqueueWriteBuffer(d_Img,true,0,inputWidth * inputHeight * sizeof(float), h_input);
    queue.enqueueWriteBuffer(d_ImgTemp,true,0,inputWidth * inputHeight * sizeof(float),h_outputGpu);
    queue.enqueueWriteBuffer(d_C,true,0,inputWidth * inputHeight * sizeof(float),h_outputGpu);

	//////// Load input data ////////////////////////////////
	std::cout<<"inputWidth:: "<<inputWidth;
	std::cout<<"inputHeight:: "<<inputHeight;
	Core::readImagePGM("Valve.pgm", inputData, inputWidth, inputHeight);
	std::cout<<"Width:: "<<inputWidth;
	for (size_t j = 0; j < countY; j++) {
		for (size_t i = 0; i < countX; i++) {
			h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}

	// Copy input data to device
	queue.enqueueWriteBuffer(d_Img,true,0,inputWidth * inputHeight * sizeof(float),h_input);

	// Do calculation on the host side
	float h = 1;
	int patchW = 3;
	nlmHost(h_input, h_outputCpu, h, patchW, inputWidth, inputHeight);

	//////// Store CPU output image ///////////////////////////////////
	// h_outputCpuVector declaration
	Core::writeImagePGM("output_nlm_cpu.pgm", h_outputCpuVector, inputHeight, inputWidth);
	std::cout<<"Image written"<<std::endl;

	// Reinitialize output memory to 0xff
	memset(h_outputGpu, 255, inputWidth * inputHeight * sizeof(float));

	//GPU
	queue.enqueueWriteBuffer(d_C,true,0,inputWidth * inputHeight * sizeof(float),h_outputGpu);	

	// Copy input data to device
	queue.enqueueWriteBuffer(d_Img, true, 0, inputWidth * inputHeight * sizeof(float),h_input);

    // Create a kernel object
    std::cout<<"Creating kernel object"<<std::endl;
	cl::Kernel NLM(program, "NonLocalMeansFilter");

	std::cout << std::endl;
	
	// Launch kernel on the device
     NLM.setArg<cl::Buffer>(0, d_Img);
     NLM.setArg<cl::Buffer>(1, d_ImgTemp);
     NLM.setArg<cl::Buffer>(2, d_C);

	 //range check
     queue.enqueueNDRangeKernel(NLM,
                            cl::NullRange,
                            cl::NullRange,
                            cl::NullRange
                        );

    // Copy output data back to host
    queue.enqueueReadBuffer(d_C,true,0,inputWidth * inputHeight * sizeof(float),h_outputGpu,NULL,NULL);

	// Print performance data
    //TODO

	//////// Store GPU output image ///////////////////////////////////
	std::cout<<"Creating Output Image"<<std::endl;
	Core::writeImagePGM("output_nonlocal_gpu.pgm", h_outputGpu, inputWidth, inputHeight);

	// Check whether results are correct
	std::size_t errorCount = 0;
	for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
		for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
			size_t index = i + j * countX;
			// Allow small differences between CPU and GPU results (due to different rounding behavior)
			if (!(std::abs (h_outputCpu[index] - h_outputGpu[index]) <= 1e-5)) {
				if (errorCount < 15)
					std::cout << "Result for " << i << "," << j << " is incorrect: GPU value is " << h_outputGpu[index] << ", CPU value is " << h_outputCpu[index] << std::endl;
					else if (errorCount == 15)
						std::cout << "..." << std::endl;
					errorCount++;
				}
			}
		}
		if (errorCount != 0) {
			std::cout << "Found " << errorCount << " incorrect results" << std::endl;
			return 1;
		}

		std::cout << std::endl;
	

	std::cout << "Success" << std::endl;

	return 0;
}
