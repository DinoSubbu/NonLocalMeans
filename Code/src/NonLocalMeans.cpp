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

//////////////////////////////////////////////////////////////////////////////
// CPU implementation of NonLocal Means Algorithm
//////////////////////////////////////////////////////////////////////////////
void nlmHost()
{
	// Have to figure out how to pass arguments
	// Add dimensions of the image

	cout<<"Starting to run nlm algorithm on cpu"<<endl;

	for(int i=0; i<height - patchW + 1; i++)
	{
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
bool m_preProcess = true;

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
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;
	//countX *= 3; countY *= 3;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof (float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);
	std::vector<float> h_outputGpu (count);
	
	auto accessInput = input->getOpenCLImageAccess(ACCESS_READ, device);
    auto accessOutput = output->getOpenCLImageAccess(ACCESS_READ_WRITE, device);
    auto accessAux = auxImage->getOpenCLImageAccess(ACCESS_READ_WRITE, device);

    auto bufferIn = accessInput->get2DImage();
    auto bufferOut = accessAux->get2DImage();
	// Allocate space for input and output data on the device
	//TODO
	cl::Image2D d_input(context, CL_MEM_READ_WRITE,cl::ImageFormat(CL_R,CL_FLOAT),countX,countY);
	cl::Image2D d_output(context, CL_MEM_READ_WRITE,cl::ImageFormat(CL_R,CL_FLOAT),countX,countY);
	//cl::Buffer d_input(context, CL_MEM_READ_WRITE,size);
	//cl::Buffer d_output(context, CL_MEM_READ_WRITE,size);
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	//TODO: GPU
	queue.enqueueWriteImage(bufferIn,true,origin,region,countX *(sizeof(float)),0,h_input.data(),NULL,NULL);
	queue.enqueueWriteImage(bufferOut,true,origin,region,countX *(sizeof(float)),0,h_outputGpu.data(),NULL,NULL);
	
/* 		queue.enqueueWriteImage(d_input,true,origin,region,countX *(sizeof(float)),0,h_input.data(),NULL,NULL);
	queue.enqueueWriteImage(d_output,true,origin,region,countX *(sizeof(float)),0,h_outputGpu.data(),NULL,NULL); */ //dinesh
	//queue.enqueueWriteBuffer(d_input, true, 0, size,h_input.data());
	//queue.enqueueWriteBuffer(d_output, true, 0, size,h_outputGpu.data());

	//////// Load input data ////////////////////////////////
	// Use random input data
	/*
	for (int i = 0; i < count; i++)
		h_input[i] = (rand() % 100) / 5.0f - 10.0f;
	*/
	// Use an image (Valve.pgm) as input data
	std::size_t inputWidth, inputHeight;
	std::vector<float> inputData;

	Core::readImagePGM("Valve.pgm", inputData, inputWidth, inputHeight);
	std::cout<<"Width:: "<<inputWidth;
	for (size_t j = 0; j < countY; j++) {
		for (size_t i = 0; i < countX; i++) {
			h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}

	// Copy input data to device
	//TODO
	queue.enqueueWriteImage(bufferIn,true,origin,region,countX *(sizeof(float)),0,h_input.data(),NULL,NULL);
//	queue.enqueueWriteImage(d_input,true,origin,region,countX *(sizeof(float)),0,h_input.data(),NULL,NULL);

	//queue.enqueueWriteBuffer(d_input, true, 0, size,h_input.data());

	// Do calculation on the host side
	//sobelHost(h_input, h_outputCpu, countX, countY);

	//////// Store CPU output image ///////////////////////////////////
	//Core::writeImagePGM("output_sobel_cpu.pgm", h_outputCpu, countX, countY);

	// Reinitialize output memory to 0xff
	memset(h_outputGpu.data(), 255, size);
	//TODO: GPU
//	queue.enqueueWriteImage(d_output,true,origin,region,countX *(sizeof(float)),0,h_outputGpu.data(),NULL,NULL); // dinesh

	queue.enqueueWriteImage(bufferOut,true,origin,region,countX *(sizeof(float)),0,h_outputGpu.data(),NULL,NULL); //priya
	//queue.enqueueWriteBuffer(d_output, true, 0, size,h_outputGpu.data());
    cl::Kernel kernelPreProcess(program, "preprocess");

	std::cout << std::endl;

	
	if(m_preProcess) 
	{
        kernelPreProcess.setArg(0, *bufferIn);
        kernelPreProcess.setArg(1, *bufferOut);
        queue.enqueueNDRangeKernel(
            kernelPreProcess,
            cl::NullRange,
            cl::NDRange(width, height),
            cl::NullRange
        );

        bufferIn = bufferOut;
        bufferOut = accessOutput->get2DImage();
		
    } else {
        queue.enqueueCopyImage(
            *bufferIn,
            *bufferOut,
            createOrigoRegion(),
            createOrigoRegion(),
            createRegion(width, height, 1)
        );
        bufferIn = bufferOut;
        bufferOut = accessOutput->get2DImage();
    }

	// Iterate over all implementations (task 1 - 3)
	for (int iteration = 0; iteration < 3; ++iteration) 
	{
		
		// Copy input data to device
		//TODO
		//queue.enqueueWriteBuffer(d_input, true, 0, size,h_input.data());

		// Create a kernel object
		cl::Kernel kernelNonLocal(program, "nonLocalMeans");

		// Launch kernel on the device
		//TODO
		const int m_searchSize = 11;
		const int m_filterSize = 3;
		const int m_parameterH = 0.15f;
		kernelNonLocal.setArg(0, *bufferIn);
        kernelNonLocal.setArg(1, *bufferOut);
//		kernelNonLocal.setArg<cl::Image2D>(0, d_input);
//		kernelNonLocal.setArg<cl::Image2D>(1, d_output);
		kernelNonLocal.setArg(2, m_searchSize);
		kernelNonLocal.setArg(3, (m_filterSize - 1)/2);
		kernelNonLocal.setArg(4, m_parameterH*(1.0f/(float)std::pow(2, iteration)));
		kernelNonLocal.setArg(5, iteration); // iteration

        queue.enqueueNDRangeKernel(
            kernelNonLocal,
            cl::NullRange,
            cl::NDRange(inputWidth, inputHeight),
            cl::NullRange
        );
        auto tmp = bufferIn;
        bufferIn = bufferOut;
        bufferOut = tmp;
/* 		auto tmp = d_input;
		d_input = d_output;
		d_output = tmp; */

	}
		// Copy output data back to host
		//TODO
		//queue.enqueueReadImage(d_output,true,origin,region,countX *(sizeof(float)),0,h_outputGpu.data(),NULL,NULL); //dinesh
		
		queue.enqueueReadImage(bufferOut,true,origin,region,countX *(sizeof(float)),0,h_outputGpu.data(),NULL,NULL); // priya

		//queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, NULL);
	
		// Print performance data
		//TODO

		//////// Store GPU output image ///////////////////////////////////
		std::cout<<"Creating Output Image"<<std::endl;
		Core::writeImagePGM("output_nonlocal_gpu.pgm", h_outputGpu, countX, countY);

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
