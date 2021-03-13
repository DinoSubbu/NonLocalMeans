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
#include <iomanip>
#include <sstream>
#include <boost/lexical_cast.hpp>

using namespace std;

// TODO: Remove constants later
int imgWidth = 640;
int imgHeight = 480;

//////////////////////////////////////////////////////////////////////////////
// CPU implementation of NonLocal Means Algorithm
//////////////////////////////////////////////////////////////////////////////
void nlmHost(std::vector<float>& h_input,
			std::vector<float>& h_outputCpu,
			const float h,
			const int patchW,
			const int width,
			const int height)
{

	cout<<"Starting to run nlm algorithm on cpu"<<endl;

	float h_temp[height][width] = {0}, C[height][width] = {0};

	/*for(int i=0; i<height - patchW + 1; i++)
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
	                	 v += (h_input[(i+p-k)*width+(j+q-l)] - h_input[p*width+ (q)]) * (h_input[(i+p-k)*width+(j+q-l)] - h_input[p*width+(q)]);
	                 }
	              }

                  // Weight matrix calculation => exp(-v/h^²)
	              float w = exp(-v/(h*h));

	              // Multiply pixels with weight matrix and add them up!
				  h_temp[i][j] += w * h_input[k*width+l];
				  C[i][j] += w;
				  h_temp[k][l] += w * h_input[i* width+j];
				  C[k][l] += w;
	           }
	         }
	     }
	   }
	}*/

	for(int i=0; i<height - patchW + 1; i++)
		    {
		      for(int j=0; j<width - patchW + 1; j++)
		      {
		    	  h_outputCpu[i * width + j] = (h_temp[i][j])/(C[i][j]);
		      }
		    }

	std::cout<<"Finishing host calculation"<<std::endl;
}


void printPerformanceHeader() {
	std::cout << "Implementation           CPU       Calc       MT      GPU+MT  Speedup (w/o MT)" << std::endl;
}

void printPerformance(const std::string& name, Core::TimeSpan timeCalc, Core::TimeSpan timeMem, Core::TimeSpan timeCpu, bool showMem = true) {
	Core::TimeSpan overallTime = timeCalc + timeMem;
	std::stringstream str;
	str << std::setiosflags (std::ios::left) << std::setw (20) << name;
	str << std::setiosflags (std::ios::right);
	str << " " << std::setw (9) << timeCpu;
	str << " " << std::setw (9) << timeCalc;
	if (showMem)
		str << " " << std::setw (9) << timeMem;
	else
		str << " " << std::setw (9) << "";
	str << " " << std::setw (9) << overallTime;
	str << "  " << (timeCpu.getSeconds() / overallTime.getSeconds());
	if (showMem)
		str << " (" << (timeCpu.getSeconds() / timeCalc.getSeconds()) << ")";
	std::cout << str.str () << std::endl;
}

void printPerformance(const std::string& name, Core::TimeSpan timeCalc, Core::TimeSpan timeCpu) {
	printPerformance(name, timeCalc, Core::TimeSpan::fromSeconds(0), timeCpu, false);
}


//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

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
	std::cout<<"Working"<<std::endl;


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


	// Use an image (Valve.pgm) as input data
	std::size_t inputWidth = 640, inputHeight = 480;
	std::vector<float> inputData;
	
	// Allocate space for output data from CPU and GPU on the host
	/*float h_input[480][640] = {0}, h_outputGpu[480][640] = {0};
	float h_outputCpu[480][640] = {0};*/
	std::vector<float> h_input (count);
	std::vector<float> h_outputCpu (count);
	std::vector<float> h_outputGpu (count);
	/*h_outputCpu = new float*[480];
	for(int i = 0; i <480; i++)
		h_outputCpu[i] = new float[640];*/

	// Allocate space for input and output data on the device
	cl::Buffer d_Img(context, CL_MEM_READ_WRITE, inputWidth * inputHeight * sizeof(float) );
    cl::Buffer d_ImgTemp(context, CL_MEM_READ_WRITE,inputWidth * inputHeight * sizeof(float) );
    cl::Buffer d_ImgTemp1(context, CL_MEM_READ_WRITE,inputWidth * inputHeight * sizeof(float) );
    cl::Buffer d_C(context, CL_MEM_READ_WRITE, inputWidth * inputHeight * sizeof(float) );

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
    memset(h_input.data(), 255, size);
    memset(h_outputGpu.data(), 255, size);
    memset(h_outputCpu.data(), 255, size);
	
	//TODO: GPU
    //queue.enqueueWriteBuffer(d_Img,true,0,inputWidth * inputHeight * sizeof(float), h_input.data());
    queue.enqueueWriteBuffer(d_ImgTemp,true,0,inputWidth * inputHeight * sizeof(float),h_outputGpu.data());
    queue.enqueueWriteBuffer(d_C,true,0,inputWidth * inputHeight * sizeof(float),h_outputGpu.data());

	//////// Load input data ////////////////////////////////

	Core::readImagePGM("noisyImage.pgm", inputData, inputWidth, inputHeight);
	std::cout<<"inputWidth:: "<<inputWidth;
	std::cout<<"inputHeight:: "<<inputHeight;
	for (size_t j = 0; j < inputHeight; j++) {
		for (size_t i = 0; i < inputWidth; i++) {
			h_input[i + j* inputWidth] = inputData[(i % inputWidth) + (j % inputHeight)*inputWidth];
		}
	}
	Core::writeImagePGM("input_nlm_cpu_test.pgm", h_input, inputWidth, inputHeight);
	// Copy input data to device
	queue.enqueueWriteBuffer(d_Img,true,0,inputWidth * inputHeight * sizeof(float),h_input.data());

	// Do calculation on the host side
	float h = 1;
	int patchW = 3;
	nlmHost(h_input, h_outputCpu, h, patchW, inputWidth, inputHeight);

	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("output_nlm_cpu.pgm", h_outputCpu, inputWidth, inputHeight);
	std::cout<<"Image written"<<std::endl;

	// Reinitialize output memory to 0xff
	memset(h_outputGpu.data(), 1.2, inputWidth * inputHeight * sizeof(float));

	//GPU
	queue.enqueueWriteBuffer(d_C,true,0,inputWidth * inputHeight * sizeof(float),h_outputGpu.data());

	// Copy input data to device
	queue.enqueueWriteBuffer(d_ImgTemp, true, 0, inputWidth * inputHeight * sizeof(float),h_outputGpu.data());

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
								//cl::NDRange(200-2, 200-2, 200-3),
								cl::NDRange(inputHeight-2, inputWidth-2, inputHeight-2),
								cl::NullRange
                        		);
 	std::vector<float> h_imgTemp (count);
 	std::vector<float> h_C (count);

 	queue.enqueueReadBuffer(d_ImgTemp,true,0,inputWidth * inputHeight * sizeof(float),h_imgTemp.data(),NULL,NULL);
 	queue.enqueueReadBuffer(d_C,true,0,inputWidth * inputHeight * sizeof(float),h_C.data(),NULL,NULL);


    for(std::size_t localI=0; localI<inputHeight - patchW + 1; localI++)
	{
    	std::cout<<h_imgTemp[localI + 1*inputWidth]<<std::endl;
		  for(std::size_t localJ=0; localJ<inputWidth - patchW + 1; localJ++)
		  {
			  h_outputGpu[localI*inputWidth + localJ] = (h_imgTemp[localI*inputWidth + localJ])/(h_C[localI*inputWidth + localJ]);
		  }
	 }

	// Print performance data
    //TODO Uncomment later
    /*Core::TimeSpan gpuTime = OpenCL::getElapsedTime(writeBufferATime);
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(writeBufferBTime);
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(kernelTime);
	Core::TimeSpan copyTime = OpenCL::getElapsedTime(readBufferTime);
	Core::TimeSpan gpuTime = writeBufferATime + writeBufferBTime + kernelTime + readBufferTime;
	printPerformance(matrixMulKernel, gpuTime, copyTime, atlasTime);

	std::cout << "GPU Time: " << gpuTime << std::endl;
	std::cout << "CPU Time: " << cpuTime << std::endl;
	std::cout << "Speedup: " << (double)cpuTime.getseconds()/ gpuTime.getseconds() << std::endl;
	std::cout << "CPU Time: " << atlasTime << std::endl;*/

	//////// Store GPU output image ///////////////////////////////////
	std::cout<<"Creating Output Image GPU"<<std::endl;
	Core::writeImagePGM("output_nlm_gpu.pgm", h_outputGpu, inputWidth, inputHeight);

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
