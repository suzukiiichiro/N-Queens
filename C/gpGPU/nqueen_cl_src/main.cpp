// N-queen solver for OpenCL
// Ping-Che Chen


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "nqueen_cpu.h"
#include "nqueen_cl.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <ctime>

int main(int argc, char** argv)
{
	std::cerr << "N-Queen solver for OpenCL\n";
	std::cerr << "Ping-Che Chen\n\n";
	if(argc < 2) {
		std::cerr << "Usage: " << argv[0] << " -cpu -clcpu -p -threads # -platform # -local N\n";
		std::cerr << "\tN: board size (1 ~ 32)\n";
		std::cerr << "\t-cpu: use CPU (single threaded)\n";
		std::cerr << "\t-clcpu: use OpenCL CPU device\n";
		std::cerr << "\t-p: enable profiler\n";
		std::cerr << "\t-threads #: set number of threads to #\n";
		std::cerr << "\t-platform #: select platform #\n";
		std::cerr << "\t-local: using local memory for arrays, good for NVIDIA's GPU\n";
		std::cerr << "\t-noatomics: do not use global atomics\n";
		std::cerr << "\t-novec: do not use vectorization\n";
		return 0;
	}

	// handle options
	bool force_cpu = false;
	bool use_clcpu = false;
	cl_uint platform_index = 0;
	bool profiling = false;
	int threads = 0;
	bool local = false;
	bool noatomics = false;
	bool novec = false;

	int start = 1;
	while(start < argc - 1) {
		if(std::strcmp(argv[start], "-cpu") == 0) {
			force_cpu = true;
		}
		else if(std::strcmp(argv[start], "-clcpu") == 0) {
			use_clcpu = true;
		}
		else if(std::strcmp(argv[start], "-p") == 0) {
			profiling = true;
		}
		else if(std::strcmp(argv[start], "-threads") == 0 && start < argc - 2) {
			//threads = std::atoi(argv[start + 1]);
			threads = ::atoi(argv[start + 1]);
			start++;
		}
		else if(std::strcmp(argv[start], "-platform") == 0 && start < argc - 2) {
			//platform_index = std::atoi(argv[start + 1]);
			platform_index = ::atoi(argv[start + 1]);
			start++;
		}
		else if(std::strcmp(argv[start], "-local") == 0) {
			local = true;
		}
		else if(std::strcmp(argv[start], "-noatomics") == 0) {
			noatomics = true;
		}
		else if(std::strcmp(argv[start], "-novec") == 0) {
			novec = true;
		}
		else {
			std::cerr << "Unknown option " << argv[start] << "\n";
		}

		start ++;
	}

	int board_size = ::atoi(argv[start]);
	if(board_size < 1 || board_size > 32) {
		std::cerr << "Inalivd board size (only 1 ~ 32 allowed)\n";
		return 0;
	}

	clock_t start_time, end_time;
	long long solutions = 0;
	long long unique_solutions = 0;
	if(force_cpu) {
		start_time = std::clock();
		solutions = nqueen_cpu(board_size, &unique_solutions);
		end_time = std::clock();
	}
	else {
		cl_int err;
		cl_uint num;
		err = clGetPlatformIDs(0, 0, &num);
		if(err != CL_SUCCESS) {
			std::cerr << "Unable to get platforms\n";
			return 0;
		}

		std::vector<cl_platform_id> platforms(num);
		err = clGetPlatformIDs(num, &platforms[0], &num);
		if(err != CL_SUCCESS) {
			std::cerr << "Unable to get platforms\n";
			return 0;
		}

		// display all platform data
		for(cl_uint i = 0; i < num; i++) {
			size_t size;
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, 0, &size);
			std::string name;
			name.reserve(size + 1);
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, size, &name[0], 0);
			std::cerr << "Platform [" << i << "]: " << name.c_str() << "\n";
		}

		if(platform_index >= num) {
			platform_index = num - 1;
		}

		std::cerr << "Select platform " << platform_index << "\n";

		cl_context context;
		cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[platform_index]), 0 };
		context = clCreateContextFromType(prop, use_clcpu ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, 0, 0, &err);
		if(err != CL_SUCCESS) {
			std::cerr << "Unable to create context\n";
			return 0;
		}

		if(use_clcpu) {
			std::cerr << "Using CPU device\n";
		}
		else {
			std::cerr << "Using GPU device\n";
		}

		try {
			NQueenSolver nqueen(context, 0, profiling, threads, local, noatomics, novec);

			std::cerr << "Using " << nqueen.GetThreads() << " threads\n";
			if(nqueen.AtomicsEnabled()) {
				std::cerr << "Using global atomics\n";
			}

			if(nqueen.VectorizationEnabled()) {
				std::cerr << "Using vectorization\n";
			}

			start_time = std::clock();
			solutions = nqueen.Compute(board_size, &unique_solutions);
			end_time = std::clock();

			if(profiling) {
				std::cerr << "Profile time: " << nqueen.GetProfilingTime() << "ns\n";
			}
		}
		catch(CLError x)
		{
			std::cerr << x << "\n";
		}

		clReleaseContext(context);
	}

	std::cerr << board_size << "-queen has " << solutions << " solutions (" << unique_solutions << " unique)\n";
	std::cerr << "Time used: " << std::setprecision(3) << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << "s\n";
	

	return 0;
}
