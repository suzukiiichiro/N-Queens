// N-queen solver for OpenCL
// Ping-Che Chen


#include <fstream>
#include <sstream>
#include "nqueen_cl.h"


#define CHECK_ERROR(err) { if(err != CL_SUCCESS) throw CLError(err, __LINE__); }

class CLMemAutoRelease
{
public:

	CLMemAutoRelease(cl_mem mem) : m_Mem(mem) {}
	~CLMemAutoRelease() { clReleaseMemObject(m_Mem); }

private:
	
	cl_mem m_Mem;
};


class CLEventAutoRelease
{
public:

	CLEventAutoRelease(cl_event ev) : m_Event(ev) {}
	~CLEventAutoRelease() { clReleaseEvent(m_Event); }

private:
	
	cl_event m_Event;
};


NQueenSolver::NQueenSolver(cl_context context, cl_device_id device, bool profiling, int threads, bool force_local, bool force_no_atomics, bool force_no_vec) :
	m_Context(context), m_Device(device), m_bProfiling(profiling), m_nThreads(threads), m_bForceLocal(force_local), m_bForceNoAtomics(force_no_atomics), m_bForceNoVectorization(force_no_vec),
	m_Queue(0), m_Program(0), m_NQueen(0), m_NQueen1(0), m_bEnableAtomics(false), m_bEnableVectorize(false)
{
	cl_int err;

	err = clRetainContext(m_Context);
	CHECK_ERROR(err);

	if(device == 0) {
		// query device
		size_t size;
		err = clGetContextInfo(m_Context, CL_CONTEXT_DEVICES, 0, 0, &size);
		CHECK_ERROR(err);

		std::vector<cl_device_id> devices(size / sizeof(cl_device_id));
		err = clGetContextInfo(m_Context, CL_CONTEXT_DEVICES, size, &devices[0], 0);
		CHECK_ERROR(err);

		m_Device = devices[0];
	}
	else {
		m_Device = device;
	}

	cl_device_type type;
	err = clGetDeviceInfo(m_Device, CL_DEVICE_TYPE, sizeof(type), &type, 0);
	CHECK_ERROR(err);

	m_bCPU = ((type & CL_DEVICE_TYPE_CPU) != 0);

	InitKernels();

	m_Queue = clCreateCommandQueue(m_Context, m_Device, m_bProfiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
	CHECK_ERROR(err);
}


NQueenSolver::~NQueenSolver()
{
	if(m_NQueen != 0) {
		clReleaseKernel(m_NQueen);
	}

	if(m_NQueen1 != 0) {
		clReleaseKernel(m_NQueen1);
	}

	if(m_Program != 0) {
		clReleaseProgram(m_Program);
	}

	if(m_Queue != 0) {
		clReleaseCommandQueue(m_Queue);
	}

	if(m_Context != 0) {
		clReleaseContext(m_Context);
	}
}


void NQueenSolver::InitKernels()
{
	cl_uint vector_width;
	cl_int err;
	err = clGetDeviceInfo(m_Device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(vector_width), &vector_width, 0);
	CHECK_ERROR(err);

	if(!m_bForceNoVectorization && !m_bCPU && vector_width != 1) {
		m_bEnableVectorize = true;
		m_bForceLocal = true;
	}

	cl_uint dimensions;
	err = clGetDeviceInfo(m_Device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dimensions), &dimensions, 0);
	CHECK_ERROR(err);

	std::vector<size_t> max_size(dimensions);
	err = clGetDeviceInfo(m_Device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dimensions, &max_size[0], 0);
	CHECK_ERROR(err);

	size_t length;
	err = clGetDeviceInfo(m_Device, CL_DEVICE_EXTENSIONS, 0, 0, &length);
	CHECK_ERROR(err);

	std::string extensions;
	extensions.resize(length + 1);
	err = clGetDeviceInfo(m_Device, CL_DEVICE_EXTENSIONS, length, &extensions[0], 0);
	CHECK_ERROR(err);

	std::stringstream ext(extensions);
	m_bEnableAtomics = false;
	while(!ext.eof()) {
		std::string name;
		ext >> name;
		if(name == "cl_khr_global_int32_base_atomics") {
			m_bEnableAtomics = true;
			break;
		}
	}

	if(m_bForceNoAtomics || m_bEnableVectorize || m_bCPU) {
		m_bEnableAtomics = false;
	}

	// load program
	std::ifstream in("kernels.cl");
	in.seekg(0, std::ios_base::end);
	std::ifstream::pos_type size = in.tellg();
	in.seekg(0, std::ios_base::beg);

	std::string buffer;
	buffer.resize(size);
	in.read(&buffer[0], size);

	BuildProgram(buffer, vector_width, m_bEnableVectorize ? 128 : 256);

	cl_uint units;
	err = clGetDeviceInfo(m_Device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, 0);
	CHECK_ERROR(err);

	if(m_nThreads == 0) {
		if(m_bEnableAtomics) {
			m_nThreads = max_size[0] * units;
		}
		else if(m_bEnableVectorize) {
			m_nThreads = max_size[0] * units * 16;
		}
		else {
			m_nThreads = max_size[0] * units * 2;
		}
	}

	err = clGetKernelWorkGroupInfo(m_NQueen, m_Device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &m_nMaxWorkItems, 0);
	CHECK_ERROR(err);

	if(m_nMaxWorkItems > 256) {
		m_nMaxWorkItems = 256;
	}

	if(m_bForceLocal && m_nMaxWorkItems < 256) {
		// rebuild program
		if(m_NQueen != 0) {
			clReleaseKernel(m_NQueen);
		}

		if(m_NQueen1 != 0) {
			clReleaseKernel(m_NQueen1);
		}

		if(m_Program != 0) {
			clReleaseProgram(m_Program);
		}

		BuildProgram(buffer, vector_width, m_bEnableVectorize ? m_nMaxWorkItems / 2 : m_nMaxWorkItems);

		if(m_nThreads % m_nMaxWorkItems != 0) {
			m_nThreads = (m_nThreads / m_nMaxWorkItems) * m_nMaxWorkItems;
		}
	}
}


void NQueenSolver::BuildProgram(const std::string& program, int vector_width, int work_items)
{
	const char* bufs[1] = { &program[0] };
	size_t lengths[1] = { program.size() };
	cl_int err;

	m_Program = clCreateProgramWithSource(m_Context, 1, bufs, lengths, &err);
	CHECK_ERROR(err);

	std::stringstream settings;
	if(m_bCPU) {
		settings << "-D FORCE_CPU";
	}
	else if(m_bForceLocal) {
		settings << "-D FORCE_LOCAL -D WORK_ITEMS=" << work_items;
	}

	if(m_bEnableAtomics && !m_bCPU) {
		settings << " -D USE_ATOMICS";
	}

	if(m_bEnableVectorize) {
		settings << " -D ENABLE_VECTORIZE";
	}

	err = clBuildProgram(m_Program, 1, &m_Device, settings.str().c_str(), 0, 0);
	if(err != CL_SUCCESS) {
		size_t param_size;
		clGetProgramBuildInfo(m_Program, m_Device, CL_PROGRAM_BUILD_LOG, 0, 0, &param_size);
		std::string log;
		log.resize(param_size);
		clGetProgramBuildInfo(m_Program, m_Device, CL_PROGRAM_BUILD_LOG, param_size, &log[0], 0);
		std::cerr << log.c_str() << "\n";
		
		CHECK_ERROR(err);
	}
/*
	cl_uint devices;
	err = clGetProgramInfo(m_Program, CL_PROGRAM_NUM_DEVICES, sizeof(devices), &devices, 0);
	CHECK_ERROR(err);

	std::vector<size_t> binary_size(devices);
	err = clGetProgramInfo(m_Program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * devices, &binary_size[0], 0);
	CHECK_ERROR(err);

	std::vector<unsigned char*> binary_pointer(devices);
	std::vector<std::vector<unsigned char> > binary(devices);
	for(size_t i = 0; i < devices; i++) {
		binary[i].resize(binary_size[i]);
		binary_pointer[i] = &binary[i][0];
	}
	err = clGetProgramInfo(m_Program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * devices, &binary_pointer[0], 0);
	CHECK_ERROR(err);

	std::ofstream out("kernels.bin", std::ios_base::binary);
	out.write(reinterpret_cast<const char*>(&binary[0][0]), binary_size[0]);
*/
	if(m_bEnableVectorize) {
		m_NQueen = clCreateKernel(m_Program, "nqueen_vec", &err);
		CHECK_ERROR(err);

		m_NQueen1 = clCreateKernel(m_Program, "nqueen1_vec", &err);
		CHECK_ERROR(err);
	}
	else {
		m_NQueen = clCreateKernel(m_Program, "nqueen", &err);
		CHECK_ERROR(err);

		m_NQueen1 = clCreateKernel(m_Program, "nqueen1", &err);
		CHECK_ERROR(err);
	}
}


inline int bit_scan(unsigned int x)
{
	int res = 0;
	res |= (x & 0xaaaaaaaa) ? 1 : 0;
	res |= (x & 0xcccccccc) ? 2 : 0;
	res |= (x & 0xf0f0f0f0) ? 4 : 0;
	res |= (x & 0xff00ff00) ? 8 : 0;
	res |= (x & 0xffff0000) ? 16 : 0;
	return res;
}


long long NQueenSolver::Compute(int board_size, long long* unique)
{
	// estimate amount of levels need to be computed on the device
	long long total = 10000000000LL;
	total /= 10;	// for atomics
	int level = 0;
	int i = board_size;
	while(total > 0 && i > 0) {
		total /= ((i + 1) / 2);
		i--;
		level++;
	}

	if(level > board_size - 2) {
		level = board_size - 2;
	}

	if(level > 11) {
		level = 11;
	}

	if(m_bCPU) {
		level = board_size - 2;
	}

	int threads;
	if(m_bEnableAtomics) {
		threads = m_nThreads * 16;
	}
	else {
		threads = m_nThreads;
	}
	std::vector<unsigned int> mask_vector(threads * (4 + 32));
	std::vector<unsigned int> results(threads * 2);
	cl_int err;

	// create data buffer
	cl_mem param_buffer = clCreateBuffer(m_Context, CL_MEM_READ_ONLY, threads * sizeof(int) * (4 + 32), 0, &err);
	CLMemAutoRelease c_param(param_buffer);
	CHECK_ERROR(err);

	cl_mem result_buffer = clCreateBuffer(m_Context, CL_MEM_WRITE_ONLY, threads * sizeof(int) * 2, 0, &err);
	CLMemAutoRelease c_result(result_buffer);
	CHECK_ERROR(err);

	cl_mem forbidden_buffer = clCreateBuffer(m_Context, CL_MEM_READ_ONLY, 32 * sizeof(int), 0, &err);
	CLMemAutoRelease c_forbidden(forbidden_buffer);
	CHECK_ERROR(err);

	cl_mem global_index = clCreateBuffer(m_Context, CL_MEM_READ_WRITE, sizeof(int), 0, &err);
	CLMemAutoRelease c_global_index(global_index);
	CHECK_ERROR(err);

	long long solutions = 0;
	long long unique_solutions = 0;
	m_TotalTime = 0;
	bool has_data = false;

	cl_event profile_event = 0;
	CLEventAutoRelease c_profile_event(profile_event);

	unsigned int board_mask = (1 << board_size) - 1;
	int total_size = 0;
	int last_total_size = 0;
	for(int j = 0; j < board_size / 2; j++) {
	// only do nqueen1
//	int j = 3;
//	{
		unsigned int masks[32];
		unsigned int left_masks[32];
		unsigned int right_masks[32];
		unsigned int ms[32];
		unsigned int ns[32];
		unsigned int forbidden[32];
		unsigned int border_mask = 0;
		int idx = 0;
		int i = 0;

		masks[0] = (1 << j);
		left_masks[0] = 1 << (j + 1);
		right_masks[0] = (1 << j) >> 1;
		ms[0] = masks[0] | left_masks[0] | right_masks[0];
		ns[0] = (1 << j);

		cl_kernel queen = (j == 0 ? m_NQueen1 : m_NQueen);

		for(int k = 0; k < j; k++) {
			border_mask |= (1 << k);
			border_mask |= (1 << (board_size - k - 1));
		}

		for(int k = 0; k < board_size; k++) {
			if(k == board_size - 2) {
				forbidden[k] = border_mask;
			}
			else if((k + 1) < j || (k + 1) > board_size - j - 1) {
				forbidden[k] = 1 | (1 << (board_size - 1));
			}
			else {
				forbidden[k] = 0;
			}
		}
		forbidden[board_size - 1] = 0xffffffff;

		err = clEnqueueWriteBuffer(m_Queue, forbidden_buffer, CL_TRUE, 0, (level + 1) * sizeof(int), forbidden + board_size - level - 1, 0, 0, 0);
		CHECK_ERROR(err);

		while(i >= 0) {
			if(j == 0) {
				if(i >= 1) {
					unsigned int m = ms[i] | (i + 1 < idx ? 2 : 0);
					ns[i + 1] = (m + 1) & ~m;
				}
				else {
					ns[i + 1] = ((ms[i] + 1) & ~ms[i]);
					if(i == 0) {
						idx = bit_scan(ns[i + 1]);
					}
				}
			}
			else {
				unsigned int m = ms[i] | forbidden[i];
				ns[i + 1] = (m + 1) & ~m;
			}

			if(i == board_size - level - 1) {
				mask_vector[total_size] = masks[i];
				mask_vector[total_size + threads] = left_masks[i];
				mask_vector[total_size + threads * 2] = right_masks[i];
				if(j == 0) {
					mask_vector[total_size + threads * 3] = idx - i < 0 ? 0 : idx - i;
				}
				else {
					// check rotation
					mask_vector[total_size + threads * 3] = j;
				}
				for(int k = 0; k <= i; k++) {
					mask_vector[total_size + threads * (k + 4)] = ns[k];
				}
				total_size++;
				if(total_size == threads) {
					if(has_data) {
						err = clEnqueueReadBuffer(m_Queue, result_buffer, CL_TRUE, 0, threads * sizeof(int) * 2, &results[0], 0, 0, 0);
						CHECK_ERROR(err);

						if(m_bProfiling) {
							cl_ulong start, end;
							clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, 0);
							clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, 0);

							m_TotalTime += end - start;

							clReleaseEvent(profile_event);
							profile_event = 0;
						}

						for(int k = 0; k < last_total_size; k++) {
							solutions += results[k];
							unique_solutions += results[k + threads];
						}
					}
					
					cl_int arg_board_size = board_size;
					cl_int arg_level = level;
					cl_int arg_threads = m_bEnableVectorize ? threads / 2 : threads;
					err = clSetKernelArg(queen, 0, sizeof(cl_int), &arg_board_size);
					err |= clSetKernelArg(queen, 1, sizeof(cl_int), &arg_level);
					err |= clSetKernelArg(queen, 2, sizeof(cl_int), &arg_threads);
					err |= clSetKernelArg(queen, 3, sizeof(cl_mem), &param_buffer);
					err |= clSetKernelArg(queen, 4, sizeof(cl_mem), &result_buffer);
					err |= clSetKernelArg(queen, 5, sizeof(cl_mem), &forbidden_buffer);
					if(m_bEnableAtomics) {
						err |= clSetKernelArg(queen, 6, sizeof(cl_mem), &global_index);
					}
					CHECK_ERROR(err);

					err = clEnqueueWriteBuffer(m_Queue, param_buffer, CL_TRUE, 0, threads * sizeof(int) * (4 + 32), &mask_vector[0], 0, 0, 0);
					CHECK_ERROR(err);

					err = clEnqueueWriteBuffer(m_Queue, global_index, CL_TRUE, 0, sizeof(int), &m_nThreads, 0, 0, 0);
					CHECK_ERROR(err);

					size_t work_dim[1] = { m_bEnableVectorize ? m_nThreads / 2 : m_nThreads };
					size_t* group_dim = 0;
					size_t n = m_nMaxWorkItems / 2;
					if(m_bForceLocal) {
						group_dim = m_bEnableVectorize ? &n : &m_nMaxWorkItems;
					}
					err = clEnqueueNDRangeKernel(m_Queue, queen, 1, 0, work_dim, group_dim, 0, 0, m_bProfiling ? &profile_event : 0);
					CHECK_ERROR(err);

					has_data = true;
					last_total_size = total_size;
					total_size = 0;
				}

				i--;
			}
			else if((ns[i + 1] & board_mask) != 0) {
				ms[i] |= ns[i + 1];
				masks[i+1] = masks[i] | ns[i + 1];
				left_masks[i+1] = (left_masks[i] | ns[i + 1]) << 1;
				right_masks[i+1] = (right_masks[i] | ns[i + 1]) >> 1;
				ms[i+1] = masks[i+1] | left_masks[i+1] | right_masks[i + 1];
				i++;
			}
			else {
				i--;
			}
		}

		if(has_data) {
			err = clEnqueueReadBuffer(m_Queue, result_buffer, CL_TRUE, 0, threads * sizeof(int) * 2, &results[0], 0, 0, 0);
			CHECK_ERROR(err);

			if(m_bProfiling) {
				cl_ulong start, end;
				clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, 0);
				clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, 0);

				m_TotalTime += end - start;

				clReleaseEvent(profile_event);
				profile_event = 0;
			}

			for(int k = 0; k < last_total_size; k++) {
				solutions += results[k];
				unique_solutions += results[k + threads];
			}

//			std::cout << solutions << "\n";
//			std::cout << unique_solutions << "\n";

			has_data = false;
		}

		if(total_size > 0) {
			cl_int arg_board_size = board_size;
			cl_int arg_level = level;
			cl_int arg_threads = m_bEnableVectorize ? threads / 2 : threads;
			err = clSetKernelArg(queen, 0, sizeof(cl_int), &arg_board_size);
			err |= clSetKernelArg(queen, 1, sizeof(cl_int), &arg_level);
			err |= clSetKernelArg(queen, 2, sizeof(cl_int), &arg_threads);
			err |= clSetKernelArg(queen, 3, sizeof(cl_mem), &param_buffer);
			err |= clSetKernelArg(queen, 4, sizeof(cl_mem), &result_buffer);
			err |= clSetKernelArg(queen, 5, sizeof(cl_mem), &forbidden_buffer);
			if(m_bEnableAtomics) {
				err |= clSetKernelArg(queen, 6, sizeof(cl_mem), &global_index);
			}
			CHECK_ERROR(err);

			for(int k = total_size; k < threads; k++) {
				mask_vector[k] = 0xffffffff;
				mask_vector[k + threads] = 0xffffffff;
				mask_vector[k + threads * 2] = 0xffffffff;
				mask_vector[k + threads * 3] = 0;
				mask_vector[k + threads * 4] = 0;
			}

			err = clEnqueueWriteBuffer(m_Queue, param_buffer, CL_TRUE, 0, threads * sizeof(int) * (4 + 32), &mask_vector[0], 0, 0, 0);
			CHECK_ERROR(err);

			size_t work_dim[1];
			if(total_size < m_nThreads) {
				work_dim[0] = m_bEnableVectorize ? (total_size + 1) / 2 : total_size;
			}
			else {
				work_dim[0] = m_bEnableVectorize ? m_nThreads / 2 : m_nThreads;
			}

			size_t* group_dim = 0;
			size_t n = m_bEnableVectorize ? m_nMaxWorkItems / 2 : m_nMaxWorkItems;
			if(m_bForceLocal) {
				group_dim = &n;
				if(work_dim[0] % n != 0) {
					work_dim[0] += n - work_dim[0] % n;
				}
			}

			int num_thread = work_dim[0];
			err = clEnqueueWriteBuffer(m_Queue, global_index, CL_TRUE, 0, sizeof(int), &num_thread, 0, 0, 0);
			CHECK_ERROR(err);

			err = clEnqueueNDRangeKernel(m_Queue, queen, 1, 0, work_dim, group_dim, 0, 0, m_bProfiling ? &profile_event : 0);
			CHECK_ERROR(err);

			has_data = true;
			last_total_size = total_size;
			total_size = 0;
		}
	}

	if(has_data) {
		err = clEnqueueReadBuffer(m_Queue, result_buffer, CL_TRUE, 0, threads * sizeof(int) * 2, &results[0], 0, 0, 0);
		CHECK_ERROR(err);

		if(m_bProfiling) {
			cl_ulong start, end;
			clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, 0);
			clGetEventProfilingInfo(profile_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, 0);

			m_TotalTime += end - start;

			clReleaseEvent(profile_event);
			profile_event = 0;
		}

		for(int k = 0; k < last_total_size; k++) {
			solutions += results[k];
			unique_solutions += results[k + threads];
		}

//		std::cout << solutions << "\n";
//		std::cout << unique_solutions << "\n";
	}

	if(unique != 0) {
		*unique = unique_solutions;
	}

	return solutions;
}
