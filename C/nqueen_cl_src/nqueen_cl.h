// N-queen solver for OpenCL
// Ping-Che Chen


#ifndef NQUEEN_CL_H
#define NQUEEN_CL_H


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>


class CLError
{
public:

	CLError(cl_int err, int line = 0) : m_ErrNo(err), m_Line(line) {}

	cl_int GetErrorNo() const { return m_ErrNo; }
	int GetErrorLine() const { return m_Line; }

private:

	cl_int m_ErrNo;
	int m_Line;
};


inline std::ostream& operator<<(std::ostream& stream, const CLError& x)
{
	stream << "OpenCL error: " << x.GetErrorNo();
	if(x.GetErrorLine() != 0) {
		stream << " (line: " << x.GetErrorLine() << ")";
	}

	return stream;
}


class NQueenSolver
{
public:

	NQueenSolver(cl_context context, cl_device_id device = 0, bool profiling = false, int threads = 0, bool force_local = false, bool force_no_atomics = false, bool force_no_vec = false);
	~NQueenSolver();

	long long Compute(int board_size, long long* unique);
	cl_ulong GetProfilingTime() { return m_TotalTime; }
	int GetThreads() { return m_nThreads; }
	bool AtomicsEnabled() { return m_bEnableAtomics; }
	bool VectorizationEnabled() { return m_bEnableVectorize; }

private:

	void InitKernels();
	void BuildProgram(const std::string& program, int vector_width, int work_items);

	cl_context m_Context;
	cl_device_id m_Device;
	cl_command_queue m_Queue;

	cl_program m_Program;
	cl_kernel m_NQueen;
	cl_kernel m_NQueen1;

	bool m_bProfiling;
	bool m_bCPU;
	bool m_bForceLocal;
	bool m_bForceNoAtomics;
	bool m_bForceNoVectorization;
	bool m_bEnableAtomics;
	bool m_bEnableVectorize;
	size_t m_nMaxWorkItems;
	cl_ulong m_TotalTime;
	int m_nThreads;
};


#endif
