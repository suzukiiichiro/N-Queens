// N-queen solver for OpenCL
// Ping-Che Chen

#include "nqueen_cpu.h"
#include <iostream>


int timing = 0;

/*
static long long nqueen_solver(int i, unsigned int board_mask, unsigned int mask, unsigned int left_mask, unsigned int right_mask)
{
	if(i == 0) {
		return 1;
	}

	unsigned int m = mask | left_mask | right_mask;
	unsigned int n = ((m + 1) & ~m);

	long long solutions = 0;
	while((n & board_mask) != 0) {
		m |= n;
		solutions += nqueen_solver(i - 1, board_mask, mask | n, (left_mask | n) << 1, (right_mask | n) >> 1);
		n = ((m + 1) & ~m);
	}

	return solutions;
}
*/


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


inline void transform(const unsigned int* ns_array, int* board_array, int size)
{
	for(int i = 0; i < size; i++) {
		board_array[i] = bit_scan(ns_array[i]);
	}
}


void display_table(const int * board_array, int size)
{
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			std::cout << ((board_array[i] == (size - j - 1)) ? "x " : "- ");
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}


void display_table2(const unsigned int * ns_array, int size)
{
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			std::cout << ((ns_array[i] == 1 << (size - j - 1)) ? "x " : "- ");
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

static long long nqueen_solver1(int size, int idx)
{
	unsigned int masks[32];
	unsigned int left_masks[32];
	unsigned int right_masks[32];
	unsigned int ms[32];
	unsigned int ns;
//	unsigned int ns_array[32];
	long long solutions = 0;
	int i = 0;

	masks[0] = 1 | 1 << idx;
	left_masks[0] = (1 << 2) | (1 << (idx + 1));
	right_masks[0] = (1 << idx) >> 1;
	ms[0] = masks[0] | left_masks[0] | right_masks[0];
	unsigned int board_mask = (1 << size) - 1;
//	ns_array[0] = 1;
//	ns_array[1] = (1 << idx);

	while(i >= 0) {
		unsigned int m = ms[i] | ((i + 2) < idx ? 2 : 0);
		ns = (m + 1) & ~m;
//		if((i + 2) < idx) {
//			unsigned int m = ms[i] | 2;
//			ns = (m + 1) & ~m;
//		}
//		else {
//			ns = ((ms[i] + 1) & ~ms[i]);
//		}

//		ns_array[i + 2] = ns;

		if((ns & board_mask) != 0) {
			if(i == size - 3) {
//				display_table2(ns_array, size);
				solutions++;
				i--;
			}
			else {
				ms[i] |= ns;
				masks[i+1] = masks[i] | ns;
				left_masks[i+1] = (left_masks[i] | ns) << 1;
				right_masks[i+1] = (right_masks[i] | ns) >> 1;
				ms[i+1] = masks[i+1] | left_masks[i+1] | right_masks[i + 1];
				i++;
			}
		}
		else {
			i--;
		}
	}

/*
	unsigned int ms;
	unsigned int ns[32];
	unsigned int left_mask_big;
	unsigned int right_mask_big;

	ms = mask | left_mask | right_mask;
	ns[0] = ((ms + 1) & ~ms);

	while(i >= 0) {
		if((ns[i] & board_mask) != 0) {
			mask |= ns[i];
			left_mask_big = (left_mask_big << 1) | (left_mask >> 31);
			left_mask = (left_mask | ns[i]) << 1;
			right_mask_big = (right_mask_big >> 1) | (right_mask << 31);
			right_mask = (right_mask | ns[i]) >> 1;
			ms = mask | left_mask | right_mask;
			ns[i + 1] = ((ms + 1) & ~ms);
			i++;
		}
		else {
			if(i == size - 1) {
				solutions++;
			}

			i--;
			if(i >= 0) {
				mask &= ~ns[i];
				left_mask = ((left_mask >> 1) | (left_mask_big << 31)) & ~ns[i];
				left_mask_big >>= 1;
				right_mask = ((right_mask << 1) | (right_mask_big >> 31)) & ~ns[i];
				right_mask_big <<= 1;
				ms = mask | left_mask | right_mask | ns[i];
				ns[i] = ((ms + ns[i]) & ~ms);
			}
		}
	}
*/
	return solutions;
}

static long long nqueen_solver(int size, unsigned int board_mask, unsigned int mask, unsigned int left_mask, unsigned int right_mask, long long* unique_solutions)
{
	unsigned int masks[32];
	unsigned int left_masks[32];
	unsigned int right_masks[32];
	unsigned int ms[32];
	unsigned int ns;
	unsigned int ns_array[32];
	unsigned int t_array[32];
	int board_array[32];
	long long solutions = 0;
	long long total_solutions = 0;
	int i = 0;
	unsigned int border_mask = 0;
	int index;

	unsigned int forbidden[32];

	masks[0] = mask;
	left_masks[0] = left_mask;
	right_masks[0] = right_mask;
	ms[0] = mask | left_mask | right_mask;
	ns_array[0] = mask;

//	transform(&mask, &index, 1);
	index = bit_scan(mask);
	//board_array[0] = index;
	for(int j = 0; j < index; j++) {
		border_mask |= (1 << j);
		border_mask |= (1 << (size - j - 1));
	}

	for(int i = 0; i < size; i++) {
		if(i == size - 2) {
			forbidden[i] = border_mask;
		}
		else if((i + 1) < index || (i + 1) > size - index - 1) {
			forbidden[i] = 1 | (1 << (size - 1));
		}
		else {
			forbidden[i] = 0;
		}
	}

	while(i >= 0) {
		unsigned int m = ms[i] | forbidden[i];
		ns = (m + 1) & ~m;
//		if(i == size - 2) {
//			unsigned int m = ms[i] | border_mask;
//			ns = (m + 1) & ~m;
//		}
//		else if((i + 1) < index || (i + 1) > size - index - 1) {
//			unsigned int m = ms[i] | 1 | (1 << (size - 1) );
//			ns = (m + 1) & ~m;
//		}
//		else {
//			ns = ((ms[i] + 1) & ~ms[i]);
//		}

		if((ns & board_mask) != 0) {
			ns_array[i+1] = ns;
			//board_array[i+1] = bit_scan(ns);
			if(i == size - 2) {
				int repeat_times = 8;
				bool rotate1 = false;
				bool rotate2 = false;
				bool rotate3 = false;

				if(ns_array[index] == (1 << (size - 1))) rotate1 = true;
				if(ns_array[size - index - 1] == 1) rotate2 = true;
				if(ns_array[size - 1] == (1 << (size - index - 1))) rotate3 = true;

				if(rotate1 || rotate2 || rotate3) {
					transform(ns_array, board_array, size);
					bool repeat = false;
					bool equal = true;

					if(rotate3) {
						// rotate 180
						equal = true;
						for(int j = 0; j < size; j++) {
							t_array[size - j - 1] = 1 << (size - board_array[j] - 1);
						}
						for(int j = 0; j < size; j++) {
							if(t_array[j] < ns_array[j]) {
								repeat = true;
								equal = false;
								break;
							}
							else if(t_array[j] > ns_array[j]) {
								equal = false;
								break;
							}
						}

						if(equal) {
							repeat_times = 4;
						}
					}
/*
				if(!repeat) {
					// mirror 180
					equal = true;
					for(int j = 0; j < size; j++) {
						t_array[size - j - 1] = 1 << board_array[j];
					}
					for(int j = 0; j < size; j++) {
						if(t_array[j] < ns_array[j]) {
							repeat = true;
							equal = false;
							break;
						}
						else if(t_array[j] > ns_array[j]) {
							equal = false;
							break;
						}
					}
				}

				if(equal) {
					repeat_times = 4;
				}
*/
				// rotate cw
					if(!repeat && rotate1) {
						equal = true;
						for(int j = 0; j < size; j++) {
							t_array[size - board_array[j] - 1] = 1 << j;
						}
						for(int j = 0; j < size; j++) {
							if(t_array[j] < ns_array[j]) {
								repeat = true;
								equal = false;
								break;
							}
							else if(t_array[j] > ns_array[j]) {
								equal = false;
								break;
							}
						}

						if(equal) {
							repeat_times = 2;
						}
					}
/*
				if(!repeat) {
					equal = true;
					// mirrow cw
					for(int j = 0; j < size; j++) {
						t_array[size - board_array[j] - 1] = 1 << (size - j - 1);
					}
					for(int j = 0; j < size; j++) {
						if(t_array[j] < ns_array[j]) {
							repeat = true;
							equal = false;
							break;
						}
						else if(t_array[j] > ns_array[j]) {
							equal = false;
							break;
						}
					}
				}

				if(equal) {
					repeat_times = 2;
				}
*/
					if(!repeat && rotate2) {
						// rotate ccw
						equal = true;
						for(int j = 0; j < size; j++) {
							t_array[board_array[j]] = 1 << (size - j - 1);
						}
						for(int j = 0; j < size; j++) {
							if(t_array[j] < ns_array[j]) {
								repeat = true;
								equal = false;
								break;
							}
							else if(t_array[j] > ns_array[j]) {
								equal = false;
								break;
							}
						}

						if(equal) {
							repeat_times = 2;
						}
					}
/*
				if(!repeat) {
					// mirror ccw
					equal = true;
					for(int j = 0; j < size; j++) {
						t_array[board_array[j]] = 1 << j;
					}
					for(int j = 0; j < size; j++) {
						if(t_array[j] < ns_array[j]) {
							repeat = true;
							equal = false;
							break;
						}
						else if(t_array[j] > ns_array[j]) {
							equal = false;
							break;
						}
					}
				}

				if(equal) {
					repeat_times = 2;
				}
*/
					if(!repeat) {
//						display_table(board_array, size);
						total_solutions += repeat_times;
						solutions++;
					}
				}
				else {
					total_solutions += 8;
					solutions++;
				}

				i--;
			}
			else {
				ms[i] |= ns;
				masks[i+1] = masks[i] | ns;
				left_masks[i+1] = (left_masks[i] | ns) << 1;
				right_masks[i+1] = (right_masks[i] | ns) >> 1;
				ms[i+1] = masks[i+1] | left_masks[i+1] | right_masks[i + 1];
				i++;
			}
		}
		else {
			i--;
		}
	}

/*
	unsigned int ms;
	unsigned int ns[32];
	unsigned int left_mask_big;
	unsigned int right_mask_big;

	ms = mask | left_mask | right_mask;
	ns[0] = ((ms + 1) & ~ms);

	while(i >= 0) {
		if((ns[i] & board_mask) != 0) {
			mask |= ns[i];
			left_mask_big = (left_mask_big << 1) | (left_mask >> 31);
			left_mask = (left_mask | ns[i]) << 1;
			right_mask_big = (right_mask_big >> 1) | (right_mask << 31);
			right_mask = (right_mask | ns[i]) >> 1;
			ms = mask | left_mask | right_mask;
			ns[i + 1] = ((ms + 1) & ~ms);
			i++;
		}
		else {
			if(i == size - 1) {
				solutions++;
			}

			i--;
			if(i >= 0) {
				mask &= ~ns[i];
				left_mask = ((left_mask >> 1) | (left_mask_big << 31)) & ~ns[i];
				left_mask_big >>= 1;
				right_mask = ((right_mask << 1) | (right_mask_big >> 31)) & ~ns[i];
				right_mask_big <<= 1;
				ms = mask | left_mask | right_mask | ns[i];
				ns[i] = ((ms + ns[i]) & ~ms);
			}
		}
	}
*/
	*unique_solutions = solutions;
	return total_solutions;
}


long long nqueen_cpu(int size, long long* unique_solutions)
{
	long long solutions = 0;
	long long u_solutions = 0;

	for(int i = 2; i < size; i++) {
		solutions += nqueen_solver1(size, i);
	}

	*unique_solutions = solutions;
	solutions *= 8;

//	std::cout << solutions << "\n";
//	std::cout << *unique_solutions << "\n";
		
	for(int i = 1; i < size / 2; i++) {
		solutions += nqueen_solver(size, (1 << size) - 1, 1 << i, 1 << (i + 1), (1 << i) >> 1, &u_solutions);
		*unique_solutions += u_solutions;

//		std::cout << solutions << "\n";
//		std::cout << *unique_solutions << "\n";
	}

	//solutions *= 2;

//	if(size % 2 == 1) {
//		int i = size / 2;
//		solutions += nqueen_solver(size, (1 << size) - 1, 1 << i, 1 << (i + 1), (1 << i) >> 1, &u_solutions);
//		*unique_solutions += u_solutions;
//	}

	timing = size;
	return solutions;
}
