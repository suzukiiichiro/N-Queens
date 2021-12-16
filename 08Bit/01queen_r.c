#include <stdlib.h>
#include <stdio.h>

#define DEPTH	 12
#define MASK		((1 << DEPTH) - 1)

int d[DEPTH];
int n;

void output() {
	int i, j;
	printf("pattern %d\n", ++n);
	for (i = 0; i < DEPTH; i++) {
		for (j = 0; j < DEPTH; j++) 
			putchar(d[i] & 1 << j ? 'Q' : '*');
		putchar('\n');
	}
}

void srch(int depth, int l, int c, int r) {
	int m, t;
	for (t = ~((l <<= 1) | c | (r >>= 1)) & MASK; t; t &= ~m) {
		d[depth] = m = -t & t;
		if (depth < DEPTH - 1) 
			srch(depth + 1, m | l, m | c, m | r);
		else output();
	}
}

int main() {
	srch(0, 0, 0, 0);
	printf("N=%d\n", n);
	return 0;
}
