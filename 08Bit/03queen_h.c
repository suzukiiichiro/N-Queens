#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

#define DEPTH_MAX	32

long long nqueen(int depth) {
	struct {
		u_long l, c, r, t;
	} d[DEPTH_MAX], *p = d, *p0;
	long long n = 0;
	p->l = p->c = p->r = 0;
	p->t = (1 << ((depth + 1) >> 1)) - 1;
	while (1) {
		if (p->t) {
			u_long m = -p->t & p->t;
			p->t &= ~m;
			if (p - d < depth - 1) {
				p0 = p++;
				p->l = (p0->l | m) << 1;
				p->c = p0->c | m;
				p->r = (p0->r | m) >> 1;
				p->t = ~(p->l | p->c | p->r) & (1 << depth) - 1;
			}
			else n += 1 + (!(depth & 1) || d->t);
		}
		else if (--p < d) return n;
		printf("%ld %ld %ld %ld %ld\n", p - d, d[0].t, d[1].t, d[2].t, d[3].t);
	}
}

int main(int argc, char *argv[]) {
	int depth;
	if (argc != 2) {
		fprintf(stderr, "Usage: queen_h <n>\n");
		exit(1);
	}
	sscanf(argv[1], "%d", &depth);
	if (depth < 1 || depth > DEPTH_MAX) {
		fprintf(stderr, "out of range\n");
		exit(1);
	}
	printf("N=%lld\n", nqueen(depth));
	return 0;
}


