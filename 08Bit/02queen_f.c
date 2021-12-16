#include <stdio.h>

#define DEPTH		12
#define MASK		((1 << DEPTH) - 1)

typedef struct {
	int l, c, r, t, m;
} rec;

int n;

void output(rec *d) {
	int i, j;
	printf("pattern %d\n", ++n);
	for (i = 0; i < DEPTH; i++) {
		for (j = 0; j < DEPTH; j++) 
			putchar(d[i].m & 1 << j ? 'Q' : '*');
		putchar('\n');
	}
}

void srch() {
	rec d[DEPTH];
	rec *p = d;
	p->l = p->c = p->r = p->m = 0;
	p->t = MASK;
	while (1) {
		if (p->t) {
			p->m = -p->t & p->t;
			p->t &= ~p->m;
			if (p - d < DEPTH - 1) {
				rec *p0 = p++;
				p->l = (p0->l | p0->m) << 1;
				p->c = p0->c | p0->m;
				p->r = (p0->r | p0->m) >> 1;
				p->t = ~(p->l | p->c | p->r) & MASK;
			}
			else output(d);
		}
		else if (--p < d) return;
	}
}

int main() {
	srch();
	printf("N=%d\n", n);
	return 0;
}

