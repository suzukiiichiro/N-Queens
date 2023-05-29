#include <stdio.h>

// http://kwhr0.g2.xrea.com/queen/index.html
#define DEPTH		8
#define MASK		((1 << DEPTH) - 1)

typedef struct {
	int left, center, right, bitmap, bit;
} rec;

int n;

void output(rec *board) {
	int i, j;
	printf("pattern %d\n", ++n);
	for (i = 0; i < DEPTH; i++) {
		for (j = 0; j < DEPTH; j++) 
			putchar(board[i].bit & 1 << j ? 'Q' : '*');
		putchar('\n');
	}
}

void srch() {
	rec board[DEPTH];
	rec *p = board;
	p->left = p->center = p->right = p->bit = 0;
	p->bitmap = MASK;
	while (1) {
		if (p->bitmap) {
			p->bit = -p->bitmap & p->bitmap;
			p->bitmap &= ~p->bit;
			if (p - board < DEPTH - 1) {
				rec *p0 = p++;
				p->left = (p0->left | p0->bit) << 1;
				p->center = p0->center | p0->bit;
				p->right = (p0->right | p0->bit) >> 1;
				p->bitmap = ~(p->left | p->center | p->right) & MASK;
			}
			else output(board);
		}
		else if (--p < board) return;
	}
}

int main() {
	srch();
	printf("N=%d\n", n);
	return 0;
}

