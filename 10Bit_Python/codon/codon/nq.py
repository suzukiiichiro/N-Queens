# from codon import par, atomic

total = 0
N = 8

@par
for i in range(N):
    # atomic:
        # global total
        total += 1

print(f"N={N}, total={total}")

