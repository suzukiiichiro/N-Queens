"""
CentOS-5.1$ codon run codonparallel.py
hello from thread hello from thread 2
hello from thread 2
hello from thread 1
hello from thread 1
hello from thread 1
2
hello from thread 4
hello from thread 4
hello from thread 3
hello from thread 3
CentOS-5.1$
"""
"""
@par
for i in range(10):
    import threading as thr
    print('hello from thread', thr.get_ident())
"""



"""
CentOS-5.1$ codon run codonparallel.py
hello from thread hello from thread 1
hello from thread 1
hello from thread 5
hello from thread 5
2
hello from thread 2
hello from thread 3
hello from thread 3
hello from thread 4
hello from thread 4
CentOS-5.1$
"""
"""
@par(num_threads=5)
for i in range(10):
    import threading as thr
    print('hello from thread', thr.get_ident())
"""

"""
CentOS-5.1$ codon run codonparallel.py
10
CentOS-5.1$
"""

"""
def foo(i:int)->int:
	return i

N=5
a = 0
@par
for i in range(N):
    a += foo(i)
print a
"""

"""
# 引数を渡してlimitを指定
CentOS-5.1$ codon run codonparallel.py 10
4
CentOS-5.1$
"""
"""
from sys import argv

def is_prime(n):
    factors = 0
    for i in range(2, n):
        if n % i == 0:
            factors += 1
    return factors == 0

limit = int(argv[1])
total = 0

@par(schedule='dynamic', chunk_size=100, num_threads=16)
for i in range(2, limit):
    if is_prime(i):
        total += 1

print(total)
"""

"""
CentOS-5.1$ codon run codonparallel.py
(x: 4950, y: 4950)
CentOS-5.1$
"""
"""
@tuple
class Vector:
    x: int
    y: int

    def __new__():
        return Vector(0, 0)

    def __add__(self, other: Vector):
        return Vector(self.x + other.x, self.y + other.y)

v = Vector()
@par
for i in range(100):
    v += Vector(i,i)
print(v)  # (x: 4950, y: 4950)
"""




"""
critical! 3
ordered! 95
critical! 3
ordered! 96
critical! 3
ordered! 97
critical! 3
ordered! 98
critical! 3
ordered! 99
CentOS-5.1$
"""

"""
import openmp as omp

@omp.critical
def only_run_by_one_thread_at_a_time():
    print('critical!', omp.get_thread_num())

@omp.master
def only_run_by_master_thread():
    print('master!', omp.get_thread_num())

@omp.single
def only_run_by_single_thread():
    print('single!', omp.get_thread_num())

@omp.ordered
def run_ordered_by_iteration(i):
    print('ordered!', i)

@par(ordered=True)
for i in range(100):
    only_run_by_one_thread_at_a_time()
    only_run_by_master_thread()
    only_run_by_single_thread()
    run_ordered_by_iteration(i)
"""

"""
:
:
only one thread at a time allowed here
only one thread at a time allowed here
only one thread at a time allowed here
only one thread at a time allowed here
only one thread at a time allowed here
only one thread at a time allowed here
only one thread at a time allowed here
CentOS-5.1$
"""
from threading import Lock
lock = Lock()  # or RLock for reentrant lock

@par
for i in range(100):
    with lock:
        print('only one thread at a time allowed here')
