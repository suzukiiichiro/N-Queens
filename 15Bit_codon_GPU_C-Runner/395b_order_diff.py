#!/usr/bin/env python3
# 395b_order_diff.py -- compare two SoA7 files record by record (positions, multiset, sortedness by (markctrl, ctrl0)).
import sys, array
def load(p):
    a=array.array('I'); a.frombytes(open(p,'rb').read())
    if sys.byteorder!='little': a.byteswap()
    return a
a=load(sys.argv[1]); b=load(sys.argv[2]); n=min(len(a),len(b))//7
diffpos=[i for i in range(n) if a[i*7:(i+1)*7]!=b[i*7:(i+1)*7]]
print(f"records={n} differing_positions={len(diffpos)} first={diffpos[:5]}")
sa=sorted(tuple(a[i*7:(i+1)*7]) for i in range(n)); sb=sorted(tuple(b[i*7:(i+1)*7]) for i in range(n))
print("same multiset:", sa==sb)
ka=[(a[i*7+5],a[i*7+3]) for i in range(n)]; kb=[(b[i*7+5],b[i*7+3]) for i in range(n)]
print("file1 sorted by (markctrl,ctrl0):", ka==sorted(ka), " file2 sorted:", kb==sorted(kb))
# adjacency retained relative to each other cannot be known without the raw order; report key-class runs instead
def runs(k):
    r=1
    for i in range(1,len(k)):
        if k[i]!=k[i-1]: r+=1
    return r
print("key-class runs: file1", runs(ka), " file2", runs(kb))
