#!/usr/bin/env python3
# 395b_sass_summary.py -- stall shares, LDL/STL counts and top long_scoreboard instructions from an ncu --page source --csv export
import csv, sys
rows=list(csv.reader(open(sys.argv[1]))); hdr=rows[1]; data=[r for r in rows[2:] if len(r)==len(hdr)]; idx={h:i for i,h in enumerate(hdr)}
def num(r,k):
    v=r[idx[k]].replace(',','')
    try: return int(v)
    except: return 0
tot=sum(num(r,'# Samples') for r in data) or 1
print("SASS instructions:", len(data))
for k in ['stall_wait','stall_branch_resolving','stall_selected','stall_long_sb','stall_no_inst']:
    print(f"{k:24s} {100*sum(num(r,k) for r in data)/tot:6.2f}%")
print("LDL count:", sum('LDL' in r[idx['Source']] for r in data), " STL count:", sum('STL' in r[idx['Source']] for r in data), " SHF.R.U64 count:", sum('SHF.R.U64' in r[idx['Source']] for r in data))
for r in sorted(data,key=lambda r:-num(r,'stall_long_sb'))[:3]:
    print(f"long_sb top: {num(r,'stall_long_sb'):8,d} ({100*num(r,'stall_long_sb')/tot:4.2f}%)  {r[idx['Source']].strip()[:50]}")
