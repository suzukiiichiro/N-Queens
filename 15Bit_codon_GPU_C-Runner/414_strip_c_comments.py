#!/usr/bin/env python3
"""414_strip_c_comments.py <file.cu> -- print the code region (block and line comments removed, blank lines dropped)."""
import sys,re
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
s=re.sub(r'/\*.*?\*/','',s,flags=re.S)
s=re.sub(r'//[^\n]*','',s)
print('\n'.join(l.rstrip() for l in s.split('\n') if l.strip()))
