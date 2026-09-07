# rev386_validation_helpers.py
#
# rev386 -- validation/diagnostic helper functions relocated out of
# 385Py_kernel_maxd14_final.py (386Py_kernel_maxd14_final.py after this
# revision's rename), following the exact local multi-file import
# mechanism 384's probe confirmed works on real hardware (codon build
# -release survives `import rev386_validation_helpers`).
#
# Selection criterion for what moved here: pure file-I/O / bookkeeping
# / string-parsing verification helpers with NO dependency on the
# kernel-adjacent data structures (TaskSoA, build_soa_for_range, the
# meta_next schedule table). Functions like check_required_maxd_for_N
# or probe_partial_load_memory are diagnostic in *purpose* but still
# depend on those core structures, so they stay in the main file for
# this revision rather than forcing a circular import or moving core
# data-structure code along with them. This is a conservative first
# pass, not a claim that every "verification" function has been moved.
#
# Naming note: this file is "rev386_..." rather than "386Py_..." for
# the same reason 384's helper was "rev384_..." -- import statements
# require a valid identifier, and identifiers cannot start with a
# digit. See 384Py_multifile_osexec_probe.py's own naming-note comment
# for the precedent.
#
# No logic was changed during the move -- every function below is
# byte-identical to its 385Py_kernel_maxd14_final.py source, modulo
# this header and blank-line spacing at the top/bottom of each block.

from typing import List,Tuple

def validate_chunk_range(label:str,start:int,end:int,total:int)->bool:
  ok:bool=True
  if start<0:
    print(f"[cross-stripe-safe][error] {label}: start < 0 start={start} total={total}")
    ok=False
  if end<start:
    print(f"[cross-stripe-safe][error] {label}: end < start start={start} end={end} total={total}")
    ok=False
  if end>total:
    print(f"[cross-stripe-safe][error] {label}: end > total start={start} end={end} total={total}")
    ok=False
  if ok and start==end:
    print(f"[cross-stripe-safe][warn] {label}: empty range start={start} end={end} total={total}")
  return ok


def validate_reordered_count(label:str,expected:int,actual:int)->bool:
  if expected!=actual:
    print(f"[stripe-reorder][error] {label}: reordered count mismatch expected={expected} actual={actual}")
    return False
  return True


def validate_reordered_indices(label:str,expected:int,idxs:List[int])->bool:
  if not validate_reordered_count(label,expected,len(idxs)):
    return False
  seen:List[int]=[0]*expected
  for v in idxs:
    if v<0 or v>=expected:
      print(f"[cross-stripe-safe][error] {label}: index out of range idx={v} expected={expected}")
      return False
    if seen[v]!=0:
      print(f"[cross-stripe-safe][error] {label}: duplicated index idx={v}")
      return False
    seen[v]=1
  missing:int=0
  first_missing:int=-1
  for i in range(expected):
    if seen[i]==0:
      missing+=1
      if first_missing<0:
        first_missing=i
  if missing!=0:
    print(f"[cross-stripe-safe][error] {label}: missing count={missing} first_missing={first_missing}")
    return False
  return True


def file_exists(fname:str)->bool:
  try:
    with open(fname,"rb"):
      return True
  except:
    return False


def validate_bin_file(fname:str)->bool:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)  # ファイル末尾に移動
      size=f.tell()
    return size%16==0
  except:
    return False


def count_constellations_bin_records(fname:str)->int:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)
      size:int=f.tell()
    if size%16!=0:
      return 0
    return size//16
  except:
    return 0


def read_stream_done_count(fname:str)->int:
  try:
    with open(fname,"r") as f:
      text:str=f.read().strip()
    if text=="":
      return -1
    return int(text)
  except:
    return -1


def write_stream_done_count(fname:str,count:int)->None:
  with open(fname,"w") as f:
    f.write(str(count))
    f.write("\n")


def read_vmhwm_kb()->int:
  try:
    text:str=""
    with open("/proc/self/status","r") as f:
      while True:
        chunk:str=f.read(4096)
        if chunk=="":
          break
        text+=chunk
    lines:List[str]=text.split("\n")
    for line in lines:
      if line.startswith("VmHWM:"):
        parts:List[str]=line.split()
        if len(parts)>=2:
          return int(parts[1])
    return -1
  except:
    return -1
# ===370-VMHWM-FIX-END===


def crunner_parse_result(log_path:str,done_prefix:str,correctness_prefix:str)->Tuple[int,int,int,int]:
  # Returns (found_done, total_sum, kernel_ms_x1000, match_status).
  # kernel_ms is carried as an integer milliseconds*1000 to avoid any
  # float-formatting ambiguity while scanning text; callers divide by
  # 1000.0 to get the float back. match_status: 0=no correctness line
  # seen, 1=MATCH, 2=MISMATCH. total_sum/kernel_ms_x1000 stay -1 if
  # never parsed out of a done_prefix line.
  found_done:int=0
  total_sum:int=-1
  kernel_ms_x1000:int=-1
  match_status:int=0
  f=open(log_path,"r")
  for raw_line in f:
    line:str=raw_line.strip()
    if line.startswith(done_prefix):
      found_done=1
      for tok in line.split(" "):
        if tok.startswith("total_sum="):
          total_sum=int(tok[len("total_sum="):])
        if tok.startswith("kernel_ms="):
          kernel_ms_x1000=int(float(tok[len("kernel_ms="):])*1000.0)
    if line.startswith(correctness_prefix+" MATCH"):
      match_status=1
    elif line.startswith(correctness_prefix+" MISMATCH"):
      match_status=2
  f.close()
  return found_done,total_sum,kernel_ms_x1000,match_status


def crunner_input_fname(stream_fname:str)->str:
  return f"{stream_fname}.soa_ref_361.bin.maxd14only_363.bin"


def crunner_input_valid(fname:str)->bool:
  if not file_exists(fname):
    return False
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)
      size:int=f.tell()
    return size>0 and size%28==0
  except:
    return False
# ===385-INPUT-FIX-END===

