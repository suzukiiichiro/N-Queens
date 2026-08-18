#!/usr/bin/env bash

set -Eeuo pipefail

# 167 single-pass validation harness
#
# Runs one N=21 full GPU pass and reconstructs cases 01-05 from the same
# 131-row progress TSV; case 06 verifies the complete total.  Candidate 167
# branches from the fastest validated 164 and changes only stack capacity:
# host code computes the exact queen-independent stored-frame schedule depth
# for each launch and selects one of four static kernels (MAXD 16/18/20/21).

SRC=${SRC:-./167Py_kernel_static_maxd_dispatch_probe.py}
CAND=${CAND:-./167Py_kernel_static_maxd_dispatch_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
LOCK_FILE=${LOCK_FILE:-/tmp/167Py_kernel_static_maxd_dispatch_validation.lock}
LOG_ROOT=${LOG_ROOT:-.}

N=${N:-21}
BLOCK=${BLOCK:-32}
MAX_BLOCKS=${MAX_BLOCKS:-484}
LOG_LEVEL=${LOG_LEVEL:-1}
SORT_MODE=${SORT_MODE:-0}
PRESET_QUEENS=${PRESET_QUEENS:-7}
BENCH_MODE=${BENCH_MODE:-31}
REORDER_WINDOW_MULT=${REORDER_WINDOW_MULT:-8}
REORDER_PHASE_JUMP=${REORDER_PHASE_JUMP:-7}
CROSS_STRIPE_SAFE=${CROSS_STRIPE_SAFE:-0}
WORKER_ID=${WORKER_ID:-0}
WORKER_COUNT=${WORKER_COUNT:-1}
BROADMARK_VARIANT=${BROADMARK_VARIANT:-2}

EXPECTED_CHUNKS=131
EXPECTED_TASKS=2025282
EXP01=16879968420
EXP02=19219113480
EXP03=16935522136
EXP04=16230260724
EXP05=79383179384
FULL_TOTAL=314666222712

BASELINE_166_SECONDS=${BASELINE_166_SECONDS:-670.976}
BASELINE_165_SECONDS=${BASELINE_165_SECONDS:-667.471}
BASELINE_164_SECONDS=${BASELINE_164_SECONDS:-633.526}
BASELINE_162_SECONDS=${BASELINE_162_SECONDS:-891.060}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed validation totals require N=21, BENCH_MODE=31, WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 167 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/167Py_kernel_static_maxd_dispatch_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
METRICS="$LOGDIR/metrics.env"
DISPATCH_METRICS="$LOGDIR/dispatch_metrics.env"
MODEL_METRICS="$LOGDIR/model_metrics.env"
ARCHIVED_PROGRESS="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"

printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

record_check() {
  local name=$1 expected=$2 actual=$3
  local status=FAIL
  if [[ "$actual" == "$expected" ]]; then status=OK; fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$status" >> "$SUMMARY"
  [[ "$status" == OK ]]
}

if [[ ! -f "$SRC" ]]; then
  echo "[error] source not found: $SRC" >&2
  exit 66
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] python3 is required for static/model validation" >&2
  exit 69
fi

if ! python3 - "$SRC" > "$MODEL_METRICS" <<'PYMODEL'
from pathlib import Path
import re, random, hashlib

import sys
src=Path(sys.argv[1])
s=src.read_text()

# Structural checks.
for md in (16,18,20,21):
    assert f'MAXD{md}:Static[int]={md}' in s
    assert s.count(f'def kernel_dfs_iter_gpu_maxd{md}(')==1
    assert s.count(f'__array__[u32](MAXD{md})')==5
assert len(re.findall(r'^@gpu\.kernel$',s,re.M))==4
assert not re.search(r'^MAXD:Static\[int\]=21$',s,re.M)
assert s.count('launch_kernel_dfs_iter_gpu_static_maxd(')==6
assert s.count('max_schedule_depth_of_tasks(')==6
assert s.count('[maxd-dispatch]')==3
assert '167 static-MAXD-dispatch GPU candidate supports N=5..27 only' in s

normalized=[]
for md in (16,18,20,21):
    st=s.index(f'@gpu.kernel\ndef kernel_dfs_iter_gpu_maxd{md}(')
    candidates=[x for x in (s.find('\n@gpu.kernel\n',st+1),s.find('\ndef launch_kernel_dfs_iter_gpu_static_maxd',st+1)) if x!=-1]
    en=min(candidates)
    b=s[st:en]
    for marker in ('root_raw:u32=ctrl0_arr[i]','root_cv:u32=(','ctrl[0]=root_cv',
                   'child_fu:u32=cv&u32(31)','child_cv:u32=(','ctrl[sp]=child_cv',
                   'while True:','if sp==0:'):
        assert marker in b, (md,marker)
    assert 'INIT_MASK' not in b
    assert 'u32(524288)' not in b
    assert 'ctrl=__array__[u16]' not in b
    n=b.replace(f'kernel_dfs_iter_gpu_maxd{md}','kernel_dfs_iter_gpu_maxdX').replace(f'MAXD{md}','MAXDX')
    n=re.sub(r'MAXD=\d+','MAXD=X',n).rstrip()
    normalized.append(n)
assert len(set(normalized))==1
assert all('if sp>=MAXD' not in b for b in normalized)

IS_BASE=69222408
IS_JMARK=4
IS_MARK=199209203
IS_P5=3840
SEL2=34742338
STP3=21266576
MASK_K_N3=185471169
MASK_K_N4=4227088
MASK_L_1=12689458
MASK_L_2=17039488
META=[1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26]

# Candidate-shaped depth model.
def depth_candidate(raw,mc):
    j=mc&31; e=(mc>>5)&31; m1=(mc>>10)&31; m2=(mc>>15)&31
    d=0
    while True:
        f=raw&31; row=(raw>>5)&31
        if ((IS_P5>>f)&1) and row==m1:
            f=META[f]
        if ((IS_BASE>>f)&1) and row==e:
            return d
        step=1; nxt=f
        if (IS_MARK>>f)&1:
            mark=m2 if ((SEL2>>f)&1) else m1
            if row==mark:
                step=3 if ((STP3>>f)&1) else 2
                nxt=META[f]
        if ((IS_JMARK>>f)&1) and row==j:
            nxt=META[f]
        child=row+step
        d+=1
        if d>21 or child>31:
            return 22
        raw=nxt|(child<<5)

# Independent reference: build the deterministic nonterminal schedule words.
def depth_reference(raw,mc):
    j=mc&31; e=(mc>>5)&31; m1=(mc>>10)&31; m2=(mc>>15)&31
    schedule=[]
    for _ in range(40):
        f=raw&31; row=(raw>>5)&31
        if f in (8,9,10,11) and row==m1:
            f=META[f]
        if ((IS_BASE>>f)&1) and row==e:
            return len(schedule) if len(schedule)<=21 else 22
        step=1; nxt=f
        if ((IS_MARK>>f)&1):
            chosen=m2 if ((SEL2>>f)&1) else m1
            if row==chosen:
                step=3 if ((STP3>>f)&1) else 2
                nxt=META[f]
        if f==2 and row==j:
            nxt=META[f]
        child=row+step
        schedule.append((f,row,nxt,child))
        if len(schedule)>21 or child>31:
            return 22
        raw=nxt|(child<<5)
    return 22

# Broad exhaustive control-state equivalence around every significant row.
depth_cases=0
for f in range(28):
    for row in range(31):
        vals=sorted({0,row,min(30,row+1),30,31})
        for j in vals:
            for e in vals:
                for m1 in vals:
                    for m2 in vals:
                        mc=j|(e<<5)|(m1<<10)|(m2<<15)
                        a=depth_candidate(f|(row<<5),mc)
                        b=depth_reference(f|(row<<5),mc)
                        if a!=b:
                            raise SystemExit(f'depth mismatch f={f} row={row} marks={(j,e,m1,m2)} {a}!={b}')
                        depth_cases+=1

# Dispatch thresholds.
def select(d):
    if d<=16:return 16
    if d<=18:return 18
    if d<=20:return 20
    if d<=21:return 21
    return 0
expected={**{i:16 for i in range(17)},17:18,18:18,19:20,20:20,21:21,22:0}
for d,v in expected.items():
    assert select(d)==v,(d,select(d),v)

# Exact 164 frame initializer and DFS model, with an explicit capacity.
def init_frame(raw,a,ld,j,e,m1,m2):
    f=raw&31; row=(raw>>5)&31
    if ((IS_P5>>f)&1) and row==m1:
        f=META[f]
    if ((IS_BASE>>f)&1) and row==e:
        add=1 if f!=14 else int((a&~1)!=0)
        return 'base',0,a,ld,add
    ism=(IS_MARK>>f)&1
    step=1; add1=0; blocks=0; bl=0; kt=0; future=1-ism; nxt=f
    if ism:
        mark=m2 if ((SEL2>>f)&1) else m1
        if row==mark:
            blocks=1
            bl=((MASK_L_1>>f)&1)|(((MASK_L_2>>f)&1)<<1)
            kt=((MASK_K_N3>>f)&1)|(((MASK_K_N4>>f)&1)<<1)
            future=0
            step=3 if ((STP3>>f)&1) else 2
            add1=1 if f==20 else 0
            nxt=META[f]
    if ((IS_JMARK>>f)&1) and row==j:
        a&=~1
        if a==0:
            return 'dead',0,a,ld,0
        ld|=1
        nxt=META[f]
    child=row+step
    fc=int(bool(future and child<e))
    cv=nxt|(child<<5)|(step<<10)|(add1<<12)|(blocks<<13)|(fc<<14)|(bl<<15)|(kt<<17)
    return 'ready',cv,a,ld,0

def dfs_count(N,raw,ld0,rd0,col0,a0,j,e,m1,m2,capacity,limit=200000):
    bm=(1<<N)-1; n3=1<<max(0,N-3); n4=1<<max(0,N-4)
    a0&=bm
    if a0==0:return 0,0
    st,cv,a0,ld0,add=init_frame(raw,a0,ld0,j,e,m1,m2)
    if st=='base':return add,0
    if st=='dead':return 0,0
    if capacity<1:raise OverflowError
    ld=[0]*capacity; rd=[0]*capacity; col=[0]*capacity; av=[0]*capacity; ctrl=[0]*capacity
    sp=0; ld[0]=ld0; rd[0]=rd0; col[0]=col0; av[0]=a0; ctrl[0]=cv
    maxframes=1; total=0; steps=0
    while True:
        steps+=1
        if steps>limit:raise RuntimeError
        a=av[sp]
        if a==0:
            if sp==0:break
            sp-=1;continue
        cv=ctrl[sp]; bit=a&-a; av[sp]=a^bit
        if cv&8192:
            step=(cv>>10)&3; add1=(cv>>12)&1; bl=(cv>>15)&3; kt=(cv>>17)&3
            bk=(n3&-(kt&1))|(n4&-((kt>>1)&1))
            nld=((ld[sp]|bit)<<step)|add1|bl; nrd=((rd[sp]|bit)>>step)|bk
        else:
            nld=(ld[sp]|bit)<<1; nrd=(rd[sp]|bit)>>1
        ncol=col[sp]|bit; nf=bm&~(nld|nrd|ncol)
        if nf==0:continue
        if (cv&16384) and bm&~((nld<<1)|(nrd>>1)|ncol)==0:continue
        st,child,nf,nld,add=init_frame(cv&1023,nf,nld,j,e,m1,m2)
        if st=='base':total+=add;continue
        if st=='dead':continue
        if sp+1>=capacity:raise OverflowError
        sp+=1; maxframes=max(maxframes,sp+1)
        ctrl[sp]=child;ld[sp]=nld;rd[sp]=nrd;col[sp]=ncol;av[sp]=nf
    return total,maxframes

random.seed(167164)
dfs_cases=0; attempts=0; max_observed=0
while dfs_cases<25000 and attempts<100000:
    attempts+=1
    N=random.randrange(5,10); bm=(1<<N)-1
    raw=random.randrange(28)|(random.randrange(0,N+2)<<5)
    j=random.randrange(0,N+1); e=random.randrange(0,N+1)
    m1=random.choice([31]+list(range(N+1))); m2=random.choice([31]+list(range(N+1)))
    mc=j|(e<<5)|(m1<<10)|(m2<<15)
    d=depth_candidate(raw,mc)
    if d>21:continue
    ld=random.randrange(bm+1);rd=random.randrange(1<<(N+2));col=random.randrange(bm+1);a=random.randrange(bm+1)
    try:
        ref,obs=dfs_count(N,raw,ld,rd,col,a,j,e,m1,m2,64)
        got,obs2=dfs_count(N,raw,ld,rd,col,a,j,e,m1,m2,select(d))
    except RuntimeError:
        continue
    if ref!=got or obs!=obs2:
        raise SystemExit('capacity DFS count mismatch')
    if obs>d:
        raise SystemExit(f'depth underestimation observed={obs} schedule={d}')
    max_observed=max(max_observed,obs)
    dfs_cases+=1
assert dfs_cases>=10000

# Broad N=21 / dynamic preset=6 superset.  preset=6 starts no earlier than row 3.
def classify(N,start,j,k,l):
    jm=0;m1=0;m2=0;e=0;f=0;N2=N-2
    jlt=j<N-3; je3=j==N-3; je2=j==N-2; klt=k<l
    sk=start<k; sl=start<l; adjl=l==k+1; adjk=k==l+1
    gate=j>2*N-34-start
    if jlt:
        jm=j+1;e=N2
        if gate:
            if klt:
                m1,m2=k-1,l-1
                if sl:
                    f=(0 if not adjl else 4) if sk else 1
                else:f=2
            else:
                m1,m2=l-1,k-1
                if sk:
                    f=(5 if not adjk else 7) if sl else 6
                else:f=2
        else:
            if klt:m1,m2=k-1,l-1;f=8 if not adjl else 9
            else:m1,m2=l-1,k-1;f=10 if not adjk else 11
    elif je3:
        e=N2
        if klt:
            m1,m2=k-1,l-1
            if sl:
                if sk:f=12 if not adjl else 15
                else:m2=l-1;f=13
            else:f=14
        else:
            m1,m2=l-1,k-1
            if sk:
                if sl:f=16 if not adjk else 18
                else:m2=k-1;f=17
            else:f=14
    elif je2:
        if klt:
            e=N2
            if sl:
                if sk:
                    m1=k-1
                    if not adjl:m2=l-1;f=19
                    else:f=22
                else:m2=l-1;f=20
            else:f=21
        else:
            if sk:
                if sl:
                    if k<N2:
                        m1,e=l-1,N2
                        if not adjk:m2=k-1;f=23
                        else:f=24
                    else:
                        if l!=N-3:m2,e=l-1,N-3;f=20
                        else:e=N-4;f=21
                else:
                    if k!=N2:m2,e=k-1,N2;f=25
                    else:e=N-3;f=21
            else:e=N2;f=21
    else:
        e=N2
        if start>k:f=26
        else:m1=k-1;f=27
    return f,jm,e,m1&31,m2&31

n21_max=0;n21_cases=0
N=21
for start in range(3,N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                if k==l:continue
                f,jm,e,m1,m2=classify(N,start,j,k,l)
                mc=jm|(e<<5)|(m1<<10)|(m2<<15)
                d=depth_candidate(f|(start<<5),mc)
                if d<=21:
                    n21_max=max(n21_max,d);n21_cases+=1
assert n21_max==16,n21_max

print(f'KERNEL_VARIANTS=4')
print(f'KERNEL_NORMALIZED_EQUAL=1')
print(f'DEPTH_MODEL_CASES={depth_cases}')
print(f'DISPATCH_BOUNDARY_CASES={len(expected)}')
print(f'DFS_CAPACITY_CASES={dfs_cases}')
print(f'DFS_MAX_OBSERVED={max_observed}')
print(f'N21_PRESET6_SUPERSET_CASES={n21_cases}')
print(f'N21_PRESET6_MAXD={n21_max}')

PYMODEL
then
  printf 'source_static_maxd_dispatch_shape\trequested layout\tmodel/static failure\tFAIL\n' >> "$SUMMARY"
  echo "[error] source/model validation failed" >&2
  exit 65
fi
# model_metrics.env contains fixed KEY=integer lines.
# shellcheck disable=SC1090
source "$MODEL_METRICS"

printf 'source_parent_baseline\tvalidated fastest 164 child-push-init layout\tverified\tOK\n' >> "$SUMMARY"
printf 'source_static_maxd_dispatch_shape\tMAXD 16/18/20/21 + host dispatch\tverified\tOK\n' >> "$SUMMARY"
printf 'kernel_variant_count\t4\t%s\tOK\n' "$KERNEL_VARIANTS" >> "$SUMMARY"
printf 'kernel_normalized_equivalence\t1\t%s\tOK\n' "$KERNEL_NORMALIZED_EQUAL" >> "$SUMMARY"
printf 'control_depth_model_equivalence\t>=500000 cases\t%s cases\tOK\n' "$DEPTH_MODEL_CASES" >> "$SUMMARY"
printf 'dispatch_threshold_cases\t23 cases\t%s cases\tOK\n' "$DISPATCH_BOUNDARY_CASES" >> "$SUMMARY"
printf 'dfs_capacity_safety_model\t>=10000 cases\t%s cases\tOK\n' "$DFS_CAPACITY_CASES" >> "$SUMMARY"
printf 'n21_preset6_superset_depth\tMAXD=16\tMAXD=%s (%s cases)\tOK\n' "$N21_PRESET6_MAXD" "$N21_PRESET6_SUPERSET_CASES" >> "$SUMMARY"

need_build=0
if [[ ! -x "$CAND" ]]; then
  need_build=1
elif [[ "$SRC" -nt "$CAND" ]]; then
  need_build=1
fi

if (( need_build )); then
  if [[ "$AUTO_BUILD" != "1" ]]; then
    echo "[error] candidate is missing/stale and AUTO_BUILD=$AUTO_BUILD: $CAND" >&2
    exit 66
  fi
  if ! command -v codon >/dev/null 2>&1; then
    echo "[error] codon was not found; cannot build $SRC" >&2
    exit 69
  fi
  echo "[build] codon build -release $SRC" | tee "$BUILD_LOG"
  set +e
  codon build -release "$SRC" 2>&1 | tee -a "$BUILD_LOG"
  build_rc=${PIPESTATUS[0]}
  set -e
  printf 'build_exit\t0\t%d\t%s\n' "$build_rc" "$([[ $build_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
  if (( build_rc != 0 )); then exit "$build_rc"; fi
else
  echo "[build] reuse executable: $CAND" | tee "$BUILD_LOG"
fi

if [[ ! -x "$CAND" ]]; then
  echo "[error] executable not found after build: $CAND" >&2
  exit 66
fi

CMD=(
  "$CAND"
  -g "$N" "$N"
  "$BLOCK" "$MAX_BLOCKS" "$LOG_LEVEL" "$SORT_MODE" "$PRESET_QUEENS"
  "$BENCH_MODE" "$REORDER_WINDOW_MULT" "$REORDER_PHASE_JUMP"
  "$CROSS_STRIPE_SAFE" "$WORKER_ID" "$WORKER_COUNT" "$BROADMARK_VARIANT"
)

{
  echo "================================================================"
  echo "candidate : $CAND"
  echo "source    : $SRC"
  echo "date      : $(date -Is)"
  echo "cwd       : $(pwd)"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "source_sha256: $(sha256sum "$SRC" | awk '{print $1}')"
  fi
  printf 'command   :'; printf ' %q' "${CMD[@]}"; echo
  echo "validation: one full run; cases 01-05 reconstructed from its progress TSV"
  echo "dispatch  : every launch must select MAXD16 (320 bytes/thread) for N=21 preset=6"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
printf 'run_exit\t0\t%d\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if (( run_rc != 0 )); then exit "$run_rc"; fi

# Runtime dispatch proof from the same 131 launches.
awk '
  /^\[maxd-dispatch\] N=21 scope=split145 / {
    rows++
    m=0; req=-1; sel=-1; bytes=-1; cap=""
    for (i=1;i<=NF;i++) {
      split($i,a,"=")
      if (a[1]=="m") m=a[2]+0
      else if (a[1]=="required_maxd") req=a[2]+0
      else if (a[1]=="selected_MAXD") sel=a[2]+0
      else if (a[1]=="stack_bytes_per_thread") bytes=a[2]+0
      else if (a[1]=="capacity_check") cap=a[2]
    }
    tasks+=m
    if (rows==1 || req<minreq) minreq=req
    if (req>maxreq) maxreq=req
    if (sel!=16) badsel++
    if (bytes!=320) badbytes++
    if (cap!="OK") badcap++
    if (req>sel) under++
  }
  END {
    printf "DISPATCH_ROWS=%d\n",rows+0
    printf "DISPATCH_TASKS=%.0f\n",tasks+0
    printf "DISPATCH_MIN_REQUIRED=%d\n",minreq+0
    printf "DISPATCH_MAX_REQUIRED=%d\n",maxreq+0
    printf "DISPATCH_NON16=%d\n",badsel+0
    printf "DISPATCH_NON320=%d\n",badbytes+0
    printf "DISPATCH_BAD_CAPACITY=%d\n",badcap+0
    printf "DISPATCH_UNDERSIZED=%d\n",under+0
  }
' "$RUN_LOG" > "$DISPATCH_METRICS"
# shellcheck disable=SC1090
source "$DISPATCH_METRICS"

failures=0
record_check "dispatch_launch_rows" "$EXPECTED_CHUNKS" "$DISPATCH_ROWS" || failures=$((failures+1))
record_check "dispatch_task_sum" "$EXPECTED_TASKS" "$DISPATCH_TASKS" || failures=$((failures+1))
record_check "dispatch_max_required" "16" "$DISPATCH_MAX_REQUIRED" || failures=$((failures+1))
record_check "dispatch_non_MAXD16" "0" "$DISPATCH_NON16" || failures=$((failures+1))
record_check "dispatch_non_320_bytes" "0" "$DISPATCH_NON320" || failures=$((failures+1))
record_check "dispatch_bad_capacity_flag" "0" "$DISPATCH_BAD_CAPACITY" || failures=$((failures+1))
record_check "dispatch_undersized_launch" "0" "$DISPATCH_UNDERSIZED" || failures=$((failures+1))

DYNAMIC_PRESET=$(sed -n 's/^\[dynamic-preset\] N=21 preset_queens=\([0-9][0-9]*\)$/\1/p' "$RUN_LOG" | tail -n1)
record_check "dynamic_preset_N21" "6" "${DYNAMIC_PRESET:-missing}" || failures=$((failures+1))

PROGRESS=$(sed -n 's/.* progress=\([^[:space:]]*\.tsv\).*/\1/p' "$RUN_LOG" | tail -n 1 | tr -d '\r')
if [[ -z "$PROGRESS" || ! -s "$PROGRESS" ]]; then
  printf 'progress_file\tpresent\t%s\tFAIL\n' "${PROGRESS:-missing}" >> "$SUMMARY"
  echo "[error] full-run progress TSV was not found" >&2
  exit 65
fi
cp -f "$PROGRESS" "$ARCHIVED_PROGRESS"
printf 'progress_file\tpresent\t%s\tOK\n' "$PROGRESS" >> "$SUMMARY"

awk -F '\t' -v expected_chunks="$EXPECTED_CHUNKS" '
  NR==1 {
    for (i=1;i<=NF;i++) {
      if ($i=="chunk") chunk_col=i
      else if ($i=="chunk_total") total_col=i
      else if ($i=="gpu_total") gpu_col=i
    }
    if (!chunk_col || !total_col || !gpu_col) { print "PARSE_OK=0"; exit 2 }
    next
  }
  {
    chunk=$(chunk_col)+0; value=$(total_col)+0; gpu=$(gpu_col)+0
    rows++; full+=value; last_gpu=gpu
    if (seen[chunk]++) duplicates++
    if (chunk==0 || chunk==20 || chunk==40 || chunk==60 || chunk==80 || chunk==100 || chunk==120) p1+=value
    if (chunk==35 || chunk==40 || chunk==41 || chunk==42 || chunk==47 || chunk==48 || chunk==52 || chunk==53) p2+=value
    if (chunk==20 || chunk==40 || chunk==55 || chunk==56 || chunk==57 || chunk==58 || chunk==60) p3+=value
    if (chunk==100 || chunk==105 || chunk==110 || chunk==115 || chunk==120 || chunk==125 || chunk==130) p4+=value
    if ((chunk%4)==0) p5+=value
  }
  END {
    if (!chunk_col || !total_col || !gpu_col) exit
    for (i=0;i<expected_chunks;i++) if (!(i in seen)) missing++
    printf "PARSE_OK=1\nROWS=%.0f\nDUPLICATES=%.0f\nMISSING=%.0f\n",rows,duplicates,missing
    printf "P1=%.0f\nP2=%.0f\nP3=%.0f\nP4=%.0f\nP5=%.0f\n",p1,p2,p3,p4,p5
    printf "FULL=%.0f\nLAST_GPU=%.0f\n",full,last_gpu
  }
' "$ARCHIVED_PROGRESS" > "$METRICS"
# shellcheck disable=SC1090
source "$METRICS"

record_check "progress_parse" "1" "$PARSE_OK" || failures=$((failures+1))
record_check "progress_rows" "$EXPECTED_CHUNKS" "$ROWS" || failures=$((failures+1))
record_check "duplicate_chunks" "0" "$DUPLICATES" || failures=$((failures+1))
record_check "missing_chunks" "0" "$MISSING" || failures=$((failures+1))
record_check "01_standard_7chunk" "$EXP01" "$P1" || failures=$((failures+1))
record_check "02_heavy_band" "$EXP02" "$P2" || failures=$((failures+1))
record_check "03_d2base14_density" "$EXP03" "$P3" || failures=$((failures+1))
record_check "04_late_tail" "$EXP04" "$P4" || failures=$((failures+1))
record_check "05_worker0of4_derived" "$EXP05" "$P5" || failures=$((failures+1))
record_check "06_full_chunk_sum" "$FULL_TOTAL" "$FULL" || failures=$((failures+1))
record_check "06_full_last_gpu" "$FULL_TOTAL" "$LAST_GPU" || failures=$((failures+1))

FINAL_LINE=$(grep -E "^[[:space:]]*${N}:" "$RUN_LOG" | tail -n1 || true)
if [[ "$FINAL_LINE" == *"$FULL_TOTAL"* && "$FINAL_LINE" == *"ok"* ]]; then
  printf 'final_output\t%s ... ok\t%s\tOK\n' "$FULL_TOTAL" "$FINAL_LINE" >> "$SUMMARY"
else
  printf 'final_output\t%s ... ok\t%s\tFAIL\n' "$FULL_TOTAL" "${FINAL_LINE:-missing}" >> "$SUMMARY"
  failures=$((failures+1))
fi

ERROR_HITS=$(grep -Eic '\[(.*-)?error\]|mismatch|ng\(' "$RUN_LOG" || true)
record_check "error_or_mismatch_hits" "0" "$ERROR_HITS" || failures=$((failures+1))

ELAPSED_TEXT=$(awk -v n="$N" '$0 ~ "^[[:space:]]*" n ":" {v=$(NF-1)} END {print v}' "$RUN_LOG")
if [[ -n "$ELAPSED_TEXT" ]]; then
  ELAPSED_SECONDS=$(awk -F: '{printf "%.3f",($1*3600)+($2*60)+$3}' <<< "$ELAPSED_TEXT")
  for pair in "166:$BASELINE_166_SECONDS" "165:$BASELINE_165_SECONDS" "164:$BASELINE_164_SECONDS" "162:$BASELINE_162_SECONDS"; do
    label=${pair%%:*}; base=${pair#*:}
    perf=$(awk -v base="$base" -v now="$ELAPSED_SECONDS" 'BEGIN {d=base-now;p=(base>0)?d/base*100:0;printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%",now,base,d,p}')
    printf 'timing_vs_%s\t%s_full=%ss\t%s\tINFO\n' "$label" "$label" "$base" "$perf" >> "$SUMMARY"
  done
fi

WARN_HITS=$(grep -Eic '\[.*warning\]' "$RUN_LOG" || true)
printf 'warning_hits\t0 preferred\t%s\tINFO\n' "$WARN_HITS" >> "$SUMMARY"

echo
echo "================================================================"
echo "[summary]"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "[progress] $ARCHIVED_PROGRESS"
echo "[logdir]   $LOGDIR"
echo "================================================================"

if (( failures != 0 )); then
  echo "[validation-failed] failures=$failures" >&2
  exit 1
fi

echo "[validation-ok] 167 static MAXD dispatch reproduced cases 01-06 with MAXD16 on all N=21 launches"
