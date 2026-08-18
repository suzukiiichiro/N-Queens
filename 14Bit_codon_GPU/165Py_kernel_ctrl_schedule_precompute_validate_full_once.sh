#!/usr/bin/env bash

set -Eeuo pipefail

# 165 single-pass validation harness
#
# Runs one N=21 full GPU pass and reconstructs cases 01-05 from the same
# 131-row progress TSV; case 06 verifies the complete total.  A flock guard
# prevents concurrent GPU validation runs.
#
# Candidate 165 branches from validated 164.  It keeps the five u32[MAXD]
# static arrays and every validated host/order/cache change, but replaces the
# per-push child classification/ctrl write with a one-time depth-indexed
# ctrl_schedule.  Each parent schedule word encodes its child entry action
# (ordinary/jmark/base/base14).  The N=5..27 GPU safety envelope is inherited.

SRC=${SRC:-./165Py_kernel_ctrl_schedule_precompute_probe.py}
CAND=${CAND:-./165Py_kernel_ctrl_schedule_precompute_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
LOCK_FILE=${LOCK_FILE:-/tmp/165Py_kernel_ctrl_schedule_precompute_validation.lock}
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
EXP01=16879968420
EXP02=19219113480
EXP03=16935522136
EXP04=16230260724
EXP05=79383179384
FULL_TOTAL=314666222712

# Direct validated 164/162..149 full runs and historical 148 mean (informational only).
BASELINE_164_SECONDS=${BASELINE_164_SECONDS:-633.526}
BASELINE_162_SECONDS=${BASELINE_162_SECONDS:-891.060}
BASELINE_161_SECONDS=${BASELINE_161_SECONDS:-905.957}
BASELINE_160_SECONDS=${BASELINE_160_SECONDS:-924.656}
BASELINE_159_SECONDS=${BASELINE_159_SECONDS:-954.667}
BASELINE_158_SECONDS=${BASELINE_158_SECONDS:-954.673}
BASELINE_157_SECONDS=${BASELINE_157_SECONDS:-958.631}
BASELINE_156_SECONDS=${BASELINE_156_SECONDS:-969.213}
BASELINE_155_SECONDS=${BASELINE_155_SECONDS:-964.864}
BASELINE_154_SECONDS=${BASELINE_154_SECONDS:-963.903}
BASELINE_153_SECONDS=${BASELINE_153_SECONDS:-965.305}
BASELINE_152_SECONDS=${BASELINE_152_SECONDS:-970.212}
BASELINE_151_SECONDS=${BASELINE_151_SECONDS:-976.854}
BASELINE_150_SECONDS=${BASELINE_150_SECONDS:-976.988}
BASELINE_149_SECONDS=${BASELINE_149_SECONDS:-976.932}
BASELINE_148_SECONDS=${BASELINE_148_SECONDS:-976.825}

if [[ "$N" != "21" || "$BENCH_MODE" != "31" || "$WORKER_ID" != "0" || "$WORKER_COUNT" != "1" ]]; then
  echo "[error] fixed validation totals require N=21, BENCH_MODE=31, WORKER_ID=0, WORKER_COUNT=1" >&2
  exit 64
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another 165 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/165Py_kernel_ctrl_schedule_precompute_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
METRICS="$LOGDIR/metrics.env"
MODEL_METRICS="$LOGDIR/model_metrics.env"
ARCHIVED_PROGRESS="$LOGDIR/progress_full.tsv"
mkdir -p "$LOGDIR"

printf 'check\texpected\tactual\tstatus\n' > "$SUMMARY"

record_check() {
  local name=$1 expected=$2 actual=$3
  local status=FAIL
  if [[ "$actual" == "$expected" ]]; then
    status=OK
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$status" >> "$SUMMARY"
  [[ "$status" == OK ]]
}

if [[ -f "$SRC" ]]; then
  source_failures=0
  KERNEL_SNIP="$LOGDIR/kernel_source.tmp"
  awk '
    /^@gpu\.kernel$/ { in_kernel=1 }
    in_kernel { print }
    in_kernel && /^####################################################################################################$/ { exit }
  ' "$SRC" > "$KERNEL_SNIP"

  # Inherited root-pop loop and removed overflow guard.  165 adds one separate
  # pre-DFS schedule loop, so exactly two while True loops are expected.
  true_loop_count=$(grep -Ec '^[[:space:]]*while[[:space:]]+True:' "$KERNEL_SNIP" || true)
  legacy_loop_count=$(grep -Ec '^[[:space:]]*while[[:space:]]+sp[[:space:]]*>=[[:space:]]*0:' "$KERNEL_SNIP" || true)
  root_guard_count=$(grep -Ec '^[[:space:]]*if[[:space:]]+sp==0:' "$KERNEL_SNIP" || true)
  pop_decrement_count=$(grep -Ec '^[[:space:]]*sp-=1[[:space:]]*$' "$KERNEL_SNIP" || true)
  push_count=$(grep -Ec '^[[:space:]]*sp\+=1[[:space:]]*$' "$KERNEL_SNIP" || true)
  stack_guard_count=$(grep -Ec '^[[:space:]]*if[[:space:]]+sp[[:space:]]*>=[[:space:]]*MAXD:' "$KERNEL_SNIP" || true)
  [[ "$true_loop_count" == "2" ]] || source_failures=$((source_failures+1))
  [[ "$legacy_loop_count" == "0" ]] || source_failures=$((source_failures+1))
  [[ "$root_guard_count" == "1" ]] || source_failures=$((source_failures+1))
  [[ "$pop_decrement_count" == "1" ]] || source_failures=$((source_failures+1))
  [[ "$push_count" == "1" ]] || source_failures=$((source_failures+1))
  [[ "$stack_guard_count" == "0" ]] || source_failures=$((source_failures+1))
  grep -Fq 'sp:int=0' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  grep -Fq 'MAXD:Static[int]=21' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'if use_gpu and (nmin<5 or nmax>28):' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq '165 ctrl-schedule-precompute GPU candidate supports N=5..27 only' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'VERSION_TAG:str="165 depth-indexed ctrl schedule precompute over 164 child frame push initialization' "$SRC" || source_failures=$((source_failures+1))

  # 165 sole structural experiment: a u32 depth schedule replaces the mutable
  # ctrl stack.  Schedule writes occur only before DFS; push writes state only.
  grep -Fq 'ctrl_schedule=__array__[u32](MAXD)' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  if grep -Eq '^[[:space:]]*ctrl=__array__\[' "$KERNEL_SNIP" || \
     grep -Eq '^[[:space:]]*ctrl\[' "$KERNEL_SNIP" || \
     grep -Eq 'ctrl_schedule=__array__\[u16\]' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi
  schedule_write_count=$(grep -Ec '^[[:space:]]*ctrl_schedule\[[^]]+\][|]?=' "$KERNEL_SNIP" || true)
  schedule_dfs_read_count=$(grep -Fc 'cv:u32=ctrl_schedule[sp]' "$KERNEL_SNIP" || true)
  [[ "$schedule_write_count" == "2" ]] || source_failures=$((source_failures+1))
  [[ "$schedule_dfs_read_count" == "1" ]] || source_failures=$((source_failures+1))
  for expr in \
    'schedule_raw:u32=ctrl0_arr[i]' \
    'schedule_depth:int=0' \
    'root_action:u32=u32(0)' \
    'ctrl_schedule[schedule_depth-1]|=(frame_action<<u32(19))' \
    'ctrl_schedule[schedule_depth]=frame_cv' \
    'schedule_raw=frame_cv&u32(1023)' \
    'frame_action=u32(3) if schedule_fu==u32(14) else u32(2)' \
    'child_action:u32=(cv>>u32(19))&u32(3)' \
    'if child_action>=u32(2):' \
    'if child_action==u32(1):'
  do
    grep -Fq "$expr" "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  done
  if ! awk '
    /schedule_raw:u32=ctrl0_arr\[i\]/ { seed=NR }
    /ctrl_schedule\[schedule_depth\]=frame_cv/ { build=NR }
    /^[[:space:]]*sp:int=0/ { sp=NR }
    /cv:u32=ctrl_schedule\[sp\]/ { read=NR }
    END { exit !((seed>0) && (build>seed) && (sp>build) && (read>sp)) }
  ' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi
  if awk '
    /cv:u32=ctrl_schedule\[sp\]/ { dfs=1 }
    dfs && /^[[:space:]]*ctrl_schedule\[[^]]+\][|]?=/ { bad=1 }
    END { exit !bad }
  ' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi

  # All frame classification and meta_next access must be confined to the
  # one-time schedule pass; the 164 child_* pack machinery is absent.
  for legacy in \
    'child_fu:u32=' \
    'child_rowv:u32=' \
    'child_isbu:u32=' \
    'child_ismu:u32=' \
    'child_isju:u32=' \
    'child_nextfidu:u32=' \
    'child_cv:u32=(' \
    'ctrl[sp]=child_cv'
  do
    if grep -Fq "$legacy" "$KERNEL_SNIP"; then
      source_failures=$((source_failures+1))
    fi
  done
  meta_next_u32_refs=$(grep -Fc 'u32(meta_next[int(schedule_fu)])' "$KERNEL_SNIP" || true)
  int_fu_refs=$(grep -Ec '^[[:space:]]*[^#].*int\(schedule_fu\)' "$KERNEL_SNIP" || true)
  [[ "$meta_next_u32_refs" == "3" ]] || source_failures=$((source_failures+1))
  [[ "$int_fu_refs" == "3" ]] || source_failures=$((source_failures+1))

  # Root entry action and child action preserve 164 base14/jmark semantics.
  for expr in \
    'if root_action==u32(2):' \
    'results[i]=w_arr[i]' \
    'if root_action==u32(3):' \
    'total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)' \
    'if root_action==u32(1):' \
    'root_a&=~u32(1)' \
    'root_ld|=u32(1)' \
    'total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)' \
    'nf&=~u32(1)' \
    'nld|=u32(1)'
  do
    grep -Fq "$expr" "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  done
  if grep -Fq 'INIT_MASK' "$KERNEL_SNIP" || \
     grep -Fq 'u32(524288)' "$KERNEL_SNIP" || \
     grep -Eq '<<[[:space:]]*u32\(21\)' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi

  # Retain the validated u32 input/mark layout and all ten classification masks.
  for decl in \
    'self.ld_arr:List[u32]' \
    'self.rd_arr:List[u32]' \
    'self.col_arr:List[u32]' \
    'self.free_arr:List[u32]' \
    'self.ctrl0_arr:List[u32]=[u32(0)]*m' \
    'self.markctrl_arr:List[u32]=[u32(0)]*m' \
    'self.jmark_arr:List[int]=[0]*m' \
    'self.end_arr:List[int]=[0]*m' \
    'self.mark1_arr:List[int]=[0]*m' \
    'self.mark2_arr:List[int]=[0]*m' \
    'm:int,board_mask:u32' \
    'n3:u32,n4:u32' \
    'soa.ctrl0_arr[t]=u32(target)|(u32(start)<<u32(5))' \
    'soa.markctrl_arr[t]=('
  do
    grep -Fq "$decl" "$SRC" || source_failures=$((source_failures+1))
  done
  for decl in \
    'IS_BASE_MASK:u32=u32(69222408)' \
    'IS_JMARK_MASK:u32=u32(4)' \
    'IS_MARK_MASK:u32=u32(199209203)' \
    'IS_P5_MASK:u32=u32(3840)' \
    'SEL2_MASK:u32=u32(34742338)' \
    'STP3_MASK:u32=u32(21266576)' \
    'MASK_K_N3:u32=u32(185471169)' \
    'MASK_K_N4:u32=u32(4227088)' \
    'MASK_L_1:u32=u32(12689458)' \
    'MASK_L_2:u32=u32(17039488)'
  do
    grep -Fq "$decl" "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  done

  markctrl_launch_refs=$(grep -Ec 'gpu\.raw\((sort_)?soa\.markctrl_arr\)' "$SRC" || true)
  sort_markctrl_copies=$(grep -Fc 'sort_soa.markctrl_arr[p]=soa.markctrl_arr[q]' "$SRC" || true)
  [[ "$markctrl_launch_refs" == "5" ]] || source_failures=$((source_failures+1))
  [[ "$sort_markctrl_copies" == "2" ]] || source_failures=$((source_failures+1))

  # Deterministic model: per-frame action equivalence, exhaustive 21-bit
  # schedule packing, schedule-chain replay, and 164-vs-165 DFS counts.
  FRAME_MODEL_CASES=0
  SCHEDULE_PACK_CASES=0
  SCHEDULE_CHAIN_CASES=0
  DFS_MODEL_CASES=0
  if ! command -v python3 >/dev/null 2>&1; then
    source_failures=$((source_failures+1))
  elif ! python3 - <<'PYMODEL' > "$MODEL_METRICS"
import random

IS_BASE_MASK=69222408
IS_JMARK_MASK=4
IS_MARK_MASK=199209203
IS_P5_MASK=3840
SEL2_MASK=34742338
STP3_MASK=21266576
MASK_K_N3=185471169
MASK_K_N4=4227088
MASK_L_1=12689458
MASK_L_2=17039488
META=[1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26]


def init164(raw,a,ld,jmark,endm,mark1,mark2):
    fu=raw&31
    row=(raw>>5)&31
    if ((IS_P5_MASK>>fu)&1) and row==mark1:
        fu=META[fu]
    if ((IS_BASE_MASK>>fu)&1) and row==endm:
        add=1 if fu!=14 else int((a&~1)!=0)
        return ('base',0,a,ld,add)
    ism=(IS_MARK_MASK>>fu)&1
    step=1; add1=0; blocks=0; bl=0; kt=0; future=1-ism; nxt=fu
    if ism:
        mark=mark2 if ((SEL2_MASK>>fu)&1) else mark1
        if row==mark:
            blocks=1
            bl=((MASK_L_1>>fu)&1)|(((MASK_L_2>>fu)&1)<<1)
            kt=((MASK_K_N3>>fu)&1)|(((MASK_K_N4>>fu)&1)<<1)
            future=0
            step=3 if ((STP3_MASK>>fu)&1) else 2
            add1=1 if fu==20 else 0
            nxt=META[fu]
    if ((IS_JMARK_MASK>>fu)&1) and row==jmark:
        a&=~1
        if a==0:
            return ('dead',0,a,ld,0)
        ld|=1
        nxt=META[fu]
    child=row+step
    fc=int(bool(future and child<endm))
    cv=nxt|(child<<5)|(step<<10)|(add1<<12)|(blocks<<13)|(fc<<14)|(bl<<15)|(kt<<17)
    return ('ready',cv,a,ld,0)


def static165(raw,jmark,endm,mark1,mark2):
    fu=raw&31
    row=(raw>>5)&31
    if ((IS_P5_MASK>>fu)&1) and row==mark1:
        fu=META[fu]
    if ((IS_BASE_MASK>>fu)&1) and row==endm:
        return (3 if fu==14 else 2,0)
    ism=(IS_MARK_MASK>>fu)&1
    step=1; add1=0; blocks=0; bl=0; kt=0; future=1-ism; nxt=fu
    if ism:
        mark=mark2 if ((SEL2_MASK>>fu)&1) else mark1
        if row==mark:
            blocks=1
            bl=((MASK_L_1>>fu)&1)|(((MASK_L_2>>fu)&1)<<1)
            kt=((MASK_K_N3>>fu)&1)|(((MASK_K_N4>>fu)&1)<<1)
            future=0
            step=3 if ((STP3_MASK>>fu)&1) else 2
            add1=1 if fu==20 else 0
            nxt=META[fu]
    action=0
    if ((IS_JMARK_MASK>>fu)&1) and row==jmark:
        action=1
        nxt=META[fu]
    child=row+step
    fc=int(bool(future and child<endm))
    cv=nxt|(child<<5)|(step<<10)|(add1<<12)|(blocks<<13)|(fc<<14)|(bl<<15)|(kt<<17)
    return action,cv


frame_cases=0
for f in range(28):
  for row in range(31):
    raw=f|(row<<5)
    scenarios=[
      (0,30,31,31),
      (row,30,31,31),
      (0,row,31,31),
      (0,30,row,31),
      (0,30,31,row),
      (row,row,row,row),
    ]
    for j,e,m1,m2 in scenarios:
      action,cv=static165(raw,j,e,m1,m2)
      for a in (1,2,3,5,0x7fffffff):
        for ld in (0,1,0x15555):
          st,oldcv,olda,oldld,add=init164(raw,a,ld,j,e,m1,m2)
          if action==2:
            if st!='base' or add!=1:
              raise SystemExit('normal base action mismatch')
          elif action==3:
            if st!='base' or add!=int((a&~1)!=0):
              raise SystemExit('base14 action mismatch')
          elif action==1:
            if (a&~1)==0:
              if st!='dead':
                raise SystemExit('jmark dead mismatch')
            else:
              if st!='ready' or oldcv!=cv or olda!=(a&~1) or oldld!=(ld|1):
                raise SystemExit('jmark ready mismatch')
          else:
            if st!='ready' or oldcv!=cv or olda!=a or oldld!=ld:
              raise SystemExit('ordinary frame mismatch')
          frame_cases+=1

pack_cases=0
for nxt in range(28):
  for row in range(32):
    for step in range(1,4):
      for add1 in range(2):
        for blocks in range(2):
          for future in range(2):
            for bl in range(4):
              for kt in range(3):
                low=nxt|(row<<5)|(step<<10)|(add1<<12)|(blocks<<13)|(future<<14)|(bl<<15)|(kt<<17)
                for action in range(4):
                  cv=low|(action<<19)
                  if (cv&31)!=nxt or ((cv>>5)&31)!=row or ((cv>>10)&3)!=step:
                    raise SystemExit('schedule low-field mismatch')
                  if ((cv>>12)&1)!=add1 or ((cv>>13)&1)!=blocks or ((cv>>14)&1)!=future:
                    raise SystemExit('schedule flag mismatch')
                  if ((cv>>15)&3)!=bl or ((cv>>17)&3)!=kt or ((cv>>19)&3)!=action:
                    raise SystemExit('schedule action mismatch')
                  if cv >= (1<<21):
                    raise SystemExit('schedule width mismatch')
                  pack_cases+=1


def build_schedule(raw,j,e,m1,m2,maxd=64):
    schedule=[0]*maxd
    depth=0
    root_action=0
    while True:
      action,cv=static165(raw,j,e,m1,m2)
      if depth==0:
        root_action=action
      else:
        schedule[depth-1]|=action<<19
      if action>=2:
        return root_action,schedule[:depth]
      if depth>=maxd:
        raise RuntimeError('schedule overflow')
      schedule[depth]=cv
      raw=cv&1023
      depth+=1

random.seed(165164)
schedule_cases=0
for _ in range(50000):
    end=random.randrange(1,31)
    row=random.randrange(0,end+1)
    raw=random.randrange(28)|(row<<5)
    j=random.randrange(0,end+1)
    m1=random.choice([31]+list(range(0,end+1)))
    m2=random.choice([31]+list(range(0,end+1)))
    try:
      root_action,schedule=build_schedule(raw,j,end,m1,m2,64)
    except RuntimeError:
      continue
    # Independently replay 164 frame initialization with a non-dead availability.
    replay=raw
    depth=0
    while True:
      st,cv,a,ld,add=init164(replay,3,0,j,end,m1,m2)
      action=root_action if depth==0 else ((schedule[depth-1]>>19)&3)
      if st=='base':
        expected=3 if ((replay&31)==14 and ((replay>>5)&31)==end) else None
        if action not in (2,3):
          raise SystemExit('schedule terminal action missing')
        break
      if st=='dead':
        raise SystemExit('availability 3 must survive jmark')
      if action>=2:
        raise SystemExit('premature schedule terminal action')
      if depth>=len(schedule) or (schedule[depth]&((1<<19)-1))!=cv:
        raise SystemExit('schedule frame metadata mismatch')
      replay=cv&1023
      depth+=1
      if depth>64:
        raise SystemExit('replay overflow')
    if depth!=len(schedule):
      raise SystemExit('schedule length mismatch')
    schedule_cases+=1
if schedule_cases<20000:
    raise SystemExit(f'insufficient schedule cases: {schedule_cases}')


def count164(N,raw,ld0,rd0,col0,a0,j,e,m1,m2,limit=200000):
    bm=(1<<N)-1; n3=1<<max(0,N-3); n4=1<<max(0,N-4)
    a0&=bm
    if a0==0:
      return 0
    st,cv,a0,ld0,add=init164(raw,a0,ld0,j,e,m1,m2)
    if st=='base':
      return add
    if st=='dead':
      return 0
    ld=[0]*64; rd=[0]*64; col=[0]*64; av=[0]*64; ctrl=[0]*64
    sp=0; ld[0]=ld0; rd[0]=rd0; col[0]=col0; av[0]=a0; ctrl[0]=cv
    total=0; steps=0
    while True:
      steps+=1
      if steps>limit:
        raise RuntimeError
      a=av[sp]
      if a==0:
        if sp==0:
          break
        sp-=1
        continue
      cv=ctrl[sp]
      bit=a&-a
      av[sp]=a^bit
      if cv&8192:
        step=(cv>>10)&3; add1=(cv>>12)&1; bl=(cv>>15)&3; kt=(cv>>17)&3
        bk=(n3&-(kt&1))|(n4&-((kt>>1)&1))
        nld=((ld[sp]|bit)<<step)|add1|bl
        nrd=((rd[sp]|bit)>>step)|bk
      else:
        nld=(ld[sp]|bit)<<1
        nrd=(rd[sp]|bit)>>1
      ncol=col[sp]|bit
      nf=bm&~(nld|nrd|ncol)
      if nf==0:
        continue
      if (cv&16384) and bm&~((nld<<1)|(nrd>>1)|ncol)==0:
        continue
      st,child,nf,nld,add=init164(cv&1023,nf,nld,j,e,m1,m2)
      if st=='base':
        total+=add
        continue
      if st=='dead':
        continue
      sp+=1
      if sp>=64:
        raise RuntimeError
      ctrl[sp]=child; ld[sp]=nld; rd[sp]=nrd; col[sp]=ncol; av[sp]=nf
    return total


def count165(N,raw,ld0,rd0,col0,a0,j,e,m1,m2,limit=200000):
    bm=(1<<N)-1; n3=1<<max(0,N-3); n4=1<<max(0,N-4)
    a0&=bm
    if a0==0:
      return 0
    root_action,schedule=build_schedule(raw,j,e,m1,m2,64)
    if root_action==2:
      return 1
    if root_action==3:
      return int((a0&~1)!=0)
    if root_action==1:
      a0&=~1
      if a0==0:
        return 0
      ld0|=1
    ld=[0]*64; rd=[0]*64; col=[0]*64; av=[0]*64
    sp=0; ld[0]=ld0; rd[0]=rd0; col[0]=col0; av[0]=a0
    total=0; steps=0
    while True:
      steps+=1
      if steps>limit:
        raise RuntimeError
      a=av[sp]
      if a==0:
        if sp==0:
          break
        sp-=1
        continue
      cv=schedule[sp]
      bit=a&-a
      av[sp]=a^bit
      if cv&8192:
        step=(cv>>10)&3; add1=(cv>>12)&1; bl=(cv>>15)&3; kt=(cv>>17)&3
        bk=(n3&-(kt&1))|(n4&-((kt>>1)&1))
        nld=((ld[sp]|bit)<<step)|add1|bl
        nrd=((rd[sp]|bit)>>step)|bk
      else:
        nld=(ld[sp]|bit)<<1
        nrd=(rd[sp]|bit)>>1
      ncol=col[sp]|bit
      nf=bm&~(nld|nrd|ncol)
      if nf==0:
        continue
      if (cv&16384) and bm&~((nld<<1)|(nrd>>1)|ncol)==0:
        continue
      action=(cv>>19)&3
      if action>=2:
        total+=1 if action==2 else int((nf&~1)!=0)
        continue
      if action==1:
        nf&=~1
        if nf==0:
          continue
        nld|=1
      sp+=1
      if sp>=len(schedule):
        raise RuntimeError
      ld[sp]=nld; rd[sp]=nrd; col[sp]=ncol; av[sp]=nf
    return total

random.seed(16516421)
dfs_cases=0
for N in range(5,10):
  bm=(1<<N)-1
  for _ in range(6000):
    end=random.randrange(1,N+1)
    row=random.randrange(0,end+1)
    raw=random.randrange(28)|(row<<5)
    ld=random.randrange(bm+1)
    rd=random.randrange(1<<(N+2))
    col=random.randrange(bm+1)
    a=random.randrange(bm+1)
    j=random.randrange(0,end+1)
    m1=random.choice([31]+list(range(0,end+1)))
    m2=random.choice([31]+list(range(0,end+1)))
    try:
      old=count164(N,raw,ld,rd,col,a,j,end,m1,m2)
      new=count165(N,raw,ld,rd,col,a,j,end,m1,m2)
    except RuntimeError:
      continue
    if old!=new:
      raise SystemExit(f'DFS mismatch N={N} raw={raw} old={old} new={new}')
    dfs_cases+=1
if dfs_cases<20000:
  raise SystemExit(f'insufficient DFS cases: {dfs_cases}')

print(f'FRAME_MODEL_CASES={frame_cases}')
print(f'SCHEDULE_PACK_CASES={pack_cases}')
print(f'SCHEDULE_CHAIN_CASES={schedule_cases}')
print(f'DFS_MODEL_CASES={dfs_cases}')
PYMODEL
  then
    source_failures=$((source_failures+1))
  else
    # model_metrics.env contains fixed KEY=integer lines from the embedded model.
    # shellcheck disable=SC1090
    source "$MODEL_METRICS"
  fi

  if (( source_failures != 0 )); then
    printf 'source_ctrl_schedule_precompute_shape\trequested layout\t%d check failures\tFAIL\n' "$source_failures" >> "$SUMMARY"
    echo "[error] source does not match the requested 165 ctrl-schedule-precomputation experiment" >&2
    exit 65
  fi
  printf 'source_parent_baseline\tvalidated 164 child-push-init layout\tverified\tOK\n' >> "$SUMMARY"
  printf 'source_ctrl_schedule_precompute_shape\tone-time schedule + action-coded child\tverified\tOK\n' >> "$SUMMARY"
  printf 'frame_action_equivalence\t78120 cases\t%s cases\tOK\n' "$FRAME_MODEL_CASES" >> "$SUMMARY"
  printf 'schedule_word_pack_equivalence\t1032192 cases\t%s cases\tOK\n' "$SCHEDULE_PACK_CASES" >> "$SUMMARY"
  printf 'schedule_chain_replay\t>=20000 cases\t%s cases\tOK\n' "$SCHEDULE_CHAIN_CASES" >> "$SUMMARY"
  printf 'dfs_164_vs_165_model\t>=20000 cases\t%s cases\tOK\n' "$DFS_MODEL_CASES" >> "$SUMMARY"
  rm -f "$KERNEL_SNIP"
fi

need_build=0
if [[ ! -x "$CAND" ]]; then
  need_build=1
elif [[ -f "$SRC" && "$SRC" -nt "$CAND" ]]; then
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
  if (( build_rc != 0 )); then
    printf 'build_exit\t0\t%d\tFAIL\n' "$build_rc" >> "$SUMMARY"
    exit "$build_rc"
  fi
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
  if command -v sha256sum >/dev/null 2>&1 && [[ -f "$SRC" ]]; then
    echo "source_sha256: $(sha256sum "$SRC" | awk '{print $1}')"
  fi
  printf 'command   :'
  printf ' %q' "${CMD[@]}"
  echo
  echo "validation: one full run; cases 01-05 reconstructed from its progress TSV"
  echo "================================================================"
} | tee "$RUN_LOG"

set +e
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
printf 'run_exit\t0\t%d\t%s\n' "$run_rc" "$([[ $run_rc -eq 0 ]] && echo OK || echo FAIL)" >> "$SUMMARY"
if (( run_rc != 0 )); then
  echo "[error] candidate exited with $run_rc" >&2
  exit "$run_rc"
fi

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
    for (i=1; i<=NF; i++) {
      if ($i=="chunk") chunk_col=i
      else if ($i=="chunk_total") total_col=i
      else if ($i=="gpu_total") gpu_col=i
    }
    if (!chunk_col || !total_col || !gpu_col) {
      print "PARSE_OK=0"
      exit 2
    }
    next
  }
  {
    chunk=$(chunk_col)+0
    value=$(total_col)+0
    gpu=$(gpu_col)+0
    rows++
    full+=value
    last_gpu=gpu
    if (seen[chunk]++) duplicates++

    if (chunk==0 || chunk==20 || chunk==40 || chunk==60 || chunk==80 || chunk==100 || chunk==120) p1+=value
    if (chunk==35 || chunk==40 || chunk==41 || chunk==42 || chunk==47 || chunk==48 || chunk==52 || chunk==53) p2+=value
    if (chunk==20 || chunk==40 || chunk==55 || chunk==56 || chunk==57 || chunk==58 || chunk==60) p3+=value
    if (chunk==100 || chunk==105 || chunk==110 || chunk==115 || chunk==120 || chunk==125 || chunk==130) p4+=value
    if ((chunk % 4)==0) p5+=value
  }
  END {
    if (!chunk_col || !total_col || !gpu_col) exit
    missing=0
    for (i=0; i<expected_chunks; i++) if (!(i in seen)) missing++
    printf "PARSE_OK=1\n"
    printf "ROWS=%.0f\n", rows
    printf "DUPLICATES=%.0f\n", duplicates
    printf "MISSING=%.0f\n", missing
    printf "P1=%.0f\n", p1
    printf "P2=%.0f\n", p2
    printf "P3=%.0f\n", p3
    printf "P4=%.0f\n", p4
    printf "P5=%.0f\n", p5
    printf "FULL=%.0f\n", full
    printf "LAST_GPU=%.0f\n", last_gpu
  }
' "$ARCHIVED_PROGRESS" > "$METRICS"

# metrics.env contains only fixed KEY=integer lines emitted by the awk block above.
# shellcheck disable=SC1090
source "$METRICS"

failures=0
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

FINAL_LINE=$(grep -E "^[[:space:]]*${N}:" "$RUN_LOG" | tail -n 1 || true)
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
  ELAPSED_SECONDS=$(awk -F: '{printf "%.3f", ($1*3600)+($2*60)+$3}' <<< "$ELAPSED_TEXT")
  PERF164=$(awk -v base="$BASELINE_164_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF162=$(awk -v base="$BASELINE_162_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF161=$(awk -v base="$BASELINE_161_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF160=$(awk -v base="$BASELINE_160_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF159=$(awk -v base="$BASELINE_159_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF158=$(awk -v base="$BASELINE_158_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF157=$(awk -v base="$BASELINE_157_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF156=$(awk -v base="$BASELINE_156_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF155=$(awk -v base="$BASELINE_155_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF154=$(awk -v base="$BASELINE_154_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF153=$(awk -v base="$BASELINE_153_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF152=$(awk -v base="$BASELINE_152_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF151=$(awk -v base="$BASELINE_151_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF150=$(awk -v base="$BASELINE_150_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF149=$(awk -v base="$BASELINE_149_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  PERF148=$(awk -v base="$BASELINE_148_SECONDS" -v now="$ELAPSED_SECONDS" 'BEGIN {
    delta=base-now
    pct=(base>0)?(delta/base*100):0
    printf "elapsed=%s baseline=%s delta=%+.3f percent=%+.3f%%", now, base, delta, pct
  }')
  printf 'timing_vs_164\t164_full=%ss\t%s\tINFO\n' "$BASELINE_164_SECONDS" "$PERF164" >> "$SUMMARY"
  printf 'timing_vs_162\t162_full=%ss\t%s\tINFO\n' "$BASELINE_162_SECONDS" "$PERF162" >> "$SUMMARY"
  printf 'timing_vs_161\t161_full=%ss\t%s\tINFO\n' "$BASELINE_161_SECONDS" "$PERF161" >> "$SUMMARY"
  printf 'timing_vs_160\t160_full=%ss\t%s\tINFO\n' "$BASELINE_160_SECONDS" "$PERF160" >> "$SUMMARY"
  printf 'timing_vs_159\t159_full=%ss\t%s\tINFO\n' "$BASELINE_159_SECONDS" "$PERF159" >> "$SUMMARY"
  printf 'timing_vs_158\t158_full=%ss\t%s\tINFO\n' "$BASELINE_158_SECONDS" "$PERF158" >> "$SUMMARY"
  printf 'timing_vs_157\t157_full=%ss\t%s\tINFO\n' "$BASELINE_157_SECONDS" "$PERF157" >> "$SUMMARY"
  printf 'timing_vs_156\t156_full=%ss\t%s\tINFO\n' "$BASELINE_156_SECONDS" "$PERF156" >> "$SUMMARY"
  printf 'timing_vs_155\t155_full=%ss\t%s\tINFO\n' "$BASELINE_155_SECONDS" "$PERF155" >> "$SUMMARY"
  printf 'timing_vs_154\t154_full=%ss\t%s\tINFO\n' "$BASELINE_154_SECONDS" "$PERF154" >> "$SUMMARY"
  printf 'timing_vs_153\t153_full=%ss\t%s\tINFO\n' "$BASELINE_153_SECONDS" "$PERF153" >> "$SUMMARY"
  printf 'timing_vs_152\t152_full=%ss\t%s\tINFO\n' "$BASELINE_152_SECONDS" "$PERF152" >> "$SUMMARY"
  printf 'timing_vs_151\t151_full=%ss\t%s\tINFO\n' "$BASELINE_151_SECONDS" "$PERF151" >> "$SUMMARY"
  printf 'timing_vs_150\t150_full=%ss\t%s\tINFO\n' "$BASELINE_150_SECONDS" "$PERF150" >> "$SUMMARY"
  printf 'timing_vs_149\t149_full=%ss\t%s\tINFO\n' "$BASELINE_149_SECONDS" "$PERF149" >> "$SUMMARY"
  printf 'timing_vs_148\t148_mean=%ss\t%s\tINFO\n' "$BASELINE_148_SECONDS" "$PERF148" >> "$SUMMARY"
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

echo "[validation-ok] 165 one full run reproduced cases 01-06"
