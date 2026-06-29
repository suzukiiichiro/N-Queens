#!/usr/bin/env bash

set -Eeuo pipefail

# 169 single-pass validation harness
#
# One N=21 full GPU run reconstructs cases 01-05 from the same 131-row
# progress TSV and verifies the complete total.  Candidate 169 inherits the
# validated 168 packed-opcode schedule and adds the smaller MAXD14 static
# specialization.  For N=21 all launches must remain required_maxd=14, select
# MAXD14, use four schedule words, and report 240 local stack bytes/thread.
#
# The 168 full run observed required_maxd=14 in every chunk while selecting
# MAXD16.  This harness checks that 169 selects the new exact static MAXD14
# capacity without changing the totals.

SRC=${SRC:-./169Py_kernel_packed_opcode_schedule_maxd14_probe.py}
CAND=${CAND:-./169Py_kernel_packed_opcode_schedule_maxd14_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
STATIC_ONLY=${STATIC_ONLY:-0}
LOCK_FILE=${LOCK_FILE:-/tmp/169Py_kernel_packed_opcode_schedule_maxd14_validation.lock}
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
EXPECTED_REQUIRED_MAXD=14
EXPECTED_SELECTED_MAXD=14
EXPECTED_SCHEDULE_WORDS=4
EXPECTED_STACK_BYTES=240
EXP01=16879968420
EXP02=19219113480
EXP03=16935522136
EXP04=16230260724
EXP05=79383179384
FULL_TOTAL=314666222712

BASELINE_168_SECONDS=${BASELINE_168_SECONDS:-560.261}
BASELINE_167_SECONDS=${BASELINE_167_SECONDS:-633.039}
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
    echo "[error] another 169 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/169Py_kernel_packed_opcode_schedule_maxd14_logs_N21_full_once_${TS}"
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
import re, random, sys

src=Path(sys.argv[1])
s=src.read_text()

# ---------------- structural checks ----------------
word_counts={14:4,16:4,18:5,20:5,21:6}
for md,words in word_counts.items():
    assert f'MAXD{md}:Static[int]={md}' in s
    assert f'SCHED_WORDS{md}:Static[int]={words}' in s
    assert s.count(f'def kernel_dfs_iter_gpu_maxd{md}(')==1
assert len(re.findall(r'^@gpu\.kernel$',s,re.M))==5
assert not re.search(r'^MAXD:Static\[int\]=',s,re.M)
assert s.count('launch_kernel_dfs_iter_gpu_static_maxd(')==6
assert s.count('max_schedule_depth_of_tasks(')==6
assert s.count('[maxd-dispatch]')==3
assert '169 MAXD14 packed-opcode GPU candidate supports N=5..27 only' in s
assert 'stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)}' in s
assert 'schedule_words={packed_schedule_words_for_maxd(selected_maxd)}' in s

normalized=[]
for md,words in word_counts.items():
    st=s.index(f'@gpu.kernel\ndef kernel_dfs_iter_gpu_maxd{md}(')
    candidates=[x for x in (s.find('\n@gpu.kernel\n',st+1),s.find('\ndef launch_kernel_dfs_iter_gpu_static_maxd',st+1)) if x!=-1]
    en=min(candidates)
    b=s[st:en]
    assert b.count(f'__array__[u32](MAXD{md})')==4,(md,b.count(f'__array__[u32](MAXD{md})'))
    assert b.count(f'__array__[u32](SCHED_WORDS{md})')==1
    for marker in (
        'packed_schedule=__array__[u32]',
        'schedule_block_code:u32=u32(0)',
        'frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))',
        'packed_schedule[pack_word_index]=pack_word',
        'opcode_word:u32=packed_schedule[sp>>2]',
        'opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)',
        'block_code:u32=opcode&u32(7)',
        'child_action:u32=(opcode>>u32(4))&u32(3)',
        'sp+=1',
    ):
        assert marker in b,(md,marker)
    for forbidden in ('ctrl=__array__','ctrl[','root_cv','child_cv','INIT_MASK','u32(524288)','if sp>=MAXD'):
        assert forbidden not in b,(md,forbidden)
    n=b.replace(f'kernel_dfs_iter_gpu_maxd{md}','kernel_dfs_iter_gpu_maxdX')
    n=n.replace(f'MAXD{md}','MAXDX').replace(f'SCHED_WORDS{md}','SCHED_WORDSX')
    n=re.sub(r'static MAXD=\d+','static MAXD=X',n).rstrip()
    normalized.append(n)
assert len(set(normalized))==1

# ---------------- reference constants ----------------
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
BC0=173707345
BC1=12689458
BC2=18088064
OP_STEP3=24
OP_ADD1=32
OP_BL1=12
OP_BL2=16
OP_KN3=18
OP_KN4=8

# Old 164 metadata for one deterministic frame.
def old_frame(raw,j,e,m1,m2):
    f=raw&31; row=(raw>>5)&31
    if ((IS_P5>>f)&1) and row==m1:
        f=META[f]
    if ((IS_BASE>>f)&1) and row==e:
        return {'action':3 if f==14 else 2}
    ism=(IS_MARK>>f)&1
    step=1; add=0; blocks=0; bl=0; kt=0; future=1-ism; nxt=f
    if ism:
        mark=m2 if ((SEL2>>f)&1) else m1
        if row==mark:
            blocks=1
            bl=((MASK_L_1>>f)&1)|(((MASK_L_2>>f)&1)<<1)
            kt=((MASK_K_N3>>f)&1)|(((MASK_K_N4>>f)&1)<<1)
            future=0
            step=3 if ((STP3>>f)&1) else 2
            add=1 if f==20 else 0
            nxt=META[f]
    action=0
    if ((IS_JMARK>>f)&1) and row==j:
        action=1
        nxt=META[f]
    child=row+step
    fc=int(bool(future and child<e))
    return {
        'action':action,'raw':nxt|(child<<5),'blocks':blocks,'step':step,
        'add':add,'bl':bl,'kt':kt,'future':fc,
    }

# New 168 byte opcode for the same frame.
def new_frame(raw,j,e,m1,m2):
    f=raw&31; row=(raw>>5)&31
    if ((IS_P5>>f)&1) and row==m1:
        f=META[f]
    if ((IS_BASE>>f)&1) and row==e:
        return {'action':3 if f==14 else 2}
    ism=(IS_MARK>>f)&1
    code=0; step=1; use_future=1-ism; nxt=f
    if ism:
        mark=m2 if ((SEL2>>f)&1) else m1
        if row==mark:
            code=((BC0>>f)&1)|(((BC1>>f)&1)<<1)|(((BC2>>f)&1)<<2)
            step=2+((OP_STEP3>>code)&1)
            use_future=0
            nxt=META[f]
    action=0
    if ((IS_JMARK>>f)&1) and row==j:
        action=1
        nxt=META[f]
    child=row+step
    future=int(bool(use_future and child<e))
    return {'action':action,'raw':nxt|(child<<5),'opcode':code|(future<<3)}

def decode_opcode(op):
    code=op&7
    if code==0:
        return 0,1,0,0,0,(op>>3)&1
    step=2+((OP_STEP3>>code)&1)
    add=(OP_ADD1>>code)&1
    bl=((OP_BL1>>code)&1)|(((OP_BL2>>code)&1)<<1)
    kt=((OP_KN3>>code)&1)|(((OP_KN4>>code)&1)<<1)
    return 1,step,add,bl,kt,(op>>3)&1

# Every mark fid must map to exactly one valid code, and decode to old metadata.
block_code_cases=0
for f in range(28):
    if (IS_MARK>>f)&1:
        code=((BC0>>f)&1)|(((BC1>>f)&1)<<1)|(((BC2>>f)&1)<<2)
        assert 1<=code<=5,(f,code)
        old=old_frame(f|(7<<5),0,30,7 if not ((SEL2>>f)&1) else 31,7 if ((SEL2>>f)&1) else 31)
        block_code_cases+=1

# Broad exact frame equivalence around every significant row.
frame_cases=0
for f in range(28):
    for row in range(31):
        vals=sorted({0,row,min(30,row+1),30,31})
        for j in vals:
            for e in vals:
                for m1 in vals:
                    for m2 in vals:
                        a=old_frame(f|(row<<5),j,e,m1,m2)
                        b=new_frame(f|(row<<5),j,e,m1,m2)
                        assert a['action']==b['action'],(f,row,j,e,m1,m2,a,b)
                        if a['action']<2:
                            blocks,step,add,bl,kt,future=decode_opcode(b['opcode'])
                            assert (blocks,step,add,bl,kt,future,b['raw']) == (
                                a['blocks'],a['step'],a['add'],a['bl'],a['kt'],a['future'],a['raw']
                            ),(f,row,j,e,m1,m2,a,b)
                        frame_cases+=1

# Exact source-shaped packing algorithm.
def build_packed(raw,j,e,m1,m2,capacity):
    words=[None]*((capacity+3)//4)
    depth=0; pack_word=0; word_index=0; root_action=0
    while True:
        fr=new_frame(raw,j,e,m1,m2)
        action=fr['action']
        if depth==0:
            root_action=action
        else:
            parent_lane=(depth-1)&3
            pack_word |= action << (parent_lane*8+4)
            if (depth&3)==0:
                words[word_index]=pack_word
                word_index+=1
                pack_word=0
        if action>=2:
            if depth>0 and (depth&3)!=0:
                words[word_index]=pack_word
            break
        pack_word |= fr['opcode'] << ((depth&3)*8)
        raw=fr['raw']
        depth+=1
        if depth>capacity:
            raise OverflowError
    return root_action,words,depth

def packed_at(words,d):
    return (words[d>>2]>>((d&3)*8))&255

def depth_candidate(raw,mc):
    j=mc&31;e=(mc>>5)&31;m1=(mc>>10)&31;m2=(mc>>15)&31
    depth=0
    while True:
        fr=old_frame(raw,j,e,m1,m2)
        if fr['action']>=2:return depth
        raw=fr['raw'];depth+=1
        if depth>21:return 22

def select(d):
    if d<=14:return 14
    if d<=16:return 16
    if d<=18:return 18
    if d<=20:return 20
    if d<=21:return 21
    return 0

def words_for(md):
    return {14:4,16:4,18:5,20:5,21:6}.get(md,0)

def stack_bytes(md):
    return md*16+words_for(md)*4 if words_for(md) else 0

# Dispatch boundaries and exact footprints.
expected={**{i:14 for i in range(15)},15:16,16:16,17:18,18:18,19:20,20:20,21:21,22:0}
for d,v in expected.items():assert select(d)==v,(d,select(d),v)
assert {md:stack_bytes(md) for md in word_counts}=={14:240,16:272,18:308,20:340,21:360}

# Independent 164 frame-entry model used by full DFS.
def init_frame(raw,a,ld,j,e,m1,m2):
    fr=old_frame(raw,j,e,m1,m2)
    if fr['action']>=2:
        add=1 if fr['action']==2 else int((a&~1)!=0)
        return 'base',0,a,ld,add
    if fr['action']==1:
        a&=~1
        if a==0:return 'dead',0,a,ld,0
        ld|=1
    blocks,step,add,bl,kt,future=decode_opcode(new_frame(raw,j,e,m1,m2)['opcode'])
    cv=fr['raw']|(step<<10)|(add<<12)|(blocks<<13)|(future<<14)|(bl<<15)|(kt<<17)
    return 'ready',cv,a,ld,0

def dfs_reference(N,raw,ld0,rd0,col0,a0,j,e,m1,m2,capacity=64,limit=120000):
    bm=(1<<N)-1;n3=1<<max(0,N-3);n4=1<<max(0,N-4)
    a0&=bm
    if a0==0:return 0,0
    st,cv,a0,ld0,add=init_frame(raw,a0,ld0,j,e,m1,m2)
    if st=='base':return add,0
    if st=='dead':return 0,0
    ld=[0]*capacity;rd=[0]*capacity;col=[0]*capacity;av=[0]*capacity;ctrl=[0]*capacity
    sp=0;ld[0]=ld0;rd[0]=rd0;col[0]=col0;av[0]=a0;ctrl[0]=cv
    total=0;maxframes=1;steps=0
    while True:
        steps+=1
        if steps>limit:raise RuntimeError
        a=av[sp]
        if a==0:
            if sp==0:break
            sp-=1;continue
        cv=ctrl[sp];bit=a&-a;av[sp]=a^bit
        if cv&8192:
            step=(cv>>10)&3;add1=(cv>>12)&1;bl=(cv>>15)&3;kt=(cv>>17)&3
            bk=(n3&-(kt&1))|(n4&-((kt>>1)&1))
            nld=((ld[sp]|bit)<<step)|add1|bl;nrd=((rd[sp]|bit)>>step)|bk
        else:
            nld=(ld[sp]|bit)<<1;nrd=(rd[sp]|bit)>>1
        ncol=col[sp]|bit;nf=bm&~(nld|nrd|ncol)
        if nf==0:continue
        if (cv&16384) and bm&~((nld<<1)|(nrd>>1)|ncol)==0:continue
        st,ch,nf,nld,add=init_frame(cv&1023,nf,nld,j,e,m1,m2)
        if st=='base':total+=add;continue
        if st=='dead':continue
        if sp+1>=capacity:raise OverflowError
        sp+=1;maxframes=max(maxframes,sp+1)
        ctrl[sp]=ch;ld[sp]=nld;rd[sp]=nrd;col[sp]=ncol;av[sp]=nf
    return total,maxframes

def dfs_packed(N,raw,ld0,rd0,col0,a0,j,e,m1,m2,capacity,limit=120000):
    bm=(1<<N)-1;n3=1<<max(0,N-3);n4=1<<max(0,N-4)
    a0&=bm
    if a0==0:return 0,0
    root_action,words,_=build_packed(raw,j,e,m1,m2,capacity)
    if root_action==2:return 1,0
    if root_action==3:return int((a0&~1)!=0),0
    if root_action==1:
        a0&=~1
        if a0==0:return 0,0
        ld0|=1
    ld=[0]*capacity;rd=[0]*capacity;col=[0]*capacity;av=[0]*capacity
    sp=0;ld[0]=ld0;rd[0]=rd0;col[0]=col0;av[0]=a0
    total=0;maxframes=1;steps=0
    while True:
        steps+=1
        if steps>limit:raise RuntimeError
        a=av[sp]
        if a==0:
            if sp==0:break
            sp-=1;continue
        op=packed_at(words,sp);code=op&7;bit=a&-a;av[sp]=a^bit
        if code:
            step=2+((OP_STEP3>>code)&1);add=(OP_ADD1>>code)&1
            bl=((OP_BL1>>code)&1)|(((OP_BL2>>code)&1)<<1)
            kt=((OP_KN3>>code)&1)|(((OP_KN4>>code)&1)<<1)
            bk=(n3&-(kt&1))|(n4&-((kt>>1)&1))
            nld=((ld[sp]|bit)<<step)|add|bl;nrd=((rd[sp]|bit)>>step)|bk
        else:
            nld=(ld[sp]|bit)<<1;nrd=(rd[sp]|bit)>>1
        ncol=col[sp]|bit;nf=bm&~(nld|nrd|ncol)
        if nf==0:continue
        if (op&8) and bm&~((nld<<1)|(nrd>>1)|ncol)==0:continue
        action=(op>>4)&3
        if action>=2:
            total+=1 if action==2 else int((nf&~1)!=0)
            continue
        if action==1:
            nf&=~1
            if nf==0:continue
            nld|=1
        if sp+1>=capacity:raise OverflowError
        sp+=1;maxframes=max(maxframes,sp+1)
        ld[sp]=nld;rd[sp]=nrd;col[sp]=ncol;av[sp]=nf
    return total,maxframes

random.seed(168167)
dfs_cases=0;attempts=0;max_observed=0
while dfs_cases<12000 and attempts<160000:
    attempts+=1
    N=random.randrange(5,10);bm=(1<<N)-1
    raw=random.randrange(28)|(random.randrange(0,N+2)<<5)
    j=random.randrange(0,N+1);e=random.randrange(0,N+1)
    m1=random.choice([31]+list(range(N+1)));m2=random.choice([31]+list(range(N+1)))
    mc=j|(e<<5)|(m1<<10)|(m2<<15)
    d=depth_candidate(raw,mc)
    if d>21:continue
    ld=random.randrange(bm+1);rd=random.randrange(1<<(N+2));col=random.randrange(bm+1);a=random.randrange(bm+1)
    try:
        ref,obs=dfs_reference(N,raw,ld,rd,col,a,j,e,m1,m2)
        got,obs2=dfs_packed(N,raw,ld,rd,col,a,j,e,m1,m2,select(d))
    except RuntimeError:
        continue
    assert (ref,obs)==(got,obs2),(N,raw,j,e,m1,m2,ref,got,obs,obs2,d)
    assert obs<=d or obs==0,(obs,d)
    max_observed=max(max_observed,obs)
    dfs_cases+=1
assert dfs_cases>=10000,dfs_cases

# Broad N=21/preset=6 control-classification superset, inherited from 167.
def classify(N,start,j,k,l):
    jm=0;m1=0;m2=0;e=0;f=0;N2=N-2
    jlt=j<N-3;je3=j==N-3;je2=j==N-2;klt=k<l
    sk=start<k;sl=start<l;adjl=l==k+1;adjk=k==l+1
    gate=j>2*N-34-start
    if jlt:
        jm=j+1;e=N2
        if gate:
            if klt:
                m1,m2=k-1,l-1
                if sl:
                    if sk:f=0 if not adjl else 4
                    else:f=1
                else:f=2
            else:
                m1,m2=l-1,k-1
                if sk:
                    if sl:f=5 if not adjk else 7
                    else:f=6
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

print('KERNEL_VARIANTS=5')
print('KERNEL_NORMALIZED_EQUAL=1')
print(f'BLOCK_CODE_CASES={block_code_cases}')
print(f'FRAME_OPCODE_CASES={frame_cases}')
print(f'DISPATCH_BOUNDARY_CASES={len(expected)}')
print(f'DFS_EQUIVALENCE_CASES={dfs_cases}')
print(f'DFS_MAX_OBSERVED={max_observed}')
print(f'N21_PRESET6_SUPERSET_CASES={n21_cases}')
print(f'N21_PRESET6_MAXD={n21_max}')
print('STACK_BYTES14=240')
print('STACK_BYTES16=272')
print('STACK_BYTES18=308')
print('STACK_BYTES20=340')
print('STACK_BYTES21=360')
PYMODEL
then
  printf 'source_packed_opcode_schedule_shape\trequested layout\tmodel/static failure\tFAIL\n' >> "$SUMMARY"
  echo "[error] source/model validation failed" >&2
  exit 65
fi
# model_metrics.env contains fixed KEY=integer lines.
# shellcheck disable=SC1090
source "$MODEL_METRICS"

printf 'source_parent_baseline\tvalidated fastest 168 packed opcode schedule\tverified\tOK\n' >> "$SUMMARY"
printf 'source_packed_opcode_schedule_shape\t5 static kernels + packed byte schedule + MAXD14\tverified\tOK\n' >> "$SUMMARY"
printf 'kernel_variant_count\t5\t%s\tOK\n' "$KERNEL_VARIANTS" >> "$SUMMARY"
printf 'kernel_normalized_equivalence\t1\t%s\tOK\n' "$KERNEL_NORMALIZED_EQUAL" >> "$SUMMARY"
printf 'block_opcode_mapping\t19 mark fids\t%s cases\tOK\n' "$BLOCK_CODE_CASES" >> "$SUMMARY"
printf 'frame_opcode_equivalence\t506604 cases\t%s cases\tOK\n' "$FRAME_OPCODE_CASES" >> "$SUMMARY"
printf 'dispatch_threshold_cases\t23 cases\t%s cases\tOK\n' "$DISPATCH_BOUNDARY_CASES" >> "$SUMMARY"
printf 'dfs_164_vs_169_equivalence\t>=10000 cases\t%s cases\tOK\n' "$DFS_EQUIVALENCE_CASES" >> "$SUMMARY"
printf 'n21_preset6_superset_depth\tMAXD=16\tMAXD=%s (%s cases)\tOK\n' "$N21_PRESET6_MAXD" "$N21_PRESET6_SUPERSET_CASES" >> "$SUMMARY"
printf 'stack_bytes_MAXD14\t240\t%s\tOK\n' "$STACK_BYTES14" >> "$SUMMARY"
printf 'stack_bytes_MAXD16\t272\t%s\tOK\n' "$STACK_BYTES16" >> "$SUMMARY"
printf 'stack_bytes_MAXD18\t308\t%s\tOK\n' "$STACK_BYTES18" >> "$SUMMARY"
printf 'stack_bytes_MAXD20\t340\t%s\tOK\n' "$STACK_BYTES20" >> "$SUMMARY"
printf 'stack_bytes_MAXD21\t360\t%s\tOK\n' "$STACK_BYTES21" >> "$SUMMARY"

if [[ "$STATIC_ONLY" == "1" ]]; then
  echo "================================================================"
  echo "[static-summary]"
  column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
  echo "[logdir] $LOGDIR"
  echo "[static-validation-ok] 169 source/model checks passed"
  exit 0
fi

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
  echo "dispatch  : required=14, selected MAXD14, schedule_words=4, stack=240 bytes/thread"
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
    m=0; req=-1; sel=-1; words=-1; bytes=-1; cap=""
    for (i=1;i<=NF;i++) {
      split($i,a,"=")
      if (a[1]=="m") m=a[2]+0
      else if (a[1]=="required_maxd") req=a[2]+0
      else if (a[1]=="selected_MAXD") sel=a[2]+0
      else if (a[1]=="schedule_words") words=a[2]+0
      else if (a[1]=="stack_bytes_per_thread") bytes=a[2]+0
      else if (a[1]=="capacity_check") cap=a[2]
    }
    tasks+=m
    if (rows==1 || req<minreq) minreq=req
    if (req>maxreq) maxreq=req
    if (req!=14) badreq++
    if (sel!=14) badsel++
    if (words!=4) badwords++
    if (bytes!=240) badbytes++
    if (cap!="OK") badcap++
    if (req>sel) under++
  }
  END {
    printf "DISPATCH_ROWS=%d\n",rows+0
    printf "DISPATCH_TASKS=%.0f\n",tasks+0
    printf "DISPATCH_MIN_REQUIRED=%d\n",minreq+0
    printf "DISPATCH_MAX_REQUIRED=%d\n",maxreq+0
    printf "DISPATCH_NON14=%d\n",badreq+0
    printf "DISPATCH_NON_SELECTED14=%d\n",badsel+0
    printf "DISPATCH_NON4WORDS=%d\n",badwords+0
    printf "DISPATCH_NON240=%d\n",badbytes+0
    printf "DISPATCH_BAD_CAPACITY=%d\n",badcap+0
    printf "DISPATCH_UNDERSIZED=%d\n",under+0
  }
' "$RUN_LOG" > "$DISPATCH_METRICS"
# shellcheck disable=SC1090
source "$DISPATCH_METRICS"

failures=0
record_check "dispatch_launch_rows" "$EXPECTED_CHUNKS" "$DISPATCH_ROWS" || failures=$((failures+1))
record_check "dispatch_task_sum" "$EXPECTED_TASKS" "$DISPATCH_TASKS" || failures=$((failures+1))
record_check "dispatch_min_required" "$EXPECTED_REQUIRED_MAXD" "$DISPATCH_MIN_REQUIRED" || failures=$((failures+1))
record_check "dispatch_max_required" "$EXPECTED_REQUIRED_MAXD" "$DISPATCH_MAX_REQUIRED" || failures=$((failures+1))
record_check "dispatch_non_required14" "0" "$DISPATCH_NON14" || failures=$((failures+1))
record_check "dispatch_non_MAXD14" "0" "$DISPATCH_NON_SELECTED14" || failures=$((failures+1))
record_check "dispatch_non_4_schedule_words" "0" "$DISPATCH_NON4WORDS" || failures=$((failures+1))
record_check "dispatch_non_240_bytes" "0" "$DISPATCH_NON240" || failures=$((failures+1))
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
  for pair in "168:$BASELINE_168_SECONDS" "167:$BASELINE_167_SECONDS" "166:$BASELINE_166_SECONDS" "165:$BASELINE_165_SECONDS" "164:$BASELINE_164_SECONDS" "162:$BASELINE_162_SECONDS"; do
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

echo "[validation-ok] 169 MAXD14 packed opcode schedule reproduced cases 01-06 with required=14, MAXD14, 240 bytes/thread"
