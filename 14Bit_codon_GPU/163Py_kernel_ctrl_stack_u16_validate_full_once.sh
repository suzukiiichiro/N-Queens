#!/usr/bin/env bash

set -Eeuo pipefail

# 163 single-pass validation harness
#
# Runs one N=21 full validation.  Cases 01-05 are reconstructed from the same
# 131-row progress TSV; case 06 verifies the full total.  A flock guard prevents
# concurrent GPU runs.
#
# Candidate 163 retains validated 162 DFS/order behavior and changes only the
# per-thread ctrl static stack from u32 to u16.  The initialized flag moves to
# bit15 and the five reachable block metadata tuples are encoded by a 3-bit
# block_code.  Ctrl values are widened to u32 after load; bitboard arithmetic,
# host representation, task order/cache, solution arithmetic, and the inherited
# N=5..27 GPU safety envelope remain unchanged from validated 162.

SRC=${SRC:-./163Py_kernel_ctrl_stack_u16_probe.py}
CAND=${CAND:-./163Py_kernel_ctrl_stack_u16_probe}
AUTO_BUILD=${AUTO_BUILD:-1}
LOCK_FILE=${LOCK_FILE:-/tmp/163Py_kernel_ctrl_stack_u16_validation.lock}
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

# Direct validated 162..149 full runs and historical 148 mean (informational only).
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
    echo "[error] another 163 validation holds: $LOCK_FILE" >&2
    exit 75
  fi
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${LOG_ROOT%/}/163Py_kernel_ctrl_stack_u16_logs_N21_full_once_${TS}"
RUN_LOG="$LOGDIR/full_once.log"
BUILD_LOG="$LOGDIR/build.log"
SUMMARY="$LOGDIR/summary.tsv"
METRICS="$LOGDIR/metrics.env"
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

  # Inherited 160 root-pop loop and 161 guard removal.
  true_loop_count=$(grep -Ec '^[[:space:]]*while[[:space:]]+True:' "$KERNEL_SNIP" || true)
  legacy_loop_count=$(grep -Ec '^[[:space:]]*while[[:space:]]+sp[[:space:]]*>=[[:space:]]*0:' "$KERNEL_SNIP" || true)
  root_guard_count=$(grep -Ec '^[[:space:]]*if[[:space:]]+sp==0:' "$KERNEL_SNIP" || true)
  root_break_count=$(grep -Ec '^[[:space:]]*break[[:space:]]*$' "$KERNEL_SNIP" || true)
  pop_decrement_count=$(grep -Ec '^[[:space:]]*sp-=1[[:space:]]*$' "$KERNEL_SNIP" || true)
  stack_guard_count=$(grep -Ec '^[[:space:]]*if[[:space:]]+sp[[:space:]]*>=[[:space:]]*MAXD:' "$KERNEL_SNIP" || true)
  push_count=$(grep -Ec '^[[:space:]]*sp\+=1[[:space:]]*$' "$KERNEL_SNIP" || true)
  [[ "$true_loop_count" == "1" ]] || source_failures=$((source_failures+1))
  [[ "$legacy_loop_count" == "0" ]] || source_failures=$((source_failures+1))
  [[ "$root_guard_count" == "3" ]] || source_failures=$((source_failures+1))
  [[ "$root_break_count" == "3" ]] || source_failures=$((source_failures+1))
  [[ "$pop_decrement_count" == "3" ]] || source_failures=$((source_failures+1))
  [[ "$stack_guard_count" == "0" ]] || source_failures=$((source_failures+1))
  [[ "$push_count" == "1" ]] || source_failures=$((source_failures+1))
  grep -Fq 'sp:int=0' "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  grep -Fq 'MAXD:Static[int]=21' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'if use_gpu and (nmin<5 or nmax>28):' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq '163 ctrl-stack-u16 GPU candidate supports N=5..27 only' "$SRC" || source_failures=$((source_failures+1))
  grep -Fq 'VERSION_TAG:str="163 ctrl stack u16 over 162 block metadata branch cleanup' "$SRC" || source_failures=$((source_failures+1))

  if ! awk '
    /^[[:space:]]*sp\+=1[[:space:]]*$/ {
      pushes++
      if ((getline nextline) <= 0 || nextline !~ /^[[:space:]]*ctrl\[sp\]=u16\(cv&u32\(1023\)\)/) bad++
    }
    END { exit !((pushes==1) && (bad==0)) }
  ' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi
  if ! awk '
    /^[[:space:]]*sp-=1[[:space:]]*$/ {
      pops++
      if (prev1 !~ /^[[:space:]]*break[[:space:]]*$/ || prev2 !~ /^[[:space:]]*if[[:space:]]+sp==0:/) bad++
    }
    { prev2=prev1; prev1=$0 }
    END { exit !((pops==3) && (bad==0)) }
  ' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi

  # Retain u32 bitboards/scalars and host ctrl0/markctrl input representation.
  for expr in \
    'self.ld_arr:List[u32]' \
    'self.rd_arr:List[u32]' \
    'self.col_arr:List[u32]' \
    'self.free_arr:List[u32]' \
    'self.ctrl0_arr:List[u32]=[u32(0)]*m' \
    'self.markctrl_arr:List[u32]=[u32(0)]*m' \
    'soa.ctrl0_arr[t]=u32(target)|(u32(start)<<u32(5))' \
    'm:int,board_mask:u32' \
    'n3:u32,n4:u32'
  do
    grep -Fq "$expr" "$SRC" || source_failures=$((source_failures+1))
  done
  for expr in \
    'markctrl:u32=markctrl_arr[i]' \
    'jmark:u32=markctrl&u32(31)' \
    'endm:u32=(markctrl>>u32(5))&u32(31)' \
    'mark1:u32=(markctrl>>u32(10))&u32(31)' \
    'mark2:u32=(markctrl>>u32(15))&u32(31)'
  do
    grep -Fq "$expr" "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  done

  # 163 ctrl-stack-u16 shape.  Only ctrl storage is narrowed; every load is
  # explicitly widened before arithmetic and every store is explicitly narrowed.
  u16_ctrl_stack_count=$(grep -Ec '^[[:space:]]*ctrl=__array__\[u16\]\(MAXD\)' "$KERNEL_SNIP" || true)
  u32_ctrl_stack_count=$(grep -Ec '^[[:space:]]*ctrl=__array__\[u32\]\(MAXD\)' "$KERNEL_SNIP" || true)
  u32_stack_count=$(grep -Ec '^[[:space:]]*(ld|rd|col|avail)=__array__\[u32\]\(MAXD\)' "$KERNEL_SNIP" || true)
  ctrl_write_count=$(grep -Ec '^[[:space:]]*ctrl\[[^]]+\]=' "$KERNEL_SNIP" || true)
  ctrl_u16_write_count=$(grep -Ec '^[[:space:]]*ctrl\[[^]]+\]=u16\(' "$KERNEL_SNIP" || true)
  [[ "$u16_ctrl_stack_count" == "1" ]] || source_failures=$((source_failures+1))
  [[ "$u32_ctrl_stack_count" == "0" ]] || source_failures=$((source_failures+1))
  [[ "$u32_stack_count" == "4" ]] || source_failures=$((source_failures+1))
  [[ "$ctrl_write_count" == "3" ]] || source_failures=$((source_failures+1))
  [[ "$ctrl_u16_write_count" == "3" ]] || source_failures=$((source_failures+1))

  for expr in \
    'INIT_MASK:u32=u32(32768)' \
    'ctrl=__array__[u16](MAXD)' \
    'ctrl[0]=u16(ctrl0_arr[i])' \
    'cv:u32=u32(ctrl[sp])' \
    'ctrl[sp]=u16(cv)' \
    'ctrl[sp]=u16(cv&u32(1023))' \
    'frame_blockcodeu:u32=u32(0)' \
    'frame_blockcodeu=(' \
    '((MASK_L_1>>fu)&u32(1))' \
    '|(((STP3_MASK>>fu)&u32(1))<<u32(1))' \
    '|(frame_addu<<u32(2))' \
    '|(frame_blockcodeu<<u32(10))' \
    '|(frame_blocksu<<u32(13))' \
    '|(frame_fcvu<<u32(14))' \
    'blockcodeu:u32=(cv>>u32(10))&u32(7)' \
    'block_l1u:u32=blockcodeu&u32(1)' \
    'step3u:u32=(blockcodeu>>u32(1))&u32(1)' \
    'addvu:u32=(blockcodeu>>u32(2))&u32(1)' \
    'stepu:u32=u32(2)+step3u' \
    'bLiu:u32=block_l1u|((step3u&(u32(1)-block_l1u))<<u32(1))' \
    'n3selu:u32=(u32(1)-block_l1u)&(u32(1)-addvu)' \
    'n4selu:u32=block_l1u&step3u'
  do
    grep -Fq "$expr" "$KERNEL_SNIP" || source_failures=$((source_failures+1))
  done

  if grep -Eq '^[[:space:]]*INIT_MASK:u32=u32\(524288\)' "$KERNEL_SNIP" || \
     grep -Eq '^[[:space:]]*ctrl\[[^]]+\]=(cv|ctrl0_arr|cv&u32)' "$KERNEL_SNIP" || \
     grep -Eq '^[[:space:]]*(frame_bLiu|frame_ktu):u32=' "$KERNEL_SNIP" || \
     grep -Eq '^[[:space:]]*(stepu|addvu|bLiu|ktu):u32=\(cv>>u32\((10|12|15|17)\)\)' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi

  # Retain validated 162 branch cleanup: metadata is created in the mark branch,
  # with no late frame_blocksu test and no conditional K-mask branch.
  frame_blocks_test_count=$(grep -Ec '^[[:space:]]*if[[:space:]]+frame_blocksu' "$KERNEL_SNIP" || true)
  kn3_branch_count=$(grep -Ec '^[[:space:]]*if[[:space:]]+\(\(MASK_K_N3>>fu\)' "$KERNEL_SNIP" || true)
  kn4_branch_count=$(grep -Ec '^[[:space:]]*if[[:space:]]+\(\(MASK_K_N4>>fu\)' "$KERNEL_SNIP" || true)
  blockcode_assign_count=$(grep -Ec '^[[:space:]]*frame_blockcodeu=\([[:space:]]*$' "$KERNEL_SNIP" || true)
  [[ "$frame_blocks_test_count" == "0" ]] || source_failures=$((source_failures+1))
  [[ "$kn3_branch_count" == "0" ]] || source_failures=$((source_failures+1))
  [[ "$kn4_branch_count" == "0" ]] || source_failures=$((source_failures+1))
  [[ "$blockcode_assign_count" == "1" ]] || source_failures=$((source_failures+1))
  if ! awk '
    /^[[:space:]]*if rowv==markv:/ { marks++; in_mark=1 }
    in_mark && /^[[:space:]]*frame_blockcodeu=\(/ { codes++ }
    in_mark && /^[[:space:]]*nextfidu=u32\(meta_next\[int\(fu\)\]\)/ { closes++; in_mark=0 }
    END { exit !((marks==1) && (codes==1) && (closes==1) && (in_mark==0)) }
  ' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi

  # Kernel-local u32 classification masks remain typed and unwrapped.
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
  if grep -Eq '^[[:space:]]*(IS_BASE_MASK|IS_JMARK_MASK|IS_MARK_MASK|IS_P5_MASK|SEL2_MASK|STP3_MASK|MASK_K_N3|MASK_K_N4|MASK_L_1|MASK_L_2|INIT_MASK):Static\[int\]' "$KERNEL_SNIP"; then
    source_failures=$((source_failures+1))
  fi

  meta_next_u32_refs=$(grep -Fc 'u32(meta_next[int(fu)])' "$KERNEL_SNIP" || true)
  int_fu_refs=$(grep -Ec '^[[:space:]]*[^#].*int\(fu\)' "$KERNEL_SNIP" || true)
  [[ "$meta_next_u32_refs" == "3" ]] || source_failures=$((source_failures+1))
  [[ "$int_fu_refs" == "3" ]] || source_failures=$((source_failures+1))

  # Verify the ten inherited classification masks for every valid fid.
  if ! classification_cases=$(awk 'BEGIN {
      masks[1]=69222408; masks[2]=4; masks[3]=199209203; masks[4]=3840; masks[5]=34742338
      masks[6]=21266576; masks[7]=185471169; masks[8]=4227088; masks[9]=12689458; masks[10]=17039488
      nxt[0]=1; nxt[1]=2; nxt[2]=3; nxt[3]=3; nxt[4]=2; nxt[5]=6; nxt[6]=2; nxt[7]=2
      nxt[8]=0; nxt[9]=4; nxt[10]=5; nxt[11]=7; nxt[12]=13; nxt[13]=14; nxt[14]=14; nxt[15]=14
      nxt[16]=17; nxt[17]=14; nxt[18]=14; nxt[19]=20; nxt[20]=21; nxt[21]=21; nxt[22]=21; nxt[23]=25
      nxt[24]=21; nxt[25]=21; nxt[26]=26; nxt[27]=26
      cases=0; mod32=4294967296
      for (f=0; f<28; f++) {
        p=2^f
        for (i=1; i<=10; i++) {
          oldv=int(masks[i]/p)%2; newv=int((masks[i]%mod32)/p)%2
          if (oldv!=newv) exit 1
          cases++
        }
        mark=int(masks[3]/p)%2
        if ((1-mark)!=((1-mark+mod32)%mod32)) exit 1
        cases++
        oldbl=(int(masks[9]/p)%2)+2*(int(masks[10]/p)%2)
        newbl=(int((masks[9]%mod32)/p)%2)+2*(int((masks[10]%mod32)/p)%2)
        if (oldbl!=newbl) exit 1
        cases++
        n3bit=int(masks[7]/p)%2; n4bit=int(masks[8]/p)%2
        if (n3bit==1 && n4bit==1) exit 1
        oldkt=n3bit+2*n4bit
        newkt=(int((masks[7]%mod32)/p)%2)+2*(int((masks[8]%mod32)/p)%2)
        if (oldkt!=newkt) exit 1
        cases++
        if (nxt[f]<0 || nxt[f]>=28 || (nxt[f]%mod32)!=nxt[f]) exit 1
        cases++
      }
      print cases
    }'); then
    source_failures=$((source_failures+1)); classification_cases=0
  fi

  # Prove that the compact code reconstructs every old block metadata tuple.
  if ! block_code_cases=$(awk 'BEGIN {
      markmask=199209203; stp3mask=21266576; n3mask=185471169; n4mask=4227088
      l1mask=12689458; l2mask=17039488; cases=0
      for (f=0; f<28; f++) {
        p=2^f
        ismark=int(markmask/p)%2
        stp3=int(stp3mask/p)%2
        add=(f==20)?1:0
        oldstep=2+stp3
        oldL=(int(l1mask/p)%2)+2*(int(l2mask/p)%2)
        oldK=(int(n3mask/p)%2)+2*(int(n4mask/p)%2)
        for (hit=0; hit<=1; hit++) {
          blocks=(ismark && hit)?1:0
          code=blocks?((int(l1mask/p)%2)+2*stp3+4*add):0
          b0=code%2; b1=int(code/2)%2; b2=int(code/4)%2
          newstep=2+b1
          newadd=b2
          newL=b0+2*(b1*(1-b0))
          newK=((1-b0)*(1-b2))+2*(b0*b1)
          if (blocks) {
            if (code<0 || code>4 || oldstep!=newstep || add!=newadd || oldL!=newL || oldK!=newK) exit 1
          } else if (code!=0) exit 1
          cases++
        }
      }
      print cases
    }'); then
    source_failures=$((source_failures+1)); block_code_cases=0
  fi

  # Exhaustively verify every reachable 16-bit ctrl pack and its decode.
  if ! ctrl_u16_cases=$(awk 'BEGIN {
      cases=0
      for (fid=0; fid<28; fid++)
      for (row=0; row<32; row++) {
        raw=fid+row*32
        if (raw<0 || raw>=1024 || (raw%32)!=fid || (int(raw/32)%32)!=row) exit 1
        cases++
        for (future=0; future<=1; future++) {
          pack=32768+fid+row*32+future*16384
          if (pack<0 || pack>=65536) exit 1
          if ((pack%32)!=fid || (int(pack/32)%32)!=row) exit 1
          if ((int(pack/8192)%2)!=0 || (int(pack/16384)%2)!=future || (int(pack/32768)%2)!=1) exit 1
          cases++
        }
        for (code=0; code<=4; code++) {
          pack=32768+fid+row*32+code*1024+8192
          if (pack<0 || pack>=65536) exit 1
          if ((pack%32)!=fid || (int(pack/32)%32)!=row || (int(pack/1024)%8)!=code) exit 1
          if ((int(pack/8192)%2)!=1 || (int(pack/16384)%2)!=0 || (int(pack/32768)%2)!=1) exit 1
          b0=code%2; b1=int(code/2)%2; b2=int(code/4)%2
          step=2+b1; add=b2; blockL=b0+2*(b1*(1-b0)); blockK=((1-b0)*(1-b2))+2*(b0*b1)
          if (code==0 && !(step==2 && add==0 && blockL==0 && blockK==1)) exit 1
          if (code==1 && !(step==2 && add==0 && blockL==1 && blockK==0)) exit 1
          if (code==2 && !(step==3 && add==0 && blockL==2 && blockK==1)) exit 1
          if (code==3 && !(step==3 && add==0 && blockL==1 && blockK==2)) exit 1
          if (code==4 && !(step==2 && add==1 && blockL==0 && blockK==0)) exit 1
          cases++
        }
      }
      print cases
    }'); then
    source_failures=$((source_failures+1)); ctrl_u16_cases=0
  fi

  if ! stack_bytes=$(awk 'BEGIN { maxd=21; bytes=4*maxd*4 + 2*maxd; if (bytes!=378) exit 1; print bytes }'); then
    source_failures=$((source_failures+1)); stack_bytes=0
  fi

  # Retain packed mark/end sentinels and row/step arithmetic checks.
  if ! markctrl_cases=$(awk 'BEGIN {
      cases=0
      for (j=0; j<=30; j++) for (e=0; e<=30; e++) for (m1=-1; m1<=30; m1++) for (m2=-1; m2<=30; m2++) {
        p1=(m1<0)?31:m1; p2=(m2<0)?31:m2
        pack=j+e*32+p1*1024+p2*32768
        if ((pack%32)!=j || (int(pack/32)%32)!=e || (int(pack/1024)%32)!=p1 || (int(pack/32768)%32)!=p2) exit 1
        if (pack<0 || pack>=1048576) exit 1
        cases++
      }
      print cases
    }'); then
    source_failures=$((source_failures+1)); markctrl_cases=0
  fi
  if ! row_cases=$(awk 'BEGIN {
      cases=0
      for (row=0; row<=30; row++) for (mark=-1; mark<=30; mark++) {
        packed=(mark<0)?31:mark
        if ((row==mark)!=(row==packed)) exit 1
        cases++
      }
      for (row=0; row<=27; row++) for (step=1; step<=3; step++) for (end=0; end<=30; end++) {
        child=row+step
        if (child<0 || child>30 || ((row+step<end)!=(child<end))) exit 1
        cases++
      }
      print cases
    }'); then
    source_failures=$((source_failures+1)); row_cases=0
  fi

  if (( source_failures != 0 )); then
    printf 'source_ctrl_stack_u16_shape\trequested layout\t%d check failures\tFAIL\n' "$source_failures" >> "$SUMMARY"
    echo "[error] source does not match the requested 163 ctrl-stack-u16 experiment" >&2
    exit 65
  fi
  printf 'source_inherited_root_pop_shape\tvalidated 160 layout\tverified\tOK\n' >> "$SUMMARY"
  printf 'source_inherited_no_stack_guard_shape\tvalidated 161 layout\tverified\tOK\n' >> "$SUMMARY"
  printf 'source_inherited_block_cleanup_shape\tvalidated 162 layout\tverified\tOK\n' >> "$SUMMARY"
  printf 'source_ctrl_stack_u16_shape\tu16 ctrl + u32 load arithmetic\tverified\tOK\n' >> "$SUMMARY"
  printf 'static_stack_bytes_per_thread\t378 bytes\t%s bytes\tOK\n' "$stack_bytes" >> "$SUMMARY"
  printf 'classification_mask_u32_equivalence\t392 cases\t%s cases\tOK\n' "$classification_cases" >> "$SUMMARY"
  printf 'block_code_equivalence\t56 cases\t%s cases\tOK\n' "$block_code_cases" >> "$SUMMARY"
  printf 'ctrl_u16_pack_decode_equivalence\t7168 cases\t%s cases\tOK\n' "$ctrl_u16_cases" >> "$SUMMARY"
  printf 'markctrl_pack_equivalence\t984064 cases\t%s cases\tOK\n' "$markctrl_cases" >> "$SUMMARY"
  printf 'row_step_u32_equivalence\t3596 cases\t%s cases\tOK\n' "$row_cases" >> "$SUMMARY"
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

echo "[validation-ok] 163 one full run reproduced cases 01-06"
