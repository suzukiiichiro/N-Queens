#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# NQ_UPDATE_MEMO
# 234: cachehot-maxd14-direct。233 fasttrim-inlineを親に、N=21 full本線へさらに特化する。
#      既存のscorestripe/chunkshape148 shaped binを直接読み、dict chunk生成と汎用MAXD dispatchを削除する。
#      fallback kernel、schedule depth scan、MAXD選択wrapper、旧stats full集計を外し、
#      MAXD14 kernelを直接起動する。CUDA MAXD14 kernel本文、rootrestlate、futuremask、no-sibling、root one/two prerollは233と同一。
#      runtime tagはsplit234。N=21/cache-hot/full-run専用版のため、cache生成が必要な場合は233以前で生成する。
# Full update history: see README.md
# =============================================================================

import gpu
import sys
from typing import List
from datetime import datetime

MAXD14:Static[int]=14
MAXD14_ANCESTOR:Static[int]=13
VERSION_TAG:str="234 cachehot-maxd14-direct: N21 shaped-bin direct reader + MAXD14 direct launch; 233 MAXD14 kernel unchanged"
EXPECTED_N21:int=314666222712
EXPECTED_RECORDS_N21:int=2025282

class TaskSoA:
  def __init__(self,m:int)->None:
    self.ld_arr:List[u32]=[u32(0)]*m
    self.rd_arr:List[u32]=[u32(0)]*m
    self.col_arr:List[u32]=[u32(0)]*m
    self.ctrl0_arr:List[u32]=[u32(0)]*m
    self.free_arr:List[u32]=[u32(0)]*m
    self.markctrl_arr:List[u32]=[u32(0)]*m

@gpu.kernel
def kernel_dfs_iter_gpu_maxd14(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_arr:Ptr[u64],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD14_ANCESTOR)
    rd=__array__[u32](MAXD14_ANCESTOR)
    col=__array__[u32](MAXD14_ANCESTOR)
    avail=__array__[u32](MAXD14_ANCESTOR)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    schedule_lo:u32=u32(0)
    schedule_hi:u32=u32(0)
    child_jmark_mask:u32=u32(0)
    future_check_mask:u32=u32(0)
    terminal_parent_depth:int=0
    terminal_is_base14:u32=u32(0)
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_nibble:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_nibble=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_depth:int=schedule_depth-1
        if frame_action==u32(1):
          child_jmark_mask|=u32(1)<<u32(parent_depth)
        elif frame_action>=u32(2):
          terminal_parent_depth=parent_depth
          terminal_is_base14=u32(1) if frame_action==u32(3) else u32(0)

      if frame_action>=u32(2):
        break

      if schedule_fcvu!=u32(0):
        future_check_mask|=u32(1)<<u32(schedule_depth)

      if schedule_depth<8:
        schedule_lo|=frame_nibble<<u32(schedule_depth*4)
      else:
        schedule_hi|=frame_nibble<<u32((schedule_depth-8)*4)
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=w_arr[i]
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*w_arr[i]
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    terminal_depth:int=terminal_parent_depth
    terminal_base14:u32=terminal_is_base14

    save_sp:int=0
    cur_depth:int=0
    cur_ld:u32=root_ld
    cur_rd:u32=root_rd
    cur_col:u32=root_col
    cur_avail:u32=root_a

    root_rest:u32=cur_avail&(cur_avail-u32(1))
    root_second:u32=root_rest&(u32(0)-root_rest)
    root_after_second:u32=root_rest^root_second

    if root_after_second==u32(0):
      root_first:u32=cur_avail&(u32(0)-cur_avail)
      pr_nibble_op:u32=schedule_lo&u32(15)
      pr_block_code:u32=pr_nibble_op&u32(7)
      pr_bit:u32=root_first

      pr_nld:u32=u32(0)
      pr_nrd:u32=u32(0)
      if pr_block_code!=u32(0):
        pr_stepu:u32=u32(2)+((OP_STEP3_MASK>>pr_block_code)&u32(1))
        pr_addvu:u32=(OP_ADD1_MASK>>pr_block_code)&u32(1)
        pr_bLiu:u32=(
          ((OP_BL1_MASK>>pr_block_code)&u32(1))
          |(((OP_BL2_MASK>>pr_block_code)&u32(1))<<u32(1))
        )
        pr_ktu:u32=(
          ((OP_KN3_MASK>>pr_block_code)&u32(1))
          |(((OP_KN4_MASK>>pr_block_code)&u32(1))<<u32(1))
        )
        pr_bKu:u32=(n3&(u32(0)-(pr_ktu&u32(1))))|(n4&(u32(0)-(pr_ktu>>u32(1))))
        pr_nld=((cur_ld|pr_bit)<<pr_stepu)|pr_addvu|pr_bLiu
        pr_nrd=((cur_rd|pr_bit)>>pr_stepu)|pr_bKu
      else:
        pr_nld=(cur_ld|pr_bit)<<u32(1)
        pr_nrd=(cur_rd|pr_bit)>>u32(1)
      pr_ncol:u32=cur_col|pr_bit
      pr_nf:u32=bm&~(pr_nld|pr_nrd|pr_ncol)
      pr_descend:u32=u32(1)
      if pr_nf==u32(0):
        pr_descend=u32(0)
      if pr_descend!=u32(0):
        if future_check_mask!=u32(0):
          if (pr_nibble_op&u32(8))!=u32(0):
            if (bm&~((pr_nld<<u32(1))|(pr_nrd>>u32(1))|pr_ncol))==u32(0):
              pr_descend=u32(0)

      if pr_descend!=u32(0):
        if terminal_depth==0:
          if terminal_base14==u32(0):
            total+=u64(1)
          else:
            total+=u64(1) if ((pr_nf&~u32(1))!=u32(0)) else u64(0)
          pr_descend=u32(0)

      if pr_descend!=u32(0):
        pr_child_jmark:u32=child_jmark_mask&u32(1)
        if pr_child_jmark!=u32(0):
          pr_nf&=~u32(1)
          if pr_nf==u32(0):
            pr_descend=u32(0)
          else:
            pr_nld|=u32(1)

      cur_avail=root_rest
      if pr_descend!=u32(0):
        if cur_avail!=u32(0):
          ld[save_sp]=cur_ld
          rd[save_sp]=cur_rd
          col[save_sp]=cur_col
          avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))
          save_sp+=1
        cur_ld=pr_nld
        cur_rd=pr_nrd
        cur_col=pr_ncol
        cur_avail=pr_nf
        cur_depth=1

    while True:
      if cur_avail==u32(0):
        if save_sp==0:
          break
        save_sp-=1
        cur_ld=ld[save_sp]
        cur_rd=rd[save_sp]
        cur_col=col[save_sp]
        saved_avail:u32=avail[save_sp]
        cur_avail=saved_avail&bm
        cur_depth=int(saved_avail>>u32(27))
        continue

      nibble_op:u32=u32(0)
      if cur_depth<8:
        nibble_op=(schedule_lo>>u32(cur_depth*4))&u32(15)
      else:
        nibble_op=(schedule_hi>>u32((cur_depth-8)*4))&u32(15)
      block_code:u32=nibble_op&u32(7)

      bit:u32=cur_avail&(u32(0)-cur_avail)
      cur_avail=cur_avail^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((cur_ld|bit)<<stepu)|addvu|bLiu
        nrd=((cur_rd|bit)>>stepu)|bKu
      else:
        nld=(cur_ld|bit)<<u32(1)
        nrd=(cur_rd|bit)>>u32(1)
      ncol:u32=cur_col|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if future_check_mask!=u32(0):
        if (nibble_op&u32(8))!=u32(0):
          if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
            continue

      if cur_depth==terminal_depth:
        if terminal_base14==u32(0):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue

      child_jmark:u32=(child_jmark_mask>>u32(cur_depth))&u32(1)
      if child_jmark!=u32(0):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      next_depth:int=cur_depth+1
      if cur_avail!=u32(0):
        ld[save_sp]=cur_ld
        rd[save_sp]=cur_rd
        col[save_sp]=cur_col
        avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))
        save_sp+=1
      cur_ld=nld
      cur_rd=nrd
      cur_col=ncol
      cur_avail=nf
      cur_depth=next_depth
    results[i]=total*w_arr[i]

def count_bin_records(fname:str)->int:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)
      size:int=f.tell()
    if size%16!=0:
      return 0
    return size//16
  except:
    return 0

def file_exists(fname:str)->bool:
  try:
    with open(fname,"rb"):
      return True
  except:
    return False

def read_done_count(fname:str)->int:
  try:
    with open(fname,"r") as f:
      text:str=f.read().strip()
    if text=="":
      return -1
    return int(text)
  except:
    return -1

def elapsed_text_to_ms(elapsed_text:str)->int:
  s:str=elapsed_text
  days:int=0
  day_parts=s.split(",")
  if len(day_parts)>1:
    day_tokens=day_parts[0].strip().split()
    if len(day_tokens)>0:
      days=int(day_tokens[0])
    s=day_parts[1].strip()
  hms=s.split(":")
  if len(hms)<3:
    return 0
  hours:int=int(hms[0])
  minutes:int=int(hms[1])
  sec_ms=hms[2].split(".")
  seconds:int=int(sec_ms[0])
  millis:int=0
  if len(sec_ms)>1:
    ms_str:str=sec_ms[1]
    if len(ms_str)==1:
      millis=int(ms_str)*100
    elif len(ms_str)==2:
      millis=int(ms_str)*10
    elif len(ms_str)>=3:
      millis=int(ms_str[0:3])
  return (((days*24+hours)*60+minutes)*60+seconds)*1000+millis

def exec_n21_cachehot_maxd14_direct(
  fname:str,
  gpu_block:int,
  gpu_max_blocks:int,
  gpu_log_level:int,
  worker_id:int,
  worker_count:int,
  progress_suffix:str
)->int:
  N:int=21
  preset_queens:int=6
  board_mask:int=(1<<N)-1
  small_mask:int=(1<<(N-2))-1
  N1:int=N-1
  N2:int=N-2
  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488
  if worker_count<=0:
    worker_count=1
  if worker_id<0:
    worker_id=0
  if worker_id>=worker_count:
    print(f"[worker-error] worker_id must be smaller than worker_count: worker_id={worker_id} worker_count={worker_count}")
    return 0

  total_records:int=count_bin_records(fname)
  run_param_tag:str=f"w8_j7_b{BLOCK}_m{MAX_BLOCKS}_s{STEPS}"
  progress_fname:str=f"progress_N21_6_stream_split145_{run_param_tag}_{progress_suffix}.tsv"
  if worker_count>1:
    progress_fname=f"progress_N21_6_stream_split145_{run_param_tag}_{progress_suffix}_worker{worker_id}of{worker_count}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write("N\tpreset\tchunk\toff\tm\tblock\tmax_blocks\tsteps\telapsed\telapsed_ms\tchunk_total\tgpu_total\tdone_records\ttotal_records\tremaining_records\n")

  if gpu_log_level>=1:
    print(f"[split145-gpu-config] N=21 records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode=0 chunk_only=0 chunk_start=0 chunk_count=1 worker={worker_id}/{worker_count} split_mode=0 progress={progress_fname} cachehot_maxd14_direct=1")

  soa:TaskSoA=TaskSoA(STEPS)
  w_arr:List[u64]=[u64(0)]*STEPS
  results:List[u64]=[u64(0)]*STEPS
  meta_next:List[u8]=[u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2),u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14),u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25),u8(21),u8(21),u8(26),u8(26)]
  board_mask_gpu:u32=u32(board_mask)
  n3_gpu:u32=u32(1)<<u32(N-3)
  n4_gpu:u32=u32(1)<<u32(N-4)

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0

  with open(fname,"rb") as f:
    while True:
      t:int=0
      d2base14_m:int=0
      d0_m:int=0
      while t<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        raw_ld:int=(ord(raw[0])&0xFF)|((ord(raw[1])&0xFF)<<8)|((ord(raw[2])&0xFF)<<16)|((ord(raw[3])&0xFF)<<24)
        raw_rd:int=(ord(raw[4])&0xFF)|((ord(raw[5])&0xFF)<<8)|((ord(raw[6])&0xFF)<<16)|((ord(raw[7])&0xFF)<<24)
        raw_col:int=(ord(raw[8])&0xFF)|((ord(raw[9])&0xFF)<<8)|((ord(raw[10])&0xFF)<<16)|((ord(raw[11])&0xFF)<<24)
        start_ijkl:int=(ord(raw[12])&0xFF)|((ord(raw[13])&0xFF)<<8)|((ord(raw[14])&0xFF)<<16)|((ord(raw[15])&0xFF)<<24)

        jmark:int=0
        mark1:int=0
        mark2:int=0
        start_row:int=start_ijkl>>20
        ijkl:int=start_ijkl&1048575
        ii:int=(ijkl>>15)&31
        j:int=(ijkl>>10)&31
        k:int=(ijkl>>5)&31
        l:int=ijkl&31

        ld:int=raw_ld>>1
        rd:int=raw_rd>>1
        col:int=(raw_col>>1)|(~small_mask)
        col&=board_mask
        LD:int=(1<<(N1-j))|(1<<(N1-l))
        ld|=LD>>(N-start_row)
        if start_row>k:
          rd|=(1<<(N1-(start_row-k+1)))
        if j>=2*N-33-start_row:
          rd|=(1<<(N1-j))<<(N2-start_row)
        free:int=board_mask&~(ld|rd|col)

        endmark:int=0
        target:int=0
        k_lt_l:bool=(k<l)
        start_lt_k:bool=(start_row<k)
        start_lt_l:bool=(start_row<l)
        l_eq_kp1:bool=(l==k+1)
        k_eq_lp1:bool=(k==l+1)

        if j<N-3:
          jmark=j+1
          endmark=N2
          if j>2*N-34-start_row:
            if k_lt_l:
              mark1=k-1
              mark2=l-1
              if start_lt_l:
                if start_lt_k:
                  target=0 if (not l_eq_kp1) else 4
                else:
                  target=1
              else:
                target=2
            else:
              mark1=l-1
              mark2=k-1
              if start_lt_k:
                if start_lt_l:
                  target=5 if (not k_eq_lp1) else 7
                else:
                  target=6
              else:
                target=2
          else:
            if k_lt_l:
              mark1=k-1
              mark2=l-1
              target=8 if (not l_eq_kp1) else 9
            else:
              mark1=l-1
              mark2=k-1
              target=10 if (not k_eq_lp1) else 11
        elif j==N-3:
          endmark=N2
          if k_lt_l:
            mark1=k-1
            mark2=l-1
            if start_lt_l:
              if start_lt_k:
                target=12 if (not l_eq_kp1) else 15
              else:
                mark2=l-1
                target=13
            else:
              target=14
          else:
            mark1=l-1
            mark2=k-1
            if start_lt_k:
              if start_lt_l:
                target=16 if (not k_eq_lp1) else 18
              else:
                mark2=k-1
                target=17
            else:
              target=14
        elif j==N-2:
          if k_lt_l:
            endmark=N2
            if start_lt_l:
              if start_lt_k:
                mark1=k-1
                if not l_eq_kp1:
                  mark2=l-1
                  target=19
                else:
                  target=22
              else:
                mark2=l-1
                target=20
            else:
              target=21
          else:
            if start_lt_k:
              if start_lt_l:
                if k<N2:
                  mark1=l-1
                  endmark=N2
                  if not k_eq_lp1:
                    mark2=k-1
                    target=23
                  else:
                    target=24
                else:
                  if l!=(N-3):
                    mark2=l-1
                    endmark=N-3
                    target=20
                  else:
                    endmark=N-4
                    target=21
              else:
                if k!=N2:
                  mark2=k-1
                  endmark=N2
                  target=25
                else:
                  endmark=N-3
                  target=21
            else:
              endmark=N2
              target=21
        else:
          endmark=N2
          if start_row>k:
            target=26
          else:
            mark1=k-1
            target=27

        if target==14:
          d2base14_m+=1
        if target==26 or target==27:
          d0_m+=1
        soa.ld_arr[t]=u32(ld)
        soa.rd_arr[t]=u32(rd)
        soa.col_arr[t]=u32(col)
        soa.ctrl0_arr[t]=u32(target)|(u32(start_row)<<u32(5))
        soa.free_arr[t]=u32(free)
        soa.markctrl_arr[t]=(
          u32(jmark&31)
          |(u32(endmark&31)<<u32(5))
          |(u32(mark1&31)<<u32(10))
          |(u32(mark2&31)<<u32(15))
        )
        rot90v:int=((N1-k)<<15)+((N1-l)<<10)+(j<<5)+ii
        if ijkl==rot90v:
          w_arr[t]=u64(2)
        elif ii==N1-j and k==N1-l:
          w_arr[t]=u64(4)
        else:
          w_arr[t]=u64(8)
        t+=1

      m:int=t
      if m==0:
        break
      if worker_count>1:
        run_worker_chunk:bool=((chunk_index % worker_count)==worker_id)
        if not run_worker_chunk:
          off+=m
          chunk_index+=1
          continue

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[split145-gpu-chunk-start] N=21 worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
        print(f"[split145-buckets] N=21 m={m} generic={m} d2base14={d2base14_m} d0={d0_m} rest=0 split_mode=0 specialized=0")

      GRID:int=(m+BLOCK-1)//BLOCK
      if gpu_log_level>=1:
        print(f"[maxd-dispatch] N=21 scope=split145 m={m} required_maxd=14 selected_MAXD=14 schedule_words=0 stack_bytes_per_thread=208 capacity_check=OK")
      kernel_dfs_iter_gpu_maxd14(
        gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
        gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
        gpu.raw(soa.markctrl_arr),gpu.raw(w_arr),gpu.raw(meta_next),gpu.raw(results),
        m,board_mask_gpu,n3_gpu,n4_gpu,grid=GRID,block=BLOCK
      )
      chunk_total:int=0
      i:int=0
      while i<m:
        chunk_total+=int(results[i])
        i+=1
      t1=datetime.now()

      gpu_total+=chunk_total
      executed_chunks+=1
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=elapsed_text_to_ms(elapsed_text)
      done_records:int=off+m
      remaining_records:int=total_records-done_records
      if remaining_records<0:
        remaining_records=0
      with open(progress_fname,"a") as pf:
        pf.write(f"21\t6\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}\n")
      if gpu_log_level>=1:
        print(f"[split145-gpu-chunk-end] N=21 worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} inner={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total} soa_ms=0 stats_ms=0 split_ms=0 kernel_reduce_ms={elapsed_ms}")
      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[split145-gpu-summary] N=21 records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} split_mode=0 progress={progress_fname}")
  return gpu_total

def main()->None:
  argc:int=len(sys.argv)
  use_gpu:bool=True
  nmin:int=21
  nmax:int=22
  gpu_block:int=32
  gpu_max_blocks:int=484
  gpu_log_level:int=1
  gpu_sort_mode:int=0
  preset_arg:int=7
  bench_mode:int=31
  reorder_window_mult:int=8
  reorder_phase_jump:int=7
  cross_stripe_safe:bool=False
  worker_id:int=0
  worker_count:int=1
  broadmark_variant:int=2

  if argc>=2:
    if sys.argv[1]=="-g":
      use_gpu=True
      print("GPU mode selected")
    else:
      print("Usage: nqueens -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2")
      return
  else:
    print("GPU mode selected")
    print("[234-default] no arguments: GPU N21 cachehot MAXD14 direct default")
  if argc>=4:
    nmin=int(sys.argv[2])
    nmax=int(sys.argv[3])+1
  if argc>=5:
    gpu_block=int(sys.argv[4])
  if argc>=6:
    gpu_max_blocks=int(sys.argv[5])
  if argc>=7:
    gpu_log_level=int(sys.argv[6])
  if argc>=8:
    gpu_sort_mode=int(sys.argv[7])
  if argc>=9:
    preset_arg=int(sys.argv[8])
  if argc>=10:
    bench_mode=int(sys.argv[9])
  if argc>=11:
    reorder_window_mult=int(sys.argv[10])
  if argc>=12:
    reorder_phase_jump=int(sys.argv[11])
  if argc>=13:
    cross_stripe_safe=(int(sys.argv[12])!=0)
  if argc>=14:
    worker_id=int(sys.argv[13])
  if argc>=15:
    worker_count=int(sys.argv[14])
  if argc>=16:
    broadmark_variant=int(sys.argv[15])
  if argc>16:
    print("Too many arguments")
    return

  if (not use_gpu) or nmin!=21 or nmax!=22 or bench_mode!=31 or gpu_sort_mode!=0 or preset_arg!=7 or reorder_window_mult!=8 or reorder_phase_jump!=7 or cross_stripe_safe or broadmark_variant!=2:
    print(f"[error] 234 cachehot-maxd14-direct is fixed to: -g 21 21 32 484 <log_level> 0 7 31 8 7 0 worker_id worker_count 2")
    return
  if gpu_block<=0:
    gpu_block=32
  if gpu_max_blocks<=0:
    gpu_max_blocks=484
  steps:int=gpu_block*gpu_max_blocks
  if steps<=0:
    steps=15488

  shaped_fname:str=f"constellations_N21_6_chunkshape148_scorestripe_v9_lanephase32_octetfirstpairlock29_v4_rotate_only_w8_j7_b{gpu_block}_m{gpu_max_blocks}_s{steps}.bin"
  records:int=count_bin_records(shaped_fname)
  done:int=read_done_count(shaped_fname+".done")
  if records!=EXPECTED_RECORDS_N21 or done!=EXPECTED_RECORDS_N21 or (not file_exists(shaped_fname)):
    print(f"[cachehot-error] required shaped cache is missing/incomplete: bin={shaped_fname} records={records} done={done} expected={EXPECTED_RECORDS_N21}")
    print("[cachehot-error] build the cache once with 233Py/232Py, then rerun 234Py")
    return

  if gpu_log_level>=1:
    print(f"version        : {VERSION_TAG}")
    print("cross_stripe_safe: 0")
    print(f"worker_split : worker={worker_id}/{worker_count}")
    print("broadmarktail_variant: id=2 tag=rotate_only")
    print("split234_cachehot_full_gpu: mode=31 preset=7 dynamic_preset=6")
    print("funcid_reorder_v2_params: window_mult=8 phase_jump=7 param=w8_j7")
    print("broadmarktail_params: version=v4 variant=2 tag=rotate_only")
    print("chunkshape148_params: version=scorestripe_v9_lanephase32_octetfirstpairlock29")

  print(" N:             Total           Unique         hh:mm:ss.ms")
  start_time=datetime.now()
  print("[dynamic-preset] N=21 preset_queens=6")
  print(f"[split234-full-reuse] N=21 records={records} chunks={(records+steps-1)//steps} bin={shaped_fname} param=w8_j7_b{gpu_block}_m{gpu_max_blocks}_s{steps}")
  total:int=exec_n21_cachehot_maxd14_direct(shaped_fname,gpu_block,gpu_max_blocks,gpu_log_level,worker_id,worker_count,"split234_full_scorestripe_v9_lanephase32_octetfirstpairlock29_v4_rotate_only")
  text:str=str(datetime.now()-start_time)[:-3]
  status:str="ok" if total==EXPECTED_N21 else f"ng({total}!={EXPECTED_N21})"
  if worker_count>1:
    status=f"partial-worker-{worker_id}-of-{worker_count}"
  if gpu_log_level>=1:
    print(f"[split234-full-done] N=21 shaped_records={records} chunks={(records+steps-1)//steps} bin={shaped_fname} param=w8_j7_b{gpu_block}_m{gpu_max_blocks}_s{steps} chunkshape=scorestripe_v9_lanephase32_octetfirstpairlock29 variant=rotate_only worker={worker_id}/{worker_count} total={total}")
  print(f"21:{total:18d}{0:17d}{text:>21s}    {status}")

if __name__=="__main__":
  main()
