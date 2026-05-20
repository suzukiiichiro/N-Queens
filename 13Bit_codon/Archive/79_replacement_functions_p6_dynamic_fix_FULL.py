# -*- coding: utf-8 -*-
# 79 preset=6 dynamic fix replacement functions
#
# 78 P5 FIX を維持した上で、preset=6 の SQd0 start>k 問題を修正します。
#
# 79 FIX:
#   SQd0 / fid=27(SQd0BkB) で DFS 開始 row が mark1(k) を既に越えている場合、
#   pending Bk は dynamic preset 側で既に通過済みなので、fid=26(SQd0B) と等価化する。
#
#   fid=27, row>mark1  =>  fid=26
#
# 既存の dfs_iter() と dfs() をこのファイル内の同名関数で置き換えてください。
# GPU kernel_dfs_iter_gpu() 側にも同じ fid 正規化を入れてください。

"""dfs()の非再帰版"""
def dfs_iter(
  meta,blockK,blockL,board_mask,
  functionid:int,ld:int,rd:int,col:int,row:int,free:int,
  jmark:int,endmark:int,mark1:int,mark2:int
)->u64:
  """
  CPU 上で DFS を非再帰で実行する。

  78 FIX:
    funcptn==4 / P5 / fid=8..11 を追加。

  79 FIX:
    preset=6 dynamic SQd0 start>k 対応。
    fid=27(SQd0BkB) かつ row>mark1(k) のとき fid=26(SQd0B) に正規化する。
  """

  total:u64=u64(0)
  stack:List[Tuple[int,int,int,int,int,int]]=[(functionid,ld,rd,col,row,free)]

  while stack:
    functionid,ld,rd,col,row,free=stack.pop()

    #######################################
    # 79 preset=6 dynamic fix
    #
    # SQd0 / fid=27(SQd0BkB):
    #   dynamic preset により DFS 開始 row が k(mark1) を越えている場合、
    #   Bk は既に preset 展開側で通過済みであり、fid=26(SQd0B) と等価。
    #
    #   fid27 + row>mark1  =>  fid26
    #
    # 注意:
    #   盤面状態(ld/rd/col/free/row)は一切変更しない。
    #   functionid だけを正規化する。
    #######################################
    if functionid==27 and row>mark1:
      functionid=26

    if not free:
      continue

    next_funcid,funcptn,avail_flag=meta[functionid]
    avail:int=free

    if funcptn==5 and row==endmark:
      if functionid==14:
        total+=u64(1) if (avail>>1) else u64(0)
      else:
        total+=u64(1)
      continue

    step:int=1
    add1:int=0
    row_step:int=row+1
    use_blocks:bool=False
    use_future:bool=(avail_flag==1)
    local_next_funcid:int=functionid
    _blockK:int=0
    _blockL:int=0

    if funcptn in (0,1,2):
      at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)
      if at_mark and avail:
        step=2 if funcptn in (0,1) else 3
        add1=1 if (funcptn==1 and functionid==20) else 0
        row_step=row+step
        _blockK=blockK[functionid]
        _blockL=blockL[functionid]
        use_blocks=True
        use_future=False
        local_next_funcid=next_funcid

    elif funcptn==3 and row==jmark:
      avail&=~1
      ld|=1
      local_next_funcid=next_funcid
      if not avail:
        continue

    elif funcptn==4 and row==mark1:
      stack.append((next_funcid,ld,rd,col,row,avail))
      continue

    if use_blocks:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1
        nld:int=((ld|bit)<<step)|add1|_blockL
        nrd:int=((rd|bit)>>step)|_blockK
        ncol:int=col|bit
        nf:int=board_mask&~(nld|nrd|ncol)
        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))
      continue

    if not use_future:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1
        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit
        nf:int=board_mask&~(nld|nrd|ncol)
        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))
      continue

    if row_step>=endmark:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1
        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit
        nf:int=board_mask&~(nld|nrd|ncol)
        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))
      continue

    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit
      nf:int=board_mask&~(nld|nrd|ncol)
      if not nf:
        continue
      if board_mask&~((nld<<1)|(nrd>>1)|ncol):
        stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

  return total


"""汎用 DFS カーネル。古い SQ???? 関数群を 1 本化し、func_meta の記述に従って切り替える。"""
def dfs(
    meta:List[Tuple[int,int,int]],
    blockK_by_funcid:List[int],blockL_by_funcid:List[int],
    board_mask:int,
    functionid:int,
    ld:int,rd:int,col:int,row:int,free:int,
    jmark:int,endmark:int,mark1:int,mark2:int)->u64:
  """
  78 FIX:
    funcptn==4 / P5 / fid=8..11 を追加。

  79 FIX:
    preset=6 dynamic SQd0 start>k 対応。
    fid=27(SQd0BkB) かつ row>mark1(k) のとき fid=26(SQd0B) に正規化する。
  """

  #######################################
  # 79 preset=6 dynamic fix
  #
  # SQd0 / fid=27(SQd0BkB):
  #   DFS 開始 row が k(mark1) を越えた後は fid=26(SQd0B) と等価。
  #   盤面状態(ld/rd/col/free/row)は変更せず、functionid のみ正規化する。
  #######################################
  if functionid==27 and row>mark1:
    functionid=26

  next_funcid,funcptn,avail_flag=meta[functionid]
  avail:int=free
  if not avail:
    return u64(0)

  total:u64=u64(0)

  if funcptn==5 and row==endmark:
    if functionid==14:
      return u64(1) if (avail>>1) else u64(0)
    return u64(1)

  step:int=1
  add1:int=0
  row_step:int=row+1
  use_blocks:bool=False
  use_future:bool=(avail_flag==1)
  local_next_funcid:int=functionid
  bK:int=0
  bL:int=0

  if funcptn in (0,1,2):
    at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)
    if at_mark and avail:
      step=2 if funcptn in (0,1) else 3
      add1=1 if (funcptn==1 and functionid==20) else 0
      row_step=row+step
      bK=blockK_by_funcid[functionid]
      bL=blockL_by_funcid[functionid]
      use_blocks=True
      use_future=False
      local_next_funcid=next_funcid

  elif funcptn==3 and row==jmark:
    avail&=~1
    ld|=1
    local_next_funcid=next_funcid
    if not avail:
      return u64(0)

  elif funcptn==4 and row==mark1:
    return dfs(
      meta,
      blockK_by_funcid,
      blockL_by_funcid,
      board_mask,
      next_funcid,
      ld,rd,col,row,avail,
      jmark,endmark,mark1,mark2
    )

  if use_blocks:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      nld:int=((ld|bit)<<step)|add1|bL
      nrd:int=((rd|bit)>>step)|bK
      ncol:int=col|bit
      nf:int=board_mask&~(nld|nrd|ncol)
      if nf:
        total+=dfs(meta,blockK_by_funcid,blockL_by_funcid,board_mask,local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
    return total

  if not use_future:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit
      nf:int=board_mask&~(nld|nrd|ncol)
      if nf:
        total+=dfs(meta,blockK_by_funcid,blockL_by_funcid,board_mask,local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
    return total

  if row_step>=endmark:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit
      nf:int=board_mask&~(nld|nrd|ncol)
      if nf:
        total+=dfs(meta,blockK_by_funcid,blockL_by_funcid,board_mask,local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)
    return total

  while avail:
    bit:int=avail&-avail
    avail&=avail-1
    nld:int=(ld|bit)<<1
    nrd:int=(rd|bit)>>1
    ncol:int=col|bit
    nf:int=board_mask&~(nld|nrd|ncol)
    if not nf:
      continue
    if board_mask&~(((nld<<1)|(nrd>>1)|ncol)):
      total+=dfs(meta,blockK_by_funcid,blockL_by_funcid,board_mask,local_next_funcid,nld,nrd,ncol,row_step,nf,jmark,endmark,mark1,mark2)

  return total
