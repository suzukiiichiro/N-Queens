#!/usr/bin/env luajit


NQueens08={};
NQueens08.new=function()
  local this={
    TOTAL=0;
    UNIQUE=0;
    COUNT2=0;
    COUNT4=0;
    COUNT8=0;
    bit;
    board={};
    trial={};
    scratch={};
  };
  function NQueens08:secstotime(secs)
    sec=math.floor(secs);
    if(sec>59) then
    local hour = math.floor(sec*0.000277777778)
    local minute = math.floor(sec*0.0166666667) - hour*60
    sec=sec-hour*3600-minute*60;
    if(sec<10)then sec="0"..sec end
    if(hour<10)then hour="0"..hour end
    if(minute<10)then minute="0"..minute end
    return hour..":"..minute..":"..sec;
    end
    if(sec<10)then sec="0"..sec end
    return "00:00:"..sec
  end
  function NQueens08:rotate(bf,af,size)
    for i=0,size,1 do
      local t=0;
      for j=0,size,1 do
        t=bit.bor(t,bit.lshift(bit.band(bit.rshift(bf[j],i),1),(size-j-1)));
      end
      af[i]=t;
    end
  end
  function NQueens08:vMirror(bf,af,size)
    local score;
    for i=0,size,1 do
      score=bf[i];
      af[i]=self:rh(score,size-1)
    end
  end
  function NQueens08:rh(a,size)
    local tmp=0;
    for i=0,size,1 do
      if bit.band(a,bit.lshift(1,i))~=0 then
        return bit.bor(tmp,bit.lshift(1,(size-i)));
      end
    end
    return tmp;
  end
  function NQueens08:intncmp(lt,rt,size)
    local rtn=0;
    for k=0,size,1 do
      rtn=lt[k]-rt[k];
      if(rtn~=0)then break; end
    end
    return rtn;
  end
  function NQueens08:rbits(byte,size)
    local score=0;
    for i=size,0,-1 do
      if bit.band(bit.arshift(byte,i),1)==0 then
        score=score+2^i;
      end
    end
    return score;
  end
  function NQueens08:symmetryOps(size)
    local nEquiv;
    for i=0,size,1 do self.trial[i]=self.board[i];end
    --90
    self:rotate(self.trial,self.scratch,size);
    local k=self:intncmp(self.board,self.scratch,size);
    if(k>0)then return; end
    if(k==0)then nEquiv=2;
    else
      --180
      self:rotate(self.scratch,self.trial,size);
      k=self:intncmp(self.board,self.trial,size);
      if(k>0)then return; end
      if(k==0)then nEquiv=4;
      else
        --270
        self:rotate(self.trial,self.scratch,size);
        k=self:intncmp(self.board,self.scratch,size);
        if(k>0)then return; end
        nEquiv=8;
      end
    end
    for i=0,size,1 do
      self.scratch[i]=self.board[i];
    end
    self:vMirror(self.scratch,self.trial,size);
    k=self:intncmp(self.board,self.trial,size);
    if(k>0)then return; end
    --90
    if(nEquiv>2)then
      self:rotate(self.trial,self.scratch,size);
      k=self:intncmp(self.board,self.scratch,size);
      if(k>0)then return; end
      --180
      if(nEquiv>4)then
        self:rotate(self.scratch,self.trial,size);
        k=self:intncmp(self.board,self.trial,size);
        if(k>0)then return; end
        --270
        self:rotate(self.trial,self.scratch,size);
        k=self:intncmp(self.board,self.scratch,size);
        if(k>0)then return ; end
      end
    end
    if(nEquiv==2)then self.COUNT2=self.COUNT2+1;end
    if(nEquiv==4)then self.COUNT4=self.COUNT4+1;end
    if(nEquiv==8)then self.COUNT8=self.COUNT8+1;end
  end
  function NQueens08:NQueens_rec(size,min,left,down,right)
    local MASK=bit.lshift(1,size)-1;
    local bitmap=bit.band(MASK,self:rbits(bit.bor(left,down,right),size-1));
    if min==size then
      if bitmap==0 then
        self.board[min]=bitmap;
        self:symmetryOps(size);
      end
    else
      if min~=0 then
        lim=size;
      else
        lim=(size+1)/2;
      end
      for s=min+1,lim,1 do
        if bitmap==0 then
          break;
        end
        self.BIT=bit.band(-bitmap,bitmap);
        self.board[min]=self.BIT;
        bitmap=bit.bxor(bitmap,self.BIT);
        self:NQueens_rec(size,min+1,bit.lshift(bit.bor(left,self.BIT),1),bit.bor(down,self.BIT),bit.rshift(bit.bor(right,self.BIT),1));
      end
    end
  end
  function NQueens08:NQueens()
    local max=15;
    print(" N:            Total       Unique    hh:mm:ss");
    for size=2,max,1 do
      self.TOTAL=0;
      self.UNIQUE=0;
      self.COUNT2=0;
      self.COUNT4=0;
      self.COUNT8=0;
      for k=0,size-1,1 do self.board[k]=k; end
      s=os.time();
      self:NQueens_rec(size,0,0,0,0);
      self.TOTAL=self.COUNT2*2+self.COUNT4*4+self.COUNT8*8;
      self.UNIQUE=self.COUNT2+self.COUNT4+self.COUNT8;
      print(string.format("%2d:%17d%13d%12s",size,self.TOTAL,self.UNIQUE,self:secstotime(os.difftime(os.time(),s))));
    end 
  end
  return setmetatable(this,{__index=NQueens08});
end

NQueens07={};
NQueens07.new=function()
  local this={
    TOTAL=0;
    UNIQUE=0;
    COUNT2=0;
    COUNT4=0;
    COUNT8=0;
    bit;
    board={};
    trial={};
    scratch={};
  };
  function NQueens07:rbits(byte,size)
    local score=0;
    for i=size,0,-1 do
      if bit.band(bit.arshift(byte,i),1)==0 then
        score=score+2^i;
      end
    end
    return score;
  end 
  function NQueens07:rotate(bf,af,size)
    for i=0,size,1 do
      local t=0;
      for j=0,size,1 do
        --t=t|((bf[j]>>i)&1)<<(size-j-1); 
        t=bit.bor(t,bit.lshift(bit.band(bit.rshift(bf[j],i),1),(size-j-1)));
      end
      af[i]=t;
    end
  end
  function NQueens07:vMirror(bf,af,size)
    local score;
    for i=0,size,1 do
      score=bf[i];
      af[i]=self:rh(score,size-1);
    end
  end
  function NQueens07:rh(a,size)
    local tmp=0;
    for i=0,size,1 do
      --if(a&(1<<i))then  
      if bit.band(a,bit.lshift(1,i))~=0 then
        --return tmp|(1<<(size-i)); 
        return bit.bor(tmp,bit.lshift(1,(size-i)));
      end
    end
    return tmp;
  end
  function NQueens07:intncmp(lt,rt,size)
    local rtn=0;
    for k=0,size,1 do
      rtn=lt[k]-rt[k];
      if(rtn~=0)then break;end
    end
    return rtn;
  end
  function NQueens07:secstotime(secs)
    sec=math.floor(secs);
    if(sec>59) then
      local hour = math.floor(sec*0.000277777778);
      local minute = math.floor(sec*0.0166666667) - hour*60;
      sec=sec-hour*3600-minute*60;
      if(sec<10)then sec="0"..sec end
      if(hour<10)then hour="0"..hour end
      if(minute<10)then minute="0"..minute end
      return hour..":"..minute..":"..sec;
    end
    if(sec<10)then sec="0"..sec end;
    return "00:00:"..sec;
  end
  function NQueens07:symmetryOps(size)
    local nEquiv;
    for i=0,size,1 do self.trial[i]=self.board[i]; end
    -- 90
    self:rotate(self.trial,self.scratch,size);
    local k=self:intncmp(self.board,self.scratch,size);
    if(k>0)then return; end
    if(k==0)then nEquiv=2;
    else
      --180
      self:rotate(self.scratch,self.trial,size);
      k=self:intncmp(self.board,self.trial,size);
      if(k>0)then return; end
      if(k==0)then nEquiv=4;
      else
        --270
        self:rotate(self.trial,self.scratch,size);
        k=self:intncmp(self.board,self.scratch,size);
        if(k>0)then return; end
        nEquiv=8;
      end
    end
    for i=0,size,1 do
      self.scratch[i]=self.board[i];
    end
    self:vMirror(self.scratch,self.trial,size);
    k=self:intncmp(self.board,self.trial,size);
    if(k>0)then return ;end
    if(nEquiv>2)then
      self:rotate(self.trial,self.scratch,size)
      k=self:intncmp(self.board,self.scratch,size);
      if(k>0)then return; end
      if(nEquiv>4)then
        self:rotate(self.scratch,self.trial,size);
        k=self:intncmp(self.board,self.trial,size);
        if(k>0)then return; end
        self:rotate(self.trial,self.scratch,size);
        k=self:intncmp(self.board,self.scratch,size);
        if(k>0)then return; end
      end
    end
    if(nEquiv==2)then self.COUNT2=self.COUNT2+1;end
    if(nEquiv==4)then self.COUNT4=self.COUNT4+1;end
    if(nEquiv==8)then self.COUNT8=self.COUNT8+1;end
  end
  function NQueens07:NQueens_rec(size,min,left,down,right)
    local MASK=bit.lshift(1,size)-1;
    local bitmap=bit.band(MASK,self:rbits(bit.bor(left,down,right),size-1));
    if min==size then
      if bitmap==0 then
        self.board[min]=bitmap;
        self:symmetryOps(size);
      end
    else
      while bitmap~=0 do
        self.BIT=bit.band(-bitmap,bitmap);
        self.board[min]=self.BIT;
        bitmap=bit.bxor(bitmap,self.BIT);
        self:NQueens_rec(size,min+1,bit.lshift(bit.bor(left,self.BIT),1),bit.bor(down,self.BIT),bit.rshift(bit.bor(right,self.BIT),1));
      end
    end
  end
  function NQueens07:NQueens()
    local max=15;
    print(" N:            Total       Unique    hh:mm:ss");
    for size=2,max,1 do
      self.TOTAL=0;
      self.UNIQUE=0;
      self.COUNT2=0;
      self.COUNT4=0;
      self.COUNT8=0;
      for k=0,size-1,1 do self.board[k]=k;end
      s=os.time();
      self:NQueens_rec(size,0,0,0,0);
      self.TOTAL=self.COUNT2*2 + self.COUNT4*4 + self.COUNT8*8;
      self.UNIQUE=self.COUNT2 + self.COUNT4 + self.COUNT8;
      print(string.format("%2d:%17d%13d%12s",size,self.TOTAL,self.UNIQUE,self:secstotime(os.difftime(os.time(),s))));
    end
  end
  return setmetatable(this,{__index=NQueens07});
end

NQueens06={};
NQueens06.new=function()
  local this={
    TOTAL=0;
    UNIQUE=0;
  };
  function NQueens06:rbits(byte,size)
    local score=0;
    for i=size,0,-1 do
      if bit.band(bit.arshift(byte,i),1)==0 then
        score=score+2^i;
      end
    end
    return score;
  end
  function NQueens06:secstotime(secs)
    sec=math.floor(secs);
    if(sec>59)then
      local hour=math.floor(sec*0.000277777778)
      local minute=math.floor(sec*0.0166666667)-hour*60
      sec=sec-hour*3600-minute*60
      if(sec<10)then sec="0"..sec end
      if(hour<10)then hour="0"..hour end
      if(nimute<10)then minute="0"..minute end
      return hour..":"..minute..":"..sec;
    end
    if(sec<10)then sec="0"..sec end
    return "00:00:"..sec;
  end
  function NQueens06:NQueens_rec(size,min,left,down,right)
    local bitmap=0;
    local BIT=0;
    local mask=bit.lshift(1,size)-1;
    if min==size then
      self.TOTAL=self.TOTAL+1;
    else
      bitmap=bit.band(self.MASK,self:rbits(bit.bor(left,down,right),size-1));
      while bitmap~=0 do
        BIT=bit.band(-bitmap,bitmap);
        bitmap=bit.bxor(bitmap,BIT);
        self:NQueens_rec(size,min+1,bit.lshift(bit.bor(left,BIT),1),bit.bor(down,BIT),bit.rshift(bit.bor(right,BIT),1));
      end
    end
  end
  function NQueens06:NQueens()
    local max=15;
    print(" N:            Total       Unique    hh:mm:ss");
    for size=2,max,1 do
      self.TOTAL=0;
      self.UNIQUE=0;
      s=os.time();
      self:NQueens_rec(size,0,0,0,0)
      print(string.format("%2d:%17d%13d%12s",size,self.TOTAL,self.UNIQUE,self:secstotime(os.difftime(os.time(),s))));
    end
  end
  return setmetatable(this,{__index=NQueens06});
end

NQueens05={};
NQueens05.new=function()
  local this={
    TOTAL=0;
    UNIQUE=0;
    colCheck={};
    diagCheck={};
    antiCheck={};
    board={};
  };
  function NQueens05:secstotime(secs)
    sec=math.floor(secs);
    if(sec>59) then
      local hour=math.floor(sec*0.000277777778)
      local minute=math.floor(sec*0.0166666667)-hour*60
      sec=sec-hour*3600-minute*60
      if(sec<10)then sec="0"..sec; end
      if(hour<10)then hour="0"..hour; end
      if(minute<10)then minute="0"..minute;end
      return hour..":"..minute..":"..sec;
    end
    if(sec<10)then sec="0"..sec end
    return "00:00:"..sec;
  end
  function NQueens05:intncmp(board,trial,size)
    local rtn=0;
    for k=0,size,1 do
      rtn=board[k]-trial[k];
      if(rtn~=0)then
        break;
      end
    end
    return rtn;
  end
  function NQueens05:rotate(trial,scratch,size,neg)
    local k;
    local incr;
    if neg then k=0; else k=size-1; end
    if neg then incr=1; else incr=-1; end
    local j=0;
    while j<size do
      scratch[j]=trial[k];
      k=k+incr;
      j=j+1;
    end
    if neg then k=size-1; else k=0; end
    local j=0;
    while j<size do
      trial[scratch[j]]=k
      k=k-incr;
      j=j+1;
    end
  end
  function NQueens05:vMirror(trial,size)
    for j=0,size,1 do
      trial[j]=(size-1)-trial[j];
    end
  end
  function NQueens05:symmetryOps(board,trial,scratch,size)
    local nEquiv=0;
    for i=0,size,1 do
      trial[i]=board[i];
    end
    -- 90
    self:rotate(trial,scratch,size,0);
    local k=self:intncmp(board,trial,size);
    if(k>0)then return 0; end
    if(k==0)then
      nEquiv=1;
    else
      --180
      self:rotate(trial,scratch,size,0);
      k=self:intncmp(board,trial,size);
      if(k>0)then return 0;end
      if(k==0)then
        nEquiv=2;
      else
        -- 270
        self:rotate(trial,scratch,size,0);
        k=self:intncmp(board,trial,size);
        if(k>0)then
          return 0;
        end
        nEquiv=4;
      end
    end
    for i=0,size,1 do
      trial[i]=board[i];
    end
    self:vMirror(trial,size);
    k=self:intncmp(board,trial,size);
    if(k>0) then
      return 0;
    end
    if(nEquiv>1)then
      -- 90
      self:rotate(trial,scratch,size,1);
      k=self:intncmp(board,trial,size);
      if(k>0)then
        return 0;
      end
      if(nEquiv>2)then
        -- 180
        self:rotate(trial,scratch,size,1);
        k=self:intncmp(board,trial,size);
        if(k>0)then
          return 0;
        end
        --270
        self:rotate(trial,scratch,size,1);
        k=self:intncmp(board,trial,size);
        if(k>0)then
          return 0;
        end
      end
    end
    return nEquiv * 2 ;
  end
  function NQueens05:NQueens_rec(board,row,size)
    local tmp;
    local trial={};
    local scratch={};
    if row==size-1 then
      if(self.diagCheck[row-board[row]+size-1]==true
        or self.antiCheck[row+board[row]]==true) then
        return ;
      end
      local tst=self:symmetryOps(board,trial,scratch,size);
      if(tst~=0)then
        self.UNIQUE=self.UNIQUE+1;
        self.TOTAL=self.TOTAL+tst;
      end
    else
      local lim;
      if row~=0 then lim=size;else lim=(size+1)/2; end
      for i=row,lim-1,1 do
        tmp=board[i];
        board[i]=board[row];
        board[row]=tmp;
        if self.antiCheck[row+board[row]]==nil
          and self.diagCheck[row-board[row]+(size-1)]==nil then
            self.antiCheck[row+board[row]],
            self.diagCheck[row-board[row]+(size-1)]=true,true;
            self:NQueens_rec(board,row+1,size);
            self.antiCheck[row+board[row]],
            self.diagCheck[row-board[row]+(size-1)]=nil,nil;
        end
      end
      tmp=board[row];
      for i=row+1,size,1 do
        board[i-1]=board[i];
      end
      board[size-1]=tmp
    end
  end
  function NQueens05:NQueens()
    local max=15;
    print(" N:            Total       Unique    hh:mm:ss");
    for size=4,max,1 do
      self.UNIQUE=0;
      self.TOTAL=0;
      s=os.time();
      for j=0,size,1 do
        self.board[j]=j;
      end
      self:NQueens_rec(self.board,0,size);
      print(string.format("%2d:%17d%13d%12s",size,self.TOTAL,self.UNIQUE,self:secstotime(os.difftime(os.time(),s))));
    end
  end
  return setmetatable(this,{__index=NQueens05});
end

NQueens04={};
NQueens04.new=function()
  local this={
    TOTAL=0;
    UNIQUE=0;
    nEquiv=0;
    colCheck={};
    diagCheck={};
    antiCheck={};
    board={};
  };
  function NQueens04:secstotime(secs)
    sec=math.floor(secs);
    if(sec>59) then
      local hour=math.floor(sec*0.000277777778)
      local minute=math.floor(sec*0.0166666667)-hour*60
      sec=sec-hour*3600-minute*60
      if(sec<10)then sec="0"..sec; end
      if(hour<10)then hour="0"..hour; end
      if(minute<10)then minute="0"..minute; end
      return hour..":"..minute..":"..sec;
    end
    if(sec<10)then sec="0"..sec end
    return "00:00:"..sec;
  end
  function NQueens04:intncmp(lt,rt,size)
    local k,rtn=0,0;
    for k=0,size-1,1 do
      rtn=lt[k]-rt[k];
      if(rtn~=0)then break; end
    end
    return rtn;
  end
  function NQueens04:rotate(trial,scratch,size,neg)
    local k=0;
    local incr=0;
    if neg then k=0; else k=size-1; end
    if neg then incr=1; else incr=-1; end
    local j=0;
    while j<size do
      scratch[j]=trial[k];
      k=k+incr;
      j=j+1;
    end
    if neg then k=size-1; else k=0; end
    local j=0;
    while j<size do
      trial[scratch[j]]=k;
      j=j+1;
      k=k-incr;
    end
  end
  function NQueens04:vMirror(check,size)
    for j=0,size-1,1 do
      check[j]=(size-1)-check[j];
    end
  end
  function NQueens04:symmetryOps(size)
    local trial={};
    local scratch={};
    self.nEquiv=0;
    for k=0,size-1,1 do trial[k]=self.board[k];end
    -- 90
    self:rotate(trial,scratch,size,nil);
    local k=self:intncmp(self.board,trial,size);
    if(k>0)then return 0; end
    if(k==0)then self.nEquiv=1;else
      -- 180
      self:rotate(trial,scratch,size,nil);
      k=self:intncmp(self.board,trial,size);
      if(k>0)then return 0; end
      if(k==0)then self.nEquiv=2; else
        -- 270
        self:rotate(trial,scratch,size,nil);
        k=self:intncmp(self.board,trial,size);
        if(k>0)then return 0; end
        self.nEquiv=4;
      end
    end
    for k=0,size-1,1 do trial[k]=self.board[k];end
    self:vMirror(trial,size);
    k=self:intncmp(self.board,trial,size);
    if(k>0)then return 0; end
    if(self.nEquiv>1)then
      self:rotate(trial,scratch,size,true);
      k=self:intncmp(self.board,trial,size);
      if(k>0)then return 0; end
      if(self.nEquiv>2)then
        self:rotate(trial,scratch,size,true);
        k=self:intncmp(self.board,trial,size);
        if(k>0)then return 0; end
        self:rotate(trial,scratch,size,true);
        k=self:intncmp(self.board,trial,size);
        if(k>0)then return 0;end
      end
    end
    return self.nEquiv * 2;
  end
  function NQueens04:NQueens_rec(size,row)
    if row==size then
      local tst=self:symmetryOps(size);
      if(tst~=0)then
        self.UNIQUE=self.UNIQUE+1;
        self.TOTAL=self.TOTAL+tst;
      end
    else
      for col=0,size-1,1 do
        self.board[row]=col;
        if self.colCheck[col]==nil
          and self.antiCheck[row+col]==nil
          and self.diagCheck[row-col+(size-1)]==nil then
          self.colCheck[col],
          self.antiCheck[row+col],
          self.diagCheck[row-col+(size-1)]=true,true,true;
          self:NQueens_rec(size,row+1);
          self.colCheck[col],
          self.antiCheck[row+col],
          self.diagCheck[row-col+(size-1)]=nil,nil,nil;
        end
      end
    end
  end
  function NQueens04:NQueens()
    local max=15;
    print(" N:            Total       Unique    hh:mm:ss");
    for size=4,max,1 do
      self.TOTAL=0;
      self.UNIQUE=0;
      self.board={};
      self.colCheck={};
      self.diagCheck={};
      self.antiCheck={};
      for k=0,size-1,1 do
        self.board[k]=k;
      end
      s=os.time();
      self:NQueens_rec(size,0);
      print(string.format("%2d:%17d%13d%12s",size,self.TOTAL,self.UNIQUE,self:secstotime(os.difftime(os.time(),s))));
    end
  end
  return setmetatable(this,{__index=NQueens04});
end

NQueens03={};
NQueens03.new=function()
  local this={
    TOTAL=0;
    UNIQUE=0;
    colCheck={};
    diagCheck={};
    antiCheck={};
    board={};
  };
  function NQueens03:secstotime(secs)
    sec=math.floor(secs);
    if(sec>59) then
      local hour=math.floor(sec*0.000277777778)
      local minute=math.floor(sec*0.0166666667)-hour*60
      sec=sec-hour*3600-minute*60
      if(sec<10)then sec="0"..sec end
      if(hour<10)then hour="0"..hour end
      if(minute<10)then minute="0"..minute end
      return hour..":"..minute..":"..sec
    end
    if(sec<10)then sec="0"..sec end
    return "00:00:"..sec
  end
  function NQueens03:NQueens_rec(size,row)
    if row==size then
      self.TOTAL=self.TOTAL+1;
    else
      for col=0,size-1,1 do
        self.board[row]=col;
        if self.colCheck[col]==nil
          and self.antiCheck[row+col]==nil
          and self.diagCheck[row-col+(size-1)]==nil then
          self.colCheck[col],
          self.diagCheck[row-self.board[row]+size-1],
          self.antiCheck[row+self.board[row]]=true,true,true;
          self:NQueens_rec(size,row+1);
          self.colCheck[col],
          self.diagCheck[row-self.board[row]+size-1],
          self.antiCheck[row+self.board[row]]=nil,nil,nil;
        end
      end
    end
  end
  function NQueens03:NQueens()
    local max=15;
    print(" N:            Total       Unique    hh:mm:ss");
    for size=4,max,1 do
      self.TOTAL=0;
      self.UNIQUE=0;
      s=os.time();
      self:NQueens_rec(size,0);
      print(string.format("%2d:%17d%13d%12s",size,self.TOTAL,0,self:secstotime(os.difftime(os.time(),s))));
    end
  end
  return setmetatable(this,{__index=NQueens03});
end

NQueens02={};
NQueens02.new=function()
  local this={
    board={};
    flag={};
    size=8;
    count=1;
  };
  function NQueens02:display()
    for col=0,self.size-1,1 do
      io.write(string.format('%2d',self.board[col]));
    end
    print(" : "..self.count);
    self.count=self.count+1;
  end
  function NQueens02:NQueens(row)
    if row==self.size then
      self:display();
    else
      for col=0,self.size-1,1 do
        if self.flag[col] then
        else 
          self.board[row]=col;
          self.flag[col]=true;
          self:NQueens(row+1);
          self.flag[col]=false;
        end
      end
    end
  end
  return setmetatable(this,{__index=NQueens02});
end

NQueens01={};
NQueens01.new=function()
  local this={
    board={};
    size=8;
    count=1;
  }
  function NQueens01:display()
    for col=0,self.size-1,1 do
      io.write(string.format('%2d',self.board[col]));
    end
    print(" : "..self.count);
    self.count=self.count+1;
  end
  function NQueens01:NQueens(row)
    if row==self.size then
      self:display();
    else
      for col=0,self.size-1,1 do
        self.board[row]=col;
        self:NQueens(row+1);
      end
    end
  end
  return setmetatable(this,{__index=NQueens01});
end

--枝刈り
NQueens08.new():NQueens();
--
--ビットマップ＋対象解除法
--NQueens07.new():NQueens();
--
--ビットマップ
--NQueens06.new():NQueens();
--
--枝刈り
--NQueens05.new():NQueens();
--
--対象解除法
--NQueens04.new():NQueens();
--
--バックトラック
--NQueens03.new():NQueens();
--
--配置フラグ
--NQueens02.new():NQueens(0);
--
--ブルートフォース
--NQueens01.new():NQueens(0);

