let aBoard = [];
let COUNT2, COUNT4, COUNT8;
let BOUND1, BOUND2, TOPBIT,ENDBIT,SIDEMASK,LASTMASK;
let aBoard2 = [];

function set(bit, row, size) {
  let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
  aBoard2[row] = pos;
  this.postMessage({status: 'total', total: getTotal(), unique: getUnique()});
  this.postMessage({status: 'process', box: aBoard2, row: row, size: size});
}
function zeroPadding(NUM, LEN){
  return ( Array(LEN).join('0') + NUM ).slice( -LEN );
}
//
function getUnique(){
  return COUNT2 + COUNT4 + COUNT8;
}
//
function getTotal(){
  return COUNT2*2 + COUNT4*4 + COUNT8*8;
}
//
function symmetryOps(si){
  let own,ptn,you,bit;
  //90度回転
  if(aBoard[BOUND2] === 1){ own=1; ptn=2;
    while(own<=si-1){ bit=1; you=si-1;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you--; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn<<=1;
    }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(own>si-1){ COUNT2++; return; }
  }
  //180度回転
  if(aBoard[si-1]===ENDBIT){ own=1; you=si-1-1;
    while(own<=si-1){ bit=1; ptn=TOPBIT;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; ptn>>=1; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; you--;
    }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(own>si-1){ COUNT4++; return; }
  }
  //270度回転
  if(aBoard[BOUND1]===TOPBIT){ own=1; ptn=TOPBIT>>1;
    while(own<=si-1){ bit=1; you=0;
      while((aBoard[you]!=ptn)&&(aBoard[own]>=bit)){ bit<<=1; you++; }
      if(aBoard[own]>bit){ return; } if(aBoard[own]<bit){ break; }
      own++; ptn>>=1;
    }
  }
  COUNT8++;
}

function backTrack2(size,mask,row,left,down,right){
  let bit;
  let bitmap = mask &~ (left|down|right); /* 配置可能フィールド */
  console.log(bitmap);
  if(row === size - 1){
    if(bitmap){
      if((bitmap&LASTMASK) === 0){   
        aBoard[row] = bitmap; //symmetryOpsの時は代入します。
        symmetryOps(size);
        let pos = zeroPadding(bitmap.toString(2), size).split("").indexOf('1');
        aBoard2[row] = pos;
        this.postMessage({status: 'total', total: getTotal(), unique: getUnique()});
        this.postMessage({status: 'find', box: aBoard2, row: row+1, size: size});
      } else {
        set(bitmap, row, size);
      }
    }
  }else{
    //【枝刈り】上部サイド枝刈り
    if(row<BOUND1){
      bitmap&=~SIDEMASK;
      //【枝刈り】下部サイド枝刈り
    }else if(row === BOUND2) {
      if((down&SIDEMASK) ===0){
        this.postMessage({status: 'total', total: getTotal(), unique: getUnique()});
        return;
      }
      if((down&SIDEMASK)!== SIDEMASK){ bitmap&=SIDEMASK; }
    }
    while(bitmap){
      bitmap ^= aBoard[row] = bit = (-bitmap&bitmap); //最も下位の１ビットを抽出
      set(bit, row, size);
      backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  this.postMessage({status: 'total', total: getTotal(), unique: getUnique()});
  this.postMessage({status: 'end'});
}
function backTrack1(size,mask,row,left,down,right){
  let bit;
  let bitmap = mask &~ (left|down|right);
  if(row === size - 1){
    if(bitmap) {
      COUNT8++;
      let pos = zeroPadding(bitmap.toString(2), size).split("").indexOf('1');
      aBoard2[row] = pos;
      this.postMessage({status: 'total', total: getTotal(), unique: getUnique()});
      this.postMessage({status: 'find', box: aBoard2, row: row+1, size: size});
    } else {
      set(bitmap, row, size);
    }
  }else{
    if(row < BOUND1) {
      bitmap&=~2;
    }
    while(bitmap){
      bitmap ^= aBoard[row] = bit = (-bitmap&bitmap); //ロジック用
      set(bitmap, row, size);
      backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
  this.postMessage({status: 'total', total: getTotal(), unique: getUnique()});
  this.postMessage({status: 'end'});
}

this.onmessage = (msg)=> {
  aBoard = msg.data.aBoard;
  COUNT2 = msg.data.COUNT2;
  COUNT4 = msg.data.COUNT4;
  COUNT8 = msg.data.COUNT8;
  BOUND1 = msg.data.BOUND1;
  BOUND2 = msg.data.BOUND2;
  TOPBIT = msg.data.TOPBIT;
  ENDBIT = msg.data.ENDBIT;
  SIDEMASK = msg.data.SIDEMASK;
  LASTMASK = msg.data.LASTMASK;
  aBoard2 = msg.data.aBoard2;
  if(msg.data.mode == "bt1") {
    backTrack1(msg.data.size, msg.data.mask, msg.data.row, msg.data.left, msg.data.down, msg.data.right);
  } else if(msg.data.mode == "bt2") {
    backTrack2(msg.data.size, msg.data.mask, msg.data.row, msg.data.left, msg.data.down, msg.data.right);
  }
}; 