self.importScripts('../lib/moment/moment.js', '../lib/moment/moment-precise-range.js', 'include.js');

const MAX = 27;

let TOTAL = 0;
let UNIQUE = 0;

let aBoard = new Array(MAX);
let aT = new Array(MAX);
let aS = new Array(MAX);
let COUNT2, COUNT4, COUNT8;
let BOUND1, BOUND2, TOPBIT,ENDBIT,SIDEMASK,LASTMASK;

let aBoard2 = new Array(MAX);
let SPEED = 0;

function set(bit, row, size) {
  let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
  aBoard2[row] = pos;
  sleep(self.SPEED);
  self.postMessage({status: 'process', box: aBoard2, row: row, size: size});
}

//
//
function rh(a,sz){
  let tmp = 0;
  for(let i = 0; i <= sz; i++){
    if(a&(1<<i)){ return tmp|=(1<<(sz-i)); }
  }
  return tmp;
}
//
function vMirror_bitmap(bf,af,si){
  let score;
  for(let i=0; i < si; i++) {
    score = bf[i];
    af[i] = rh(score, si-1);
  }
}
//
function rotate_bitmap(bf, af, si){
  for(let i=0; i<si; i++){
    let t = 0;
    for(let j = 0; j < si; j++){
      t |= ((bf[j]>>i)&1)<<(si-j-1); // x[j] の i ビット目を
    }
    af[i] = t;                        // y[i] の j ビット目にする
  }
}
//
function intncmp(lt,rt,n){
  let rtn = 0;
  for(let k = 0 ; k < n; k++){
    rtn = lt[k] - rt[k];
    if(rtn != 0){
      break;
    }
  }
  return rtn;
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
function symmetryOps_bitmap(si){
  let nEquiv;
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(let i=0;i<si;i++){
    aT[i] = aBoard[i];
  }
  rotate_bitmap(aT, aS, si);    //時計回りに90度回転
  let k = intncmp(aBoard, aS, si);
  if(k > 0) { return; }
  if(k === 0){
    nEquiv = 2;
  } else {
    rotate_bitmap(aS, aT, si);  //時計回りに180度回転
    k = intncmp(aBoard, aT, si);
    if(k > 0) { return; }
    if(k === 0){
      nEquiv=4;
    } else {
      rotate_bitmap(aT, aS, si);//時計回りに270度回転
      k = intncmp(aBoard, aS, si);
      if(k > 0){ return; }
      nEquiv = 8;
    }
  }
  // 回転・反転・対称チェックのためにboard配列をコピー
  for(let i = 0; i < si; i++){ aS[i] = aBoard[i];}
  vMirror_bitmap(aS,aT,si);   //垂直反転
  k = intncmp(aBoard,aT,si);
  if(k > 0){ return; }
  if(nEquiv > 2){             //-90度回転 対角鏡と同等
    rotate_bitmap(aT,aS,si);
    k = intncmp(aBoard,aS,si);
    if(k > 0){ return; }
    if(nEquiv > 4){           //-180度回転 水平鏡像と同等
      rotate_bitmap(aS, aT, si);
      k = intncmp(aBoard, aT, si);
      if(k > 0){ return;}       //-270度回転 反対角鏡と同等
      rotate_bitmap(aT,aS,si);
      k = intncmp(aBoard,aS,si);
      if(k > 0){ return;}
    }
  }
  if(nEquiv === 2){ COUNT2++; }
  if(nEquiv === 4){ COUNT4++; }
  if(nEquiv === 8){ COUNT8++; }
}

function backTrack2_NR(size,mask,row,left,down,right){
  let bitmap, bit;
  let p = new Array(size);
  let pID = 0;
  let sizeE = size-1;
  let odd = size & 1;
  
  for(let i = 0; i < (1+odd); ++i) {
    bitmap = 0;
    if(0 === i){
      let half = size >> 1;
      bitmap = (1 << half) - 1;
    }else{
      bitmap = 1 << (size >> 1);
    }

    let label = "mais1";
    goto: while (true) {
      switch (label) {
        case 'mais1':
          bitmap = mask &~ (left|down|right);
          set(bitmap, row, size);
          if(row === sizeE){
            if(bitmap){
              if((bitmap&LASTMASK)===0){
                aBoard[row] = bitmap;
                let pos = zeroPadding(bitmap.toString(2), size).split("").indexOf('1');
                aBoard2[row] = pos;
                sleep(self.SPEED);
                self.postMessage({status: 'process', box: aBoard2, row: row+1, size: size});
                symmetryOps_bitmap(size);
              }
            }
          } else {
            //【枝刈り】上部サイド枝刈り
            if(row<BOUND1){
              bitmap&=~SIDEMASK;
              set(bitmap, row, size);
              //【枝刈り】下部サイド枝刈り
            }else if(row==BOUND2){
              if(!(down&SIDEMASK)) {
                label = 'volta';
                continue goto;
              }
              if((down&SIDEMASK)!=SIDEMASK) {
                bitmap&=SIDEMASK;
                set(bitmap, row, size);
              }
            }
            if(bitmap){
              set(bitmap, row, size);
              label = 'outro';
              continue goto;
            }
          }
          label = 'volta';
          continue goto;
        case 'outro':
          bitmap ^= aBoard[row] = bit = -bitmap&bitmap;
          set(bit, row, size);
          if(bitmap){
            p[pID++] = left;
            p[pID++] = down;
            p[pID++] = right;
          }
          p[pID++] = bitmap;
          row++;
          left=(left|bit)<<1;
          down=down|bit;
          right=(right|bit)>>1;

          label = 'mais1';
          continue goto;
        case 'volta':
          if(pID < 0) { break goto; }
          row--;
          bitmap = p[--pID];
          if(bitmap){
            right = p[--pID];
            down = p[--pID];
            left = p[--pID];
            label = 'outro';
            continue goto;
          } else {
            label = 'volta';
            continue goto;
          }
          break;
        default:
          break;
      }
    }
    set(1, row, size);
  }
}

function backTrack1_NR(size,mask,row,left,down,right){
  let bitmap, bit;
  let p = new Array(size);
  let pID = 0;
  let sizeE = size - 1;
  let odd = size & 1;
  
  for(let i = 0; i < (1+odd); ++i) {
    bitmap = 0;
    if(0 === i){
      let half = size >> 1;
      bitmap = (1 << half) - 1;
    }else{
      bitmap = 1 << (size >> 1);
    }

    let label = "b1mais1";
    goto: while (true) {
      switch (label) {
        case 'b1mais1':
          bitmap = mask &~ (left|down|right);
          set(bitmap, row, size);
          if(row === sizeE){
            if(bitmap){
              let pos = zeroPadding(bitmap.toString(2), size).split("").indexOf('1');
              aBoard2[row] = pos;
              sleep(self.SPEED);
              self.postMessage({status: 'process', box: aBoard2, row: row+1, size: size});
              symmetryOps_bitmap(size);
              COUNT8++;
            }
          } else {
            if(row < BOUND1){
              bitmap&=~2;
              set(bitmap, row, size);
            }
            if(bitmap){
              label = 'b1outro';
              continue goto;
            }
          }
          label = 'b1volta';
          continue goto;
        case 'b1outro':
          bitmap ^= aBoard[row] = bit = -bitmap&bitmap;
          set(bit, row, size);
          if(bitmap){
            p[pID++] = left;
            p[pID++] = down;
            p[pID++] = right;
          }
          p[pID++] = bitmap;
          row++;
          left=(left|bit)<<1;
          down=down|bit;
          right=(right|bit)>>1;

          label = 'b1mais1';
          continue goto;
        case 'b1volta':
          if(pID < 0) { break goto; }
          row--;
          bitmap = p[--pID];
          if(bitmap){
            right = p[--pID];
            down = p[--pID];
            left = p[--pID];
            label = 'b1outro';
            continue goto;
          } else {
            label = 'b1volta';
            continue goto;
          }
        default:
          break;
      }
    }
    set(1, row, size);
  }
}
//EOS1
function NQueen(size, mask){
  let bit;
  TOPBIT = 1 << (size - 1);
  aBoard[0] = 1;
  set(1, 0, size);
  for(BOUND1 = 2; BOUND1 < size - 1; BOUND1++){
    aBoard[1] = bit = (1 << BOUND1);
    set(bit, 1, size);
    backTrack1_NR(size, mask, 2, (2|bit)<<1, (1|bit), (bit>>1));
  }
  SIDEMASK = LASTMASK = (TOPBIT | 1);
  ENDBIT = (TOPBIT >> 1);
  for(BOUND1 = 1, BOUND2 = size - 2; BOUND1<BOUND2; BOUND1++, BOUND2--){
    aBoard[0] = bit = (1 << BOUND1);
    set(bit, 0, size);
    backTrack2_NR(size, mask, 1, bit<<1, bit, bit>>1);
    LASTMASK |= LASTMASK >>1 | LASTMASK << 1;
    ENDBIT >>= 1;
  }
}
//EOS1

function backTrack2(size,mask,row,left,down,right){
  let bit;
  let bitmap = mask &~ (left|down|right); /* 配置可能フィールド */
  if(row === size - 1){
    if(bitmap){
      if((bitmap&LASTMASK) === 0){   
        aBoard[row] = bitmap; //symmetryOpsの時は代入します。
        symmetryOps_bitmap(size);

        let pos = zeroPadding(bitmap.toString(2), size).split("").indexOf('1');
        aBoard2[row] = pos;
        sleep(self.SPEED);
        self.postMessage({status: 'process', box: aBoard2, row: row+1, size: size});
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
      if((down&SIDEMASK) ===0){ return; }
      if((down&SIDEMASK)!== SIDEMASK){ bitmap&=SIDEMASK; }
    }
    while(bitmap){
      bitmap ^= aBoard[row] = bit = (-bitmap&bitmap); //最も下位の１ビットを抽出
      set(bit, row, size);
      backTrack2(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}
function backTrack1(size,mask,row,left,down,right){
  let bit;
  let bitmap = mask &~ (left|down|right);
  if(row === size - 1){
    if(bitmap) {
      COUNT8++;
      let pos = zeroPadding(bitmap.toString(2), size).split("").indexOf('1');
      aBoard2[row] = pos;
      sleep(self.SPEED);
      self.postMessage({status: 'process', box: aBoard2, row: row+1, size: size});
    } else {
      set(bitmap, row, size);
    }
  }else{
    if(row < BOUND1) {
      bitmap&=~2;
    }
    while(bitmap){
      bitmap ^= aBoard[row] = bit = (-bitmap&bitmap); //ロジック用
      set(bit, row, size);
      backTrack1(size,mask,row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
    }
  }
}

//EOS2
function NQueenR(size,mask){
  let bit;
  TOPBIT=1<<(size-1);
  aBoard[0] = 1;
  set(1, 0, size);
  for(BOUND1 = 2; BOUND1 < size-1; BOUND1++){
    aBoard[1] = bit = (1<<BOUND1);
    set(bit, 1, size);
    backTrack1(size, mask, 2, (2|bit)<<1, (1|bit), (bit>>1));
  }
  SIDEMASK = LASTMASK = (TOPBIT|1);
  ENDBIT = (TOPBIT>>1);
  for(BOUND1 = 1, BOUND2 = size - 2; BOUND1 < BOUND2; BOUND1++, BOUND2--){
    aBoard[0] = bit = (1<<BOUND1);
    set(bit, 0, size);
    backTrack2(size, mask, 1, bit<<1, bit, bit>>1);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT >>= 1;
  }
}
//EOS2

function main(size, mode = 1){
  let from = new Date();
  let min = 4;
  let targetN = size;
  let mask = 0;

  aBoard2 = new Array(size);
  aBoard2.fill(-1);

  aT.fill(-1);
  aS.fill(-1);

  for(let i = targetN; i <= targetN; i++){
    COUNT2 = COUNT4 = COUNT8 = 0;
    mask = ((1<<i)-1);
    if(mode == 1) {
      self.NQueen(i, mask);
    } else {
      self.NQueenR(i, mask);
    }
    self.postMessage({status: 'process', result: `n:${i}\t\tTotal:${self.getTotal()}\t\tUnique:${self.getUnique()}\t\ttime:${timeFormat(from)}`});
  }
  self.postMessage({status: 'success', result: '' });
}

self.addEventListener('message', (msg) => {
  self.SPEED = Number(msg.data.speed) * 1000;
  if(msg.data.size) {
    main(Number(msg.data.size), Number(msg.data.mode));
  }
});

function zeroPadding(NUM, LEN){
  return ( Array(LEN).join('0') + NUM ).slice( -LEN );
}