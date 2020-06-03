self.importScripts('../lib/moment/moment.js', '../lib/moment/moment-precise-range.js', 'include.js');

const MAX = 27;

let down = new Array(2*MAX-1);
let right = new Array(2*MAX-1);
let left = new Array(2*MAX-1);

let TOTAL = 0;
let UNIQUE = 0;
let aT = new Array(MAX);
let aS = new Array(MAX);
let aBoard = new Array(MAX);
let aBoard2 = new Array(MAX);
let COUNT2, COUNT4, COUNT8;

let SPEED = 0;

function set(bit, row, size) {
  let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
  aBoard2[row] = pos;
  sleep(self.SPEED);
  self.postMessage({status: 'process', box: aBoard2, row: row, size: size});
}

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
  for(let i=0;i<si;i++){ aT[i] = aBoard[i];}
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
//EOS1
function NQueen(size, mask) {
  let aStack = new Array(size);
  let pnStack = new Array(size);
  let row = 0;
  let bit;
  let bitmap;
  let odd = size & 1;
  let sizeE = size - 1;
  let pnStackID = 0;
  pnStack.fill(0);

  aStack[0] = -1;

  for(let i = 0; i < (1+odd); ++i) {
    bitmap=0;
    if(0 == i){
      let half=size>>1; // size/2
      bitmap=(1<<half)-1;
      pnStack[pnStackID] = aStack[row]+1;
    }else{
      bitmap=1<<(size>>1);
      down[1]=bitmap;
      right[1]=(bitmap>>1);
      left[1]=(bitmap<<1);
      pnStack[pnStackID] = aStack[row]+1;
      pnStack[pnStackID++]=0;
    }

    while(true) {
      if(bitmap) {
        bitmap ^= aBoard[row] = bit = (-bitmap&bitmap);
        if(row == sizeE) {
          // self.TOTAL++;
          let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
          aBoard2[row] = pos;
          sleep(self.SPEED);
          self.postMessage({status: 'process', box: aBoard2, row: row+1, size: size});

          symmetryOps_bitmap(size);
          bitmap = pnStack[--pnStackID];
          --row;
          continue;
        } else {
          set(bit, row, size);
          let n = row++;
          left[row] = (left[n] | bit) << 1;
          down[row] = down[n] | bit;
          right[row] = (right[n] | bit) >> 1;
          pnStack[pnStackID++] = bitmap;
          bitmap=mask&~(left[row]|down[row]|right[row]);
          continue;
        }
      } else {
        set(bit, row, size);
        bitmap = pnStack[--pnStackID];
        if(pnStack[pnStackID] == aStack[row]) { break; }
        --row;
        continue;
      }
    }
  }
}
//EOS1

//EOS2
function NQueenR(size, mask, row, left, down, right, ex1, ex2) {
  let bit;
  let bitmap = (mask&~(left|down|right|ex1));
  if(row == size) { 
    symmetryOps_bitmap(size);
    // sleep(self.SPEED);
    self.postMessage({status: 'process', box: aBoard2, row: row, size: size});
  } else {
    while(bitmap) {
      if(ex2 !== 0) {
        bitmap ^= aBoard[row] = bit =(1<<(size/2+1));
      } else {
        bitmap ^= aBoard[row] = bit =(-bitmap&bitmap);
      }
      let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
      aBoard2[row] = pos;
      sleep(self.SPEED);
      self.postMessage({status: 'process', box: aBoard2, row: row, size: size});
      NQueenR(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1, ex2, 0);
      ex2 = 0;
    }
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

  let excl;

  for(let i = targetN; i <= targetN; i++){
    COUNT2 = COUNT4 = COUNT8 = 0;
    mask = ((1<<i)-1);
    excl = (1<<((i/2)^0))-1;
    if(i%2){
     excl=excl<<(i/2+1);
    }else{
     excl=excl<<(i/2);
    }
    if(mode == 1) {
      self.NQueen(i, mask);
    } else {
      self.NQueenR(i,mask,0,0,0,0, excl, i%2 ? excl : 0);
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