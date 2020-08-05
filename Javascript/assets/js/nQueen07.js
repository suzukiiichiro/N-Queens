self.importScripts('../lib/moment/moment.js', '../lib/moment/moment-precise-range.js', 'include.js');

const MAX = 27;

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
  if(k == 0){
    nEquiv = 2;
  } else {
    rotate_bitmap(aS, aT, si);  //時計回りに180度回転
    k = intncmp(aBoard, aT, si);
    if(k > 0) { return; }
    if(k == 0){
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
  if(nEquiv==2){ COUNT2++; }
  if(nEquiv==4){ COUNT4++; }
  if(nEquiv==8){ COUNT8++; }
}
//EOS1
function NQueen(size, mask, row) {
  let aStack = new Array(size);
  let pnStack = new Array(size);
  let bit;
  let bitmap;
  let sizeE = size - 1;
  let down = new Array(size);
  let right = new Array(size);
  let left = new Array(size);
  aStack[0] = -1;
  pnStack.fill(0);
  let pnStackID = 0;
  bit = 0;
  bitmap = mask;
  down[0] = left[0] = right[0] = 0;

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
//EOS1

//EOS2
function NQueenR(size, mask, row, left, down, right) {
  let bitmap = 0;
  let bit = mask&~(left|down|right);
  if(row == size) { 
    symmetryOps_bitmap(size);
    sleep(self.SPEED);
    self.postMessage({status: 'process', box: aBoard2, row: row, size: size});
  } else {
    bitmap = (mask&~(left|down|right));
    while(bitmap) {
      bitmap ^= aBoard[row] = bit = (-bitmap&bitmap);
      let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
      aBoard2[row] = pos;
      sleep(self.SPEED);
      self.postMessage({status: 'process', box: aBoard2, row: row, size: size});
      NQueenR(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1);
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

  for(let i = targetN; i <= targetN; i++){
    COUNT2 = COUNT4 = COUNT8 = 0;
    mask = ((1<<i)-1);
    if(mode == 1) {
      self.NQueen(i, mask, 0);
    } else {
      self.NQueenR(i,mask,0,0,0,0);
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