self.importScripts('../lib/moment/moment.js', '../lib/moment/moment-precise-range.js', 'include.js');

const MAX = 27;

let TOTAL = 0;
let UNIQUE = 0;
let aT = new Array([MAX]);
let aS = new Array([MAX]);

let SPEED = 0;

function set(bit, row, size) {
  let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
  aBoard[row] = pos;
  sleep(self.SPEED);
  self.postMessage({status: 'process', box: aBoard, row: row, size: size});
}
//EOS1
function NQueen(size, mask, row) {
  let aStack = new Array([size]);
  let pnStack = new Array([size]);
  let bit;
  let bitmap;
  let sizeE = size - 1;
  let down = new Array([size]);
  let right = new Array([size]);
  let left = new Array([size]);
  aStack[0] = -1;
  pnStack.fill(0);
  let pnStackID = 0;
  bit = 0;
  bitmap = mask;
  down[0] = left[0] = right[0] = 0;

  while(true) {
    if(bitmap) {
      bitmap ^= bit = (-bitmap&bitmap);
      if(row == sizeE) {
        self.TOTAL++;

        let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
        aBoard[row] = pos;
        sleep(self.SPEED);
        self.postMessage({status: 'process', box: aBoard, row: row+1, size: size});

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
  let bit = 0;
  if(row == size) {
    self.TOTAL++;
    // sleep(self.SPEED);
    self.postMessage({status: 'process', box: aBoard, row: row, size: size});
  } else {
    bitmap = (mask&~(left|down|right));
    while(bitmap) {
      bitmap ^= bit = (-bitmap&bitmap);
      /*
        配置可能なパターンを取得
      http://www.ic-net.or.jp/home/takaken/nt/queen
      */
      let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
      aBoard[row] = pos;
      sleep(self.SPEED);
      self.postMessage({status: 'process', box: aBoard, row: row, size: size});
      NQueenR(size, mask, row+1, (left|bit)<<1, down|bit, (right|bit)>>1);
    }
  }
}
//EOS2

var aBoard = [];
function main(size, mode = 1){
  let from = new Date();
  let min = 4;
  let targetN = size;
  let mask = 0;

  aBoard = new Array(size);
  aBoard.fill(-1);
  
  for(let i = targetN; i <= targetN; i++){
    self.TOTAL = 0;
    self.UNIQUE = 0;
    mask = ((1<<i)-1);
    // for(let j = 0; j < targetN; j++){ aBoard[j]=-1; }
    if(mode == 1) {
      self.NQueen(i, mask, 0);
    } else {
      self.NQueenR(i,mask,0,0,0,0);
    }

    self.postMessage({status: 'process', result: `n:${i}\t\tTotal:${self.TOTAL}\t\tUnique:${self.UNIQUE}\t\ttime:${timeFormat(from)}`});
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