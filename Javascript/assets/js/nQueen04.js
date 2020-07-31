self.importScripts('../lib/moment/moment.js', '../lib/moment/moment-precise-range.js', 'include.js');

const MAX = 27;

let aBoard = new Array([MAX]);
let down = new Array(2*MAX-1);
let right = new Array(2*MAX-1);
let left = new Array(2*MAX-1);
down.fill(0);
right.fill(0);
left.fill(0);
let TOTAL = 0;
let UNIQUE = 0;
let SPEED = 0;
let aT = new Array([MAX]);
let aS = new Array([MAX]);

// function print(size) {
//   let out = `${TOTAL}:\t`;
//   for(let j = 0; j < size; j++){
//     out += `${aBoard[j]}`;
//   }
//   self.postMessage({status: 'process', result: out});
// }

function symmetryOps(size) {
  let nEquiv;
  for(let i = 0; i < size; i++) {
    aT[i] = aBoard[i];
  }
  rotate(aT, aS, size, 0);
  let k = intncmp(aBoard, aT, size);
  if(k > 0) return 0;
  if(k == 0) {
    nEquiv = 1;
  } else {
  rotate(aT, aS, size, 0);
    k = intncmp(aBoard, aT, size);
    if(k > 0) return 0;
    if(k == 0) {
      nEquiv = 2;
    } else {
      rotate(aT, aS, size, 0);
      k = intncmp(aBoard, aT, size);
      if(k > 0) {
        return 0;
      }
      nEquiv = 4;
    }
  }
  for(let i = 0; i < size; i++) {
    aT[i] = aBoard[i];
  }
  vMirror(aT, size);
  k = intncmp(aBoard, aT, size);
  if(k > 0) {
    return 0;
  }
  if(nEquiv > 1) {
    rotate(aT, aS, size, 1);
    k = intncmp(aBoard, aT, size);
    if(k > 0) {
      return 0;
    }
    if(nEquiv > 2) {
      rotate(aT, aS, size, 1);
      k = intncmp(aBoard, aT, size);
      if(k > 0) {
        return 0;
      }
      rotate(aT, aS, size, 1);
      k = intncmp(aBoard, aT, size);
      if(k > 0) {
        return 0;
      }
    }
  }
  return nEquiv * 2;
}

//EOS1
function NQueen(row, size) {
  let sizeE = size - 1;
  let matched;
  while(row >= 0) {
    matched = false;
    for(let col = aBoard[row] + 1; col < size; col++) {
      if(down[col] == 0 && right[col-row+sizeE] == 0 && left[col+row] == 0) {
        if(aBoard[row] != -1) {
          down[aBoard[row]] = right[aBoard[row]-row+sizeE] = left[aBoard[row]+row] = 0;
        }
        aBoard[row] = col;
        down[col] = right[col-row+sizeE] = left[col+row] = 1;
        matched = true;
        break;
      }
    }

    if(matched) {
      row++;
      if(row == size) {
        let s = symmetryOps(size);
        if(s != 0) {
          self.UNIQUE++;
          self.TOTAL += s;

          sleep(self.SPEED);
          self.postMessage({status: 'process', box: aBoard, row: row, size: size});
        }
        row--;
      }
    } else {
      if(aBoard[row] != -1) {
        let col = aBoard[row];
        down[col] = right[col-row+sizeE] = left[col+row] = 0;
        aBoard[row] = -1;
      } else {
      sleep(self.SPEED);
      self.postMessage({status: 'process', box: aBoard, row: row, size: size});
      }
      row--;
    }
  }
}
//EOS1

//EOS2
function NQueenR(row, size) {
  let sizeE = size - 1;
  if(row == size) {
    let s = symmetryOps(size);
    if(s != 0) {
      self.UNIQUE++;
      self.TOTAL += s;
      sleep(self.SPEED);
      self.postMessage({status: 'process', box: aBoard, row: row, size: size});
    }
  } else {
    sleep();
    self.postMessage({status: 'process', box: aBoard, row: row, size: size});
    for(let col = aBoard[row] + 1; col < size; col++) {
      aBoard[row] = col;
      if(down[col] == 0 && right[row-col+sizeE] == 0 && left[row+col] == 0) {
        down[col] = right[row-col+sizeE] = left[row+col] = 1;
        NQueenR(row + 1, size);
        down[col] = right[row-col+sizeE] = left[row+col] = 0;
      }
      aBoard[row] = -1;
    }
  }
}
//EOS2

function main(size, mode = 1){
  let from = new Date();
  let min = 4;
  let targetN = size;
  for(let i = targetN; i <= targetN; i++){
    self.TOTAL = 0;
    self.UNIQUE = 0;
    for(let j = 0; j < targetN; j++){ aBoard[j]=-1; }
    if(mode == 1) {
      self.NQueen(0, i);
    } else {
      self.NQueenR(0, i);
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