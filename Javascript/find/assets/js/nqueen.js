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

let current_table = [];
let current_row = [];

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


function print(size) {
  let out = `${TOTAL}:\t`;
  for(let j = 0; j < size; j++){
    out += `${aBoard[j]}`;
  }
  self.postMessage({status: 'process', result: out});
}

function NQueen(row, size) {
  let sizeE = size - 1;
  let matched;
  // console.log(this.current_table);
    // console.log("exit");
    // aBoard[row] = size - this.current_table[row];
//     NQueenR(row + 1, size);
  while(row >= 0) {
    this.current_row = row;
    console.log(`NQueenR:${this.current_table[row]}`);
    if(this.current_table[row] !== undefined) {
      row++;
      if(row == size) {
        self.TOTAL++;
        row--;
      }
      continue;
    }
    matched = false;
    for(let col = aBoard[row]+1; col < size; col++) {
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
    sleep(self.SPEED);
    self.postMessage({status: 'process', box: aBoard, row: matched ? row + 1 : row, size: size});
    if(matched) {
      row++;
      if(row == size) {
        self.TOTAL++;
        row--;
      }
    } else {
      if(aBoard[row] != -1) {
        let col = aBoard[row];
        down[col] = right[col - row + sizeE] = left[col + row] = 0;
        aBoard[row] = -1;
      }
      row--;
    }
  }
}
//EOS2
function NQueenR(row, size) {
  this.current_row = row;
  let sizeE = size - 1;
  let postFlag = this.current_table[row] !== undefined ? true : true;
  if(row == size) {
    self.TOTAL++;
    // if(postFlag) {
      sleep(self.SPEED);
      self.postMessage({status: true, box: aBoard, row: row, size: size});
    // }
  } else {
    if(postFlag) {
      sleep(self.SPEED);
      self.postMessage({status: true, box: aBoard, row: row, size: size});
    }
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

function NQueenR5(row, size) {
  this.current_row = row;
  let sizeE = size - 1;
  let postFlag = this.current_table[row] !== undefined ? true : true;
  if(row === size) {
    let s = symmetryOps(size);
    if(s !== 0) {
      self.UNIQUE++;
      self.TOTAL += s;
      if(self.crow < row) {
        sleep(self.SPEED);
        self.postMessage({status: true, box: aBoard, row: row, size: size});
      }
    }
  } else {
    let lim = (row != 0) ? size : (size + 1) / 2;
    if(self.  crow < row) {
      if(postFlag) {
        sleep(self.SPEED);
        self.postMessage({status: true, box: aBoard, row: row, size: size});
      }
    }
    // sleep(self.SPEED);
    // self.postMessage({status: true, box: aBoard, row: row, size: size});
    for(let col = aBoard[row] + 1; col < lim; col++) {
      aBoard[row] = col;
      if(down[col] === 0 && right[row-col+sizeE] === 0 && left[row+col] == 0) {
        down[col] = right[row-col+sizeE] = left[row+col] = 1;
        NQueenR5(row + 1, size);
        down[col] = right[row-col+sizeE] = left[row+col] = 0;
      }
      aBoard[row] = -1;
    }
  }
}

function main(size){
  let from = new Date();
  self.TOTAL = 0;
  self.UNIQUE = 0;
  for(let j = 0; j < size; j++){ aBoard[j]=-1; }
  if(self.crow != -1) {
    self.NQueenR5(0, size);
  }
  // self.NQueenR(0, size);
  self.postMessage({status: false, result: `Total:${self.TOTAL}\t\tUnique:${self.UNIQUE}\t\ttime:${timeFormat(from)}`});
  // self.postMessage({status: 'success', result: '' });
}

let crow = -1;
self.addEventListener('message', (msg) => {
  self.SPEED = Number(msg.data.speed);
  self.crow = msg.data.row.indexOf(undefined) > -1 ? msg.data.row.indexOf(undefined) : -1;
  if(msg.data.size) {
    this.current_table = msg.data.row.concat();
    main(Number(msg.data.size));
  }
});