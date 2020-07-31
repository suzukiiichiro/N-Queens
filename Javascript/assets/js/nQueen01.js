const MAX = 17;
let COUNT = 0;
let aBoard = new Array([MAX]);

function print(size) {
  let out = `${++COUNT}:\t`;
  for(let j = 0; j < size; j++){
    out += `${aBoard[j]}`;
  }
  self.postMessage({status: 'process', result: out});
}

//EOS1
function NQueen(row, size) {
  let matched;
  while(row >= 0) {
    matched = false;
    for(let col = aBoard[row]+1; col < size; col++) {
      aBoard[row] = col;
      matched = true;
      break;
    }
    if(matched) {
      row++;
      if(row == size) {
        self.print(size);
        row--;
      }
    } else {
      if(aBoard[row] != -1) {
        aBoard[row] = -1;
      }
      row--;
    }
  }
  self.postMessage({status: 'success', result: '' });
}
//EOS1

//EOS2
function NQueenR(row, size) {
  if(row == size) {
    self.print();
  } else {
    for(let col = aBoard[row] + 1; col < size; col++) {
      aBoard[row] = col;
      NQueenR(row + 1, size);
      aBoard[row] = -1;
    }
  }
  self.postMessage({status: 'success', result: '' });
}
//EOS2

function main(size, mode = 1){
  COUNT = 0;
  for(let i = 0; i < size; i++){ aBoard[i]=-1; }
  if(mode == 1) {
    self.NQueen(0, size);
  } else {
    self.NQueenR(0, size);
  }
}

self.addEventListener('message', (msg) => {
  main(msg.data.size, msg.data.mdoe);
});