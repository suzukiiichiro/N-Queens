self.importScripts('../lib/moment/moment.js', '../lib/moment/moment-precise-range.js', 'include.js');

const MAX = 27;


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
  sleep(this.SPEED);
  this.postMessage({status: 'process', box: aBoard2, row: row, size: size});
}

function set2(box, row, size) {
  sleep(this.SPEED);
  this.postMessage({status: 'process', box: box, row: row, size: size});
}

function zeroPadding(NUM, LEN){
  return ( Array(LEN).join('0') + NUM ).slice( -LEN );
}


//EOS1

//EOS1

let jobs = [];
//EOS2
function NQueenR(size, mask){
  let bit;
  TOPBIT=1<<(size-1);
  aBoard[0] = 1;
  set(1, 0, size);
  for(BOUND1 = 2; BOUND1 < size-1; BOUND1++){
    aBoard[1] = bit = (1<<BOUND1);
    let worker = new Worker('nQueen13_thread.js');
    let promise = new Promise((resolve, reject) => {
      worker.addEventListener('message', (msg) => {
        this.output(msg.data, 'bt1');
        if(msg.data.status === 'end') {
          resolve(msg.data);
        } else if(msg.data.status !== 'total'){
          set2(msg.data.box, msg.data.row, msg.data.size);
        }
      });
    });
    
    set(bit, 1, size);
    let data = {size: size, mask: mask, row: 2, left: (2|bit)<<1, down: (1|bit), right: (bit>>1), mode: "bt1", aBoard: aBoard, aBoard2: aBoard2, BOUND1: BOUND1, BOUND2: BOUND2, TOPBIT: TOPBIT, ENDBIT: ENDBIT, SIDEMASK: SIDEMASK, LASTMASK: LASTMASK, COUNT2: COUNT2, COUNT4: COUNT4, COUNT8: COUNT8};
    jobs.push(promise);
    worker.postMessage(data);
  }
  SIDEMASK = LASTMASK = (TOPBIT|1);
  ENDBIT = (TOPBIT>>1);
  for(BOUND1 = 1, BOUND2 = size - 2; BOUND1 < BOUND2; BOUND1++, BOUND2--){

    let worker = new Worker('nQueen13_thread.js');
    let promise = new Promise((resolve, reject) => {
      worker.addEventListener('message', (msg) => {
        this.output(msg.data, 'bt2');
        if(msg.data.status === 'end') {
          resolve(msg.data);
        } else if(msg.data.status !== 'total'){
          set2(msg.data.box, msg.data.row, msg.data.size);
        }
      });
    });

    aBoard[0] = bit = (1<<BOUND1);
    set(bit, 0, size);
    let data = {size: size, mask: mask, row: 1, left: bit<<1, down: bit, right: bit>>1, mode: "bt2", aBoard: aBoard, aBoard2: aBoard2, BOUND1: BOUND1, BOUND2: BOUND2, TOPBIT: TOPBIT, ENDBIT: ENDBIT, SIDEMASK: SIDEMASK, LASTMASK: LASTMASK, COUNT2: COUNT2, COUNT4: COUNT4, COUNT8: COUNT8};
    jobs.push(promise);
    worker.postMessage(data);
    LASTMASK|=LASTMASK>>1|LASTMASK<<1;
    ENDBIT >>= 1;
  }
}
//EOS2


let outputTimer = null;
let currentSize = 0;
let response = { "bt1": {"total": 0, "unique": 0}, "bt2": {"total": 0, "unique": 0}};
function output(data, mode) {
  if(data.status === 'total') {
    response[mode]["total"] = response[mode]["total"] < data.total ? data.total : response[mode]["total"];
    response[mode]["unique"] = response[mode]["unique"] < data.unique ? data.unique : response[mode]["unique"];
  }
  if(outputTimer != null) { clearTimeout(outputTimer); }
  outputTimer = setTimeout(function(){
    let total = response["bt1"]["total"] + response["bt2"]["total"];
    let unique = response["bt1"]["unique"] + response["bt2"]["unique"];
    this.postMessage({status: 'process', result: `n:${currentSize}\t\tTotal:${total}\t\tUnique:${unique}\t\ttime:${this.timeFormat(from)}`});
    this.postMessage({status: 'success', result: '' });
  }, 100);
}

let from = null;
function main(size, mode = 1){

  from = new Date();
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
    jobs = [];
    outputTimer = null;
    currentSize = i;
    // if(mode == 1) {
      // self.NQueen(i, mask);
    // } else {
      // self.NQueenR(i, mask);
    // }
    _self.NQueenR(i, mask);
  }
}

let _self = this;
_self.addEventListener('message', (msg) => {
  _self.SPEED = Number(msg.data.speed) * 1000;
  if(msg.data.size) {
    _self.main(Number(msg.data.size), Number(msg.data.mode));
  }
});