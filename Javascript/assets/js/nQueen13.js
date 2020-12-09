self.importScripts('../lib/moment/moment.js', '../lib/moment/moment-precise-range.js', 'include.js');

const MAX = 27;

let TOTAL = 0;
let UNIQUE = 0;

var aBoard2 = new Array(MAX);
let SPEED = 0;

function set(bit, row, size) {
  let pos = zeroPadding(bit.toString(2), size).split("").indexOf('1');
  aBoard2[row] = pos;
  sleep(self.SPEED);
  self.postMessage({status: 'process', box: aBoard2, row: row, size: size});
}

function zeroPadding(NUM, LEN){
  return ( Array(LEN).join('0') + NUM ).slice( -LEN );
}

//EOS1
//EOS1

//EOS2
var children = [];
class WorkingEngine {
  constructor() {
    this.board = null;
    this.MASK = null;
    this.size = null;
    this.sizeE = null;
    this.TOPBIT = null;
    this.ENDBIT = null;
    this.SIDEMASK = null;
    this.LASTMASK = null;
    this.BOUND1 = null;
    this.BOUND2 = null;
    this.B1 = null;
    this.B2  = null;
    this.info = null;
    this.nMore = null;
    this.bThread =true;

  }
  WorkingEngine(size, nMore, info, B1, B2) {
    this.size = size;
    this.info = info;
    this.nMore = nMore;
    //追加
    this.B1=B1;
    this.B2=B2;
    this.board = [size];
    for(let k=0;k<size;k++){
      this.board[k]=k;
    }
    if(nMore > 0){
      // try{
        if(this.bThread){
          // this.child.start();
          let worker = new Worker('nQueen13_thread.js');
          const promise = new Promise((resolve, reject) => {
             worker.addEventListener('message', (msg) => {
              if(msg.data.mode === 'print') {
                // aBoard2 = msg.data.pos;
                // set(msg.data.bit, msg.data.row, msg.data.size);
                aBoard2[msg.data.row] = msg.data.pos;
                // sleep(self.SPEED);
                // self.postMessage({status: 'process', box: aBoard2, row: msg.data.row+1, size: msg.data.size});

              } else if(msg.data.mode === 'setCount') {
                this.info.setCount(msg.data.val[0],msg.data.val[1],msg.data.val[2]);
              } else if(msg.data.mode === 'end') {
                resolve(msg.data);
              }
             });
          });
          children.push(promise);
          worker.postMessage(this);
          
          this.WorkingEngine(size, nMore-1, info, B1-1, B2+1);
        }
      // } catch(e){
      //   console.warn(e);
      //   // System.out.println(e);
      // }
    } else {
      // this.child = null;
    }
  }
}

var COUNT8 = null;
var COUNT4 = null;
var COUNT2 = null;
class Board{
  constructor() {
  }
  //
  Board(){
    COUNT8=COUNT4=COUNT2=0;
  }
  //
  getTotal(){
    return COUNT8*8+COUNT4*4+COUNT2*2;
  }
  //
  getUnique(){
    return COUNT8+COUNT4+COUNT2;
  }
  //
  setCount(c8,c4,c2){
    COUNT8 += c8;
    COUNT4 += c4;
    COUNT2 += c2;
  }
}

function main(size, mode = 1){
  let from = new Date();
  let targetN = size;
  

  aBoard2 = new Array(size);
  aBoard2.fill(-1);

  let nThreads = size;
  let info = new Board();
  info.Board();

  for(let i = targetN; i <= targetN; i++){
    let child = new WorkingEngine();
    child.WorkingEngine(size, nThreads, info, size-1, 0);

    // Promise.all(child.child).then((results) => {
    //   console.log(results);
    //   // self.postMessage({status: 'success', result: '' }); 
    // });


    // self.postMessage({status: 'process', result: `n:${i}\t\tTotal:${info.getTotal()}\t\tUnique:${info.getUnique()}\t\ttime:${timeFormat(from)}`});
  }
  Promise.all(children).then((results) => {
    // console.log(results);
    self.postMessage({status: 'process', result: `n:${targetN}\t\tTotal:${info.getTotal()}\t\tUnique:${info.getUnique()}\t\ttime:${timeFormat(from)}`}); 
    self.postMessage({status: 'success', result: '' }); 
  });

  // self.postMessage({status: 'success', result: '' });

}


self.addEventListener('message', (msg) => {
  self.SPEED = Number(msg.data.speed) * 1000;
  if(msg.data.size) {
    main(Number(msg.data.size), Number(msg.data.mode));
  }
});

//EOS2