class WorkingEngine {
  constructor(opt, _this) {
    this.workingEngine = opt;
  }

  run(){
    //マルチスレッド
    this.workingEngine.board[0] = 1;
    this.workingEngine.sizeE = this.workingEngine.size - 1;
    this.workingEngine.MASK = (1 << this.workingEngine.size) - 1;
    this.workingEngine.TOPBIT = 1 << this.workingEngine.sizeE;

    if(this.workingEngine.B1 > 1 && this.workingEngine.B1 < this.workingEngine.sizeE){
      this.BOUND1(this.workingEngine.B1);
    }
    this.workingEngine.ENDBIT = (this.workingEngine.TOPBIT >> this.workingEngine.B1);
    this.workingEngine.SIDEMASK = this.workingEngine.LASTMASK = (this.workingEngine.TOPBIT | 1);
    if(this.workingEngine.B1 > 0 && this.workingEngine.B2 < this.workingEngine.size - 1 && this.workingEngine.B1 < this.workingEngine.B2){
      for(let i=1; i < this.workingEngine.B1; i++){
        this.workingEngine.LASTMASK = this.workingEngine.LASTMASK | this.workingEngine.LASTMASK >>1 | this.workingEngine.LASTMASK << 1;
      }
      this.BOUND2(this.workingEngine.B1, this.workingEngine.B2);
      this.workingEngine.ENDBIT >>= this.workingEngine.nMore;
    }
  }

  symmetryOps(bitmap){
    let own, ptn, you, bit;
    //90度回転
    if(this.workingEngine.board[this.workingEngine.BOUND2]==1){
      own = 1;
      for(ptn = 2; own <= this.workingEngine.sizeE; own++, ptn <<= 1){
        bit = 1;
        let bown = this.workingEngine.board[own];
        for(you = this.workingEngine.sizeE; (this.workingEngine.board[you]!=ptn)&&(bown>=bit);you--){ bit<<=1; }
        if(bown>bit){ return; }
        if(bown<bit){ break; }
      }
      //90度回転して同型なら180度/270度回転も同型である
      if(own>this.workingEngine.sizeE){
        //        COUNT2++;
        self.postMessage({mode: 'setCount', val: [0,0,1]});
        // this.workingEngine.info.setCount(0,0,1);
        return;
      }
    }
    //180度回転
    if(bitmap==this.workingEngine.ENDBIT){
      own=1;
      for(you=this.workingEngine.sizeE-1;own<=this.workingEngine.sizeE;own++,you--){
        bit=1;
        for(ptn=this.workingEngine.TOPBIT;(ptn!=this.workingEngine.board[you])&&(this.workingEngine.board[own]>=bit);ptn>>=1){ bit<<=1;}
        if(this.workingEngine.board[own]>bit){ return; }
        if(this.workingEngine.board[own]<bit){ break; }
      }
      //90度回転が同型でなくても180度回転が同型である事もある
      if(own>this.workingEngine.sizeE){
        //        COUNT4++;
        self.postMessage({mode: 'setCount', val: [0,1,0]});
        // this.workingEngine.info.setCount(0,1,0);
        return;
      }
    }
    //270度回転
    if(this.workingEngine.board[this.workingEngine.BOUND1]==this.workingEngine.TOPBIT){
      own=1;
      for(ptn=this.workingEngine.TOPBIT>>1;own<=this.workingEngine.sizeE;own++,ptn>>=1){
        bit=1;
        for(you=0;this.workingEngine.board[you]!=ptn&&this.workingEngine.board[own]>=bit;you++){ bit<<=1; }
        if(this.workingEngine.board[own]>bit){ return; }
        if(this.workingEngine.board[own]<bit){ break; }
      }
    }
    //    COUNT8++;
    self.postMessage({mode: 'setCount', val: [1,0,0]});
    // this.workingEngine.info.setCount(1,0,0);
  }

  BOUND2(B1, B2){
    let bit;
    this.workingEngine.BOUND1 = B1;
    this.workingEngine.BOUND2 = B2;
    this.workingEngine.board[0] = bit = (1 << this.workingEngine.BOUND1);
    this.backTrack2(1, bit << 1, bit, bit >> 1);
    this.workingEngine.LASTMASK |= this.workingEngine.LASTMASK >> 1 | this.workingEngine.LASTMASK<<1;
    this.workingEngine.ENDBIT>>=1;
  }
  BOUND1(B1){
    let bit;
    this.workingEngine.BOUND1 = B1;
    this.workingEngine.board[1] = bit = (1 << this.workingEngine.BOUND1);
    this.backTrack1(2, (2|bit) << 1, (1|bit), bit>>1);
  }
  backTrack2(row,left,down,right){
    let bit;
    let bitmap=this.workingEngine.MASK&~(left|down|right);
    this.set(bitmap, row, this.workingEngine.size);
    
    // this.set(bitmap, this.workingEngine.row, this.workingEngine.size);

    if(row==this.workingEngine.sizeE){
      if(bitmap!=0){
        if((bitmap&this.workingEngine.LASTMASK)==0){
          this.workingEngine.board[row]=bitmap;

          this.set(bitmap, row+1, this.workingEngine.size);
          // let pos = this.zeroPadding(bitmap.toString(2), this.workingEngine.size).split("").indexOf('1');
          // self.postMessage({mode: 'print', pos: pos, row: row, size: this.workingEngine.size});

          this.symmetryOps(bitmap);
        }
      }
    }else{
      if(row<this.workingEngine.BOUND1){
        bitmap&=~this.workingEngine.SIDEMASK;
        this.set(bitmap, row, this.workingEngine.size);
      }
      else if(row==this.workingEngine.BOUND2){
        if((down&this.workingEngine.SIDEMASK)==0){ return; }
        if((down&this.workingEngine.SIDEMASK)!=this.workingEngine.SIDEMASK){
          bitmap&=this.workingEngine.SIDEMASK;
        }
      }
      while(bitmap>0){
        //最も下位の１ビットを抽出
        bitmap^=this.workingEngine.board[row]=bit=(-bitmap&bitmap);
        
        this.set(bitmap, row, this.workingEngine.size);

        this.backTrack2(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      }
    }
  }
  backTrack1(row,left,down,right){
    let bit;
    let bitmap=this.workingEngine.MASK&~(left|down|right);
    this.set(bitmap, row, this.workingEngine.size);
    
    // this.set(bitmap, this.workingEngine.row, this.workingEngine.size);

    if(row==this.workingEngine.sizeE){
      if(bitmap!=0){
        this.workingEngine.board[row]=bitmap;
        this.set(bitmap, row, this.workingEngine.size);
        //        COUNT8++;
        self.postMessage({mode: 'setCount', val: [1,0,0]});
        // this.workingEngine.info.setCount(1,0,0);
      }
    }else{
      if(row<this.workingEngine.BOUND1){ bitmap&=~2; }
      while(bitmap>0){
        //最も下位の１ビットを抽出
        bitmap^=this.workingEngine.board[row]=bit=(-bitmap&bitmap); 
        this.set(bitmap, row, this.workingEngine.size);
        this.backTrack1(row+1,(left|bit)<<1,down|bit,(right|bit)>>1);
      }
    }
  }

  set(bit, row, size) {
    let pos = this.zeroPadding(bit.toString(2), size).split("").indexOf('1');
    self.postMessage({mode: 'print', pos: pos, row: row, size: size});
  }
  zeroPadding(NUM, LEN){
    return ( Array(LEN).join('0') + NUM ).slice( -LEN );
  }

}


this.onmessage = (msg)=> {
  // console.log(msg.data.info.setCount());
  let we = new WorkingEngine(msg.data, this);
  we.run();
  // self.postMessage({mode: 'total'});
  // self.postMessage({mode: 'getUnique'});
  self.postMessage({mode: 'end'});  
  // self.postMessage({mode: 'total'});
  // self.postMessage({mode: 'getUnique'});
  // self.postMessage({mode: 'end'});  
  // console.log(`${we.size}: ${we.COUNT8*8+we.COUNT4*4+we.COUNT2*2} ${we.COUNT8+we.COUNT4+we.COUNT2}`);
};