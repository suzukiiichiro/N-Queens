class UI {
  constructor() {
    this.fileSelect = document.querySelector('#selectFile');
    this.reverse = document.querySelector('#rev');
    this.fileName = "";
    this.worker = null;
    this.scroll = null;
    this.end = false;
  }
  message(msg) {
    if(!this.end) {
      setTimeout(() => {
        if(msg.data.box) {
        } else {
          document.querySelector('#output').insertAdjacentHTML('beforeend', `<p>${msg.data.result}</p>`);
          //最下部にスクロール
          let out = document.querySelector('#results');
          var y = out.scrollHeight - out.clientHeight;
          out.scroll(0, y);
        }
        if(msg.data.status == "success") {
          this.end = true;
          this.start();
        }
      }, 0);
    }
  }
  async load(file) {
    let target = document.querySelector('code');
    target.textContent = "";
    try  {
      target.classList.remove('prettyprinted');
    } catch(e) {
    }
    await fetch(file).then(res => {
      return res.text();
    }).then(res => {
      let outText = res;
      let mode = document.querySelector('#rev').value;
      let reg = new RegExp(`EOS${mode}(.*?)EOS${mode}`, 's');
      outText = outText.match(reg);
      target.textContent = "//"+outText[0];
      PR.prettyPrint();
      this.fileName = file;
    });
  }
  //読み込むファイルを選択
  selectFile(e) {
    let value = this.fileSelect.value;
    let fileID = ( '00' + value ).slice( -2 );
    let file = `./assets/js/nQueen${fileID}.js`;
    this.load(file);
  }
  start(e) {
    let target = document.querySelector('#btn-done');
    if(this.end || target.classList.contains('start')) {
      //Stop
      this.end = false;
      target.classList.remove('start');
      try {
        this.worker.terminate();
      } catch(e){
      }
      this.worker = null;
    } else {
      this.end = false;
      //Start
      if(this.scroll) { clearTimeout(this.scroll); }
      try {
        this.worker = new Worker(this.fileName);
        this.worker.addEventListener('message', (msg) => {
          this.message(msg);
        }, false);
      } catch(e) {
      }
      document.querySelector('#output').textContent = "";
      target.classList.add('start');
      let number = document.querySelector('#num').value;
      let mode = document.querySelector('#rev').value;
      this.worker.postMessage({size: number, mode: mode});
    }
  }

  //初期設定
  async init() {
    //最初のselectをスクリプトを読み込む
    this.selectFile(this.fileSelect);
    //読み込むファイルを選択
    this.fileSelect.addEventListener('change', this.selectFile.bind(this), false);
    this.reverse.addEventListener('change', this.selectFile.bind(this), false);
    //開始
    document.querySelector('#btn-done').addEventListener('click', this.start.bind(this), false);
    //終了
  }
}


const ui = new UI();
ui.init();
