class UI {
  constructor() {
    this.fileSelect = document.querySelector('#selectFile');
    this.reverse = document.querySelector('#rev');
    this.worker = null;
    this.scroll = null;
  }
  message(msg) {
    document.querySelector('#res #output').insertAdjacentHTML('beforeend', `<p>${msg.data.result}</p>`);
    //最下部にスクロール
    clearTimeout( this.scroll );
    this.scroll = setTimeout(() => {
      let out = document.querySelector('#res');
      out.scrollTop = out.scrollHeight;
    }, 250);

    if(msg.data.status == "success") {
      this.stop();
    }
  }
  async load(file) {
    let target = document.querySelector('#editor code');
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
      try {
        this.worker = new Worker(file);
        this.worker.addEventListener('message', (msg) => {
          this.message(msg);
        }, false);
      } catch(e) {
      }
    });
  }
  //読み込むファイルを選択
  selectFile(e) {
    let value = this.fileSelect.value;
    let fileID = ( '00' + value ).slice( -2 );
    let file = `./js/nQueen${fileID}.js`;
    this.load(file);
  }
  start() {
    document.querySelector('#res #output').textContent = "";
    document.querySelector('#start').classList.add('hide');
    document.querySelector('#stop').classList.remove('hide');

    let number = document.querySelector('#num').value;
    let mode = document.querySelector('#rev').value;
    this.worker.postMessage({size: number, mode: mode});
  }
  stop() {
    document.querySelector('#start').classList.remove('hide');
    document.querySelector('#stop').classList.add('hide');
    // this.worker.terminate();
  }
  //初期設定
  async init() {
    //最初のselectをスクリプトを読み込む
    this.selectFile(this.fileSelect);
    //読み込むファイルを選択
    this.fileSelect.addEventListener('change', this.selectFile.bind(this), false);
    this.reverse.addEventListener('change', this.selectFile.bind(this), false);
    //開始
    document.querySelector('#start').addEventListener('click', this.start.bind(this), false);
    //終了
    document.querySelector('#stop').addEventListener('click', this.stop.bind(this), false);
  }
}


const ui = new UI();
ui.init();

// const botWorker = new Worker(`js01.js`);
// botWorker.addEventListener('message', (msg)  => {
// document.querySelector("body").insertAdjacentHTML('beforeend', `${( '00000000' + msg.data.count ).slice( -8 )} : ${msg.data.result}<br>`);
// });
// botWorker.postMessage('');