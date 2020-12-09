class UI {
  constructor() {
    this.fileSelect = document.querySelector('#selectFile');
    this.reverse = document.querySelector('#rev');
    this.views = document.querySelector('#table');
    this.fileName = "";
    this.worker = null;
    this.scroll = null;
    this.end = false;
    this.viewArray = [];
  }
  setTable(data) {
    //配置したクイーンをリセット
    let q = document.querySelectorAll(`.queen`);
    for(let k = 0; k < q.length; k++) { q[k].classList.remove('queen'); }
    //aBoardの値をみる
    for(let i = 0; i < data.box.length; i++) {
      if(data.box[i] != -1) {
        document.querySelector(`.td-${i}${data.box[i]}`).classList.add('queen');
      }
    }
    //解が正しい場合にキャプチャする
    if(data.row == data.size) {
      html2canvas(document.querySelector("#table table"), {
        backgroundColor: null
      }).then(canvas => {
          document.querySelector('#queens').appendChild(canvas);
      });
    }
  }
  message(msg) {
    if(!this.end) {
      setTimeout(() => {
        if(msg.data.box) {
          this.setTable(msg.data);
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
  makeTable() {
    this.viewArray = [];
    document.querySelector('#queens').innerHTML = '';
    this.views.innerHTML = '<table id="output-table"><tbody></tbody></table>';
    let table = document.querySelector('#output-table tbody');
    let size = document.querySelector('#num').value;
    for(let x = 0; x < size; x++) {
      table.insertAdjacentHTML('beforeend', `<tr class="tr-${x}"></tr>`);
      for(let y = 0; y < size; y++) {
        table.querySelector(`.tr-${x}`).insertAdjacentHTML('beforeend', `<td class="td-${x}${y}" data-val="${x},${y}"></td>`);
        this.viewArray.push([x, y]);
      }
    }
  }
  //読み込むファイルを選択
  selectFile(e) {
    let value = this.fileSelect.value;
    if(value >= 3) { this.makeTable(); }
    let fileID = ( '00' + value ).slice( -2 );
    let file = `./assets/js/nQueen${fileID}.js`;
    this.load(file);
  }
  start(e) {
    let target = document.querySelector('#btn-done');
    if(this.end || target.classList.contains('start')) {
      document.body.classList.remove('q-start');
      //Stop
      this.end = false;
      target.classList.remove('start');
      try {
        this.worker.terminate();
      } catch(e){
      }
      this.worker = null;
    } else {
      document.body.classList.add('q-start');
      if(this.fileSelect.value >= 3) {
        this.makeTable();
      }
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
      let speed = document.querySelector('#speed').value;
      let analyze = document.querySelector('#analyze').checked;
      this.worker.postMessage({size: number, mode: mode, speed: speed, analyze: analyze});
    }
  }

  //初期設定
  async init() {
    //最初のselectをスクリプトを読み込む
    this.selectFile(this.fileSelect);
    //読み込むファイルを選択
    this.fileSelect.addEventListener('change', this.selectFile.bind(this), false);
    this.reverse.addEventListener('change', this.selectFile.bind(this), false);
    document.querySelector('#num').addEventListener('change', this.makeTable.bind(this), false);
    //開始
    document.querySelector('#btn-done').addEventListener('click', this.start.bind(this), false);
    //終了
  }
}


const ui = new UI();
ui.init();
