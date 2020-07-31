let NQueen = function() {
  this.selected = [];
  this.before_selected = [];
  this.active = 'active';
  this.queen = 'queen';
  this.mainTable = document.querySelector('#mainTable');
  this.worker = null;
  this.current = [];
  this.timer = null;
  this.current_row = -1;
};
NQueen.prototype = {
  setQueen(e) {
    if(e.target.tagName !== "TD") return;
    let tr = e.target.getAttribute('data-tr');
    let td = e.target.getAttribute('data-td');

    for(let elem of document.querySelectorAll(`tr[data-tr="${tr}"]`)) {
      elem.classList.remove(this.active);
      elem.querySelectorAll('td.queen').forEach(el => {
        if(el.getAttribute('data-td') !== td) {
          el.classList.remove(this.queen);
        }
      });
    }

    for(let item of document.querySelectorAll(`td`)) {
      item.classList.remove(this.active);
      item.innerHTML = "";
      if( e.target.classList.contains(this.queen) ) {
        //自分自身のクイーンを削除
        this.selected[tr] = undefined;
        document.querySelectorAll(`tr[data-tr="${tr}"] td[data-td="${td}"]`).forEach(elem => {
          elem.classList.remove(this.queen);
        });
        document.querySelectorAll( `tr[data-tr="${tr}"]`).forEach(elem => {
          elem.classList.remove(this.active);
        });
        document.querySelectorAll(`.tai-${tr}`).forEach(elem => {
          elem.remove();
        });
        document.querySelectorAll(`.vertical-${tr}`).forEach(elem => {
          elem.remove();
        });
        return;
      }
    }

    this.selected[tr] = Number(td);
    // console.log(this.selected);
    this.selected.forEach((item, id) => {
      if(item !== undefined) {
        document.querySelectorAll(`tr[data-tr="${id}"] td[data-td="${item}"]`).forEach(elem => {
          elem.classList.add(this.queen);
        });
        document.querySelectorAll(`tr[data-tr="${id}"]`).forEach(elem => {
          elem.classList.add(this.active);
        });
        for(let td of document.querySelectorAll(`td[data-td="${item}"]`)) {
          td.insertAdjacentHTML('afterbegin', `<span class="vertical-${id}"></span>`);
        }
        this.taikaku(id, item);
      }
    });
  },
  taikaku(tr, value, add = true) {
    let td = value;
    let td2 = value;
    document.querySelectorAll('.table').forEach(table => {
      for(let i = tr + 1; i < table.querySelectorAll('tr').length; i++) {
        td = td - 1;
        td2 = td2 + 1;
        if(table.querySelector(`tr[data-tr="${i}"] td[data-td="${td}"]`)) {
          if(add) {
            table.querySelector(`tr[data-tr="${i}"] td[data-td="${td}"]`).insertAdjacentHTML('afterbegin', `<span class="tai-${tr}"></span>`);
          }
        }
        if(table.querySelector(`tr[data-tr="${i}"] td[data-td="${td2}"]`)) {
          if(add) {
            table.querySelector(`tr[data-tr="${i}"] td[data-td="${td2}"]`).insertAdjacentHTML('afterbegin', `<span class="tai-${tr}"></span>`);
          }
        }
      }
      td = value;
      td2 = value;
      for(let i = tr - 1; i >= 0; i--) {
        td = td + 1;
        td2 = td2 - 1;
        if(table.querySelector(`tr[data-tr="${i}"] td[data-td="${td}"]`)){
          if(add) {
            table.querySelector(`tr[data-tr="${i}"] td[data-td="${td}"]`).insertAdjacentHTML('afterbegin', `<span class="tai-${tr}"></span>`);
          }
        }
        if(table.querySelector(`tr[data-tr="${i}"] td[data-td="${td2}"]`)){
          if(add) {
            table.querySelector(`tr[data-tr="${i}"] td[data-td="${td2}"]`).insertAdjacentHTML('afterbegin', `<span class="tai-${tr}"></span>`);
          }
        }
      }
    });
  },
  tableTR(table, column) {
    for(let i = 0; i < column; i++) {
      let tr = document.createElement("tr");
      tr.setAttribute('data-tr', i);
      for(let l = column - 1; l >= 0; l--) {
        let td = document.createElement("td");
        td.setAttribute('data-td', l);
        td.setAttribute('data-tr', i);
        tr.appendChild(td);
      }
      table.appendChild(tr);
    }
  },
  run() {
    //初期化
    this.mainTable.innerHTML = "";
    document.querySelector('#sub').innerHTML = "";

    if(this.worker) {
      document.body.classList.remove('run');
      this.worker.terminate();
      this.worker = null;
    }
    document.querySelector('#text').textContent = "";

    //テーブルを作成
    let column = Number(document.querySelector('select').value);
    this.selected = new Array(column).fill();
    this.tableTR(this.mainTable, column);

    //右袖を作成
    for(let i = 4; i < column; i++) {
      let table = document.createElement("table"); 
      table.classList.add('table-'+i);
      table.classList.add('table');
      this.tableTR(table, i);
      document.querySelector('#sub').insertAdjacentHTML('beforeend', table.outerHTML);
    }
  },
  setMsg(msg) {

  },
  message(msg) {
    this.timer = setTimeout(() => {
      if(!msg.data.status) {
        document.body.classList.remove('run');
        clearTimeout(this.timer);
        this.worker.terminate();
        this.timer = null;
      }
      this.current_row = msg.data.row;
      if(msg.data.status) {
        msg.data.box.forEach((select, index) => {
          if(select !== -1) {
            if(this.current[index] === undefined) {
              if(!document.querySelector(`#mainTable tr[data-tr="${index}"] td[data-td="${select}"]`).classList.contains(this.queen)) {
               document.querySelector(`#mainTable tr[data-tr="${index}"] td[data-td="${select}"]`).click();
              }
            }
          } else {
            if(document.querySelector(`#mainTable tr[data-tr="${index}"] td.${this.queen}`)) {
              document.querySelector(`#mainTable tr[data-tr="${index}"] td.${this.queen}`).click();
            }
          }
        });
        //解が正しい場合にキャプチャする
        let cl = msg.data.row == msg.data.size ? 'set-queen' : '';
        let clData = msg.data.box.join("");
        html2canvas(this.mainTable, {
          backgroundColor: null
        }).then(canvas => {
          // if(this.selected.join('') != this.before_selected.join('')) {
            document.querySelector('#queens').appendChild(canvas);
            try { document.querySelector('#queens canvas:last-child').classList.add(cl); } catch(e){}
            document.querySelector('#queens canvas:last-child').setAttribute('data-val', clData);
          // }
          this.before_selected = this.selected.concat();
        });
      } else {
        document.querySelector('#text').textContent = msg.data.result;
        clearTimeout(this.timer);
      }
    }, 0);
  },
  init() {
    this.mainTable.addEventListener('click', this.setQueen.bind(this), false);
    document.querySelector('select').addEventListener('change', this.run.bind(this), false);
    document.querySelector('#btn-reset').addEventListener('click', this.run.bind(this), false);
    this.run();

    document.querySelector('#btn-start').addEventListener('click', function(){
      document.body.classList.add('run');
      if(this.worker) {
        document.body.classList.remove('run');
        this.worker.terminate();
        this.worker = null;
        return;
      }
      this.before_selected = [];
      document.querySelector('#text').textContent = "";
      document.querySelector('#queens').innerHTML = '';
      this.worker = new Worker('assets/js/nqueen.js');
      this.worker.addEventListener('message', (msg) => {
        this.message(msg);
      }, false);

      this.current = this.selected.concat();
      this.current_row = -1;
      this.worker.postMessage({
        size: Number(document.querySelector('select').value),
        speed: Number(document.querySelector('#number').value),
        row: this.selected
      });
    }.bind(this));
  }
};
let nq = new NQueen();
nq.init(); 