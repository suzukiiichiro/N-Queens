/*--------------------------------------------
共通変数
---------------------------------------------*/
var SIZE = $('#num').val();
var COUNT = 0;

//JS追加変数
var TIMER = false;
var startTime = new Date();
var SELECT = [];
var STEP = 1;
/*--------------------------------------------
設定
---------------------------------------------*/
class init {
  //値を初期化
  initValue(){
    startTime = new Date();
    $('#output').html('');
    $('#res').html('');
    $('#text > *').html('');
    SIZE = $('#num').val();
    SELECT = [];
    COUNT = 0;
  }
  //テーブルを設定
  makeTable(target){
    var ARR = [];
    var html = '<table id="table" class="table table-bordered table-sm"></table>';
    $(target).append(html);
    for(var x = 0; x < SIZE; x++) {
      $(target+' table').append('<tr class="tr-'+x+'"></tr>');
      for(var y = 0; y < SIZE; y++) {
        $(target+' table').find('.tr-'+x).append('<td class="td-'+x+y+'" data-val="'+x+','+y+'"></td>').html();
        if(x == 0) { SELECT.push([x, y]); }
        ARR.push([x,y])
      }
    }
  }
}

/*--------------------------------------------
クイーンを配置
---------------------------------------------*/
class setQueen {
  conflictQueen(){
    for(var i = 0; i < SELECT.length; i++){
      for(var l = 0; l < i; l++){
        //縦横のチェック
        if( SELECT[i][0] == SELECT[l][0] || SELECT[i][1] == SELECT[l][1] ) { return false; }
        //対角のチェック
        for(var k = 0; k < i; k++){
          if( Number(SELECT[k][0]+i-k) == SELECT[i][0]) { return false; }
          if( Number(SELECT[k][0]-i+k) == SELECT[i][0]) { return false; }
        }
      }
    }
    return true;
  }
  selects(){
    $('#count-text').text(`総当たり：${COUNT+1} 回目`)
    $('#output .queen').removeClass('queen');
    $(`#res .id${COUNT+1}`).remove();
    for(var i = 0; i < SELECT.length; i++){
      $(`#output .td-${SELECT[i].join('')}`).addClass('queen');
    }
    if( this.conflictQueen() ) {
      if($(`#res .id${COUNT}`).length > 0){ $(`#res .id${COUNT}`).remove(); }
      $('#output table').clone().addClass(`id${COUNT}`).appendTo('#res')
      if(SIZE % 2 == 1){
        $('#res tr:nth-child('+Math.ceil(SIZE / 2)+')').addClass('c_g');
        $('#res td:nth-child('+Math.ceil(SIZE / 2)+')').addClass('c_g');
      }
      if($('#isstop').prop('checked')){
        $('#stop').trigger('click');
      }
    }
  }
}

/*--------------------------------------------
NQueen1
ブルートフォース
---------------------------------------------*/
class nQueen01 {
  //1ステップ進める
  increaseStep(n){
    if(SELECT[SIZE-n][0] < SIZE - 1) {
      for(var i = 0; i < n; i++){ SELECT[SIZE-i] = [0, SIZE-i]; }
      SELECT.splice(SIZE);
      SELECT[SIZE-n] = [++SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
      if(n != 1) {  STEP = 1; }
    } else {
      for(var i = 0; i < n; i++){  SELECT[SIZE-i] = [0, SIZE-i]; }
      SELECT.splice(SIZE);
      SELECT[SIZE-n] = [++SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
      ++STEP;
      this.increaseStep(STEP);
    }
    _setQueen.selects();
  }
  //1ステップ戻る
  decreaseStep(n){
    if(SELECT[SIZE-n][0] > 0) {
      for(var i = 0; i < n; i++){
        SELECT[SIZE-i] = [SIZE-1, SIZE-i];
      }
      SELECT[SIZE-n] = [--SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
      if(n != 1) { STEP = 1; }
    } else {
      for(var  i = 0; i < n; i++){
        SELECT[SIZE-i] = [0, SIZE-i];
      }
      SELECT[SIZE-n] = [--SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
      ++STEP;
      if(STEP > SIZE) { return; }
      this.decreaseStep(STEP);
    }
    _setQueen.selects();
  }
}


//実行
const _init = new init()
_init.initValue()
_init.makeTable('#output'); //テーブルを作成

const _setQueen = new setQueen(); //クイーンの配置

const _nqueen = new nQueen01();  //どのnQueenで解くかを決める


//1ステップ進める
$('#btn-p').click(function(){
  ++COUNT;
  try { _nqueen.increaseStep(STEP); } catch(e) {}
})
//1ステップ戻る
$('#btn-m').click(function(){
  --COUNT;
  try { _nqueen.decreaseStep(STEP); } catch(e) {}
})
//自動でステップを進める
$('#start').click(function(){
  $('#start').hide();
  $('#stop').show();
  var max = SIZE ** SIZE;
  TIMER = setInterval(function(){
    if(COUNT < max) {
      $('#btn-p').trigger('click');
    } else {
      clearInterval(TIMER)
      $('#start').show();
      $('#stop').hide();
      $('#res-text').text('解：'+$('#res table').length);
      var endTime = new Date();
      $('#res-time').text(`処理時間：${Math.ceil((endTime - startTime) / 60)} s`);
    }
  }, 0)
})
//一時停止を行う
$('#stop').click(function(){
  $('#start').show();
  $('#stop').hide();
  clearInterval(TIMER);
})
//値に変更があった場合
$('#num').keyup(function(){
  _init.initValue()
  _init.makeTable('#output'); 
})


