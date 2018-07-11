var startTime = new Date();

function nq(len, target){
  var html = '<table id="table" class="table table-bordered table-sm"></table>';
  $(target).append(html);
  for(var x = 0; x < len; x++) {
    $(target+' table').append('<tr class="tr-'+x+'"></tr>');
    for(var y = 0; y < len; y++) {
      $(target+' table').find('.tr-'+x).append('<td class="td-'+x+y+'" data-val="'+x+','+y+'"></td>').html();
      if(x == 0) { SELECT.push([x, y]); }
      ARR.push([x,y])
    }
  }
  $(target).append(html);
}

var SIZE = $('#num').val();
var ARR = [];
var SELECT = [];
var UPPER = 1;
var COUNT = 0;
var CHECK = true;
var RES = [];
var TIMER;
var MAX = SIZE ^ SIZE;

function init(){
  $('#output').html('');
  $('#res').html('');
  $('#text > *').html('');
  SIZE = $('#num').val();
  ARR = [];
  SELECT = [];
  UPPER = 1;
  COUNT = 0;
  CHECK = true;
  RES = [];

  nq(SIZE, '#output');
}

$(function(){ init(); })

function selects(){
  $('#count-text').text(`総当たり：${COUNT} 回目`)
  CHECK = true;
  $('#output .queen').removeClass('queen');
  for(i in SELECT){
    i = Number(i);
    var orgX = SELECT[i][0]
    var orgY = SELECT[i][1]
    $(`#output .td-${SELECT[i].join('')}`).addClass('queen');

    for(l = 0; l < i; l++){
      //横のチェック
      if( orgX == SELECT[l][0] ) { CHECK = false; break; }
      //縦のチェック
      if( orgY == SELECT[l][1] ) { CHECK = false; break; }
      //体格のチェック
      for(k = 0; k < i; k++){
        if( SELECT[k][0] + i - k == orgX) { CHECK = false; break; }
        if( SELECT[k][0] - i + k == orgX) { CHECK = false; break; }
      }
    }
  }
  if(CHECK) {
    RES.push(SELECT);
    if($(`#res .id${COUNT}`).length > 0){
      $(`#res .id${COUNT}`).remove();
      RES.pop();
    }
    $('#output table').clone().addClass(`id${COUNT}`).appendTo('#res')
    $('#find-text').text(`${RES.length} 個発見`);

    if(SIZE % 2 == 1){
      $('#res tr:nth-child('+Math.ceil(SIZE / 2)+')').addClass('c_g');
      $('#res td:nth-child('+Math.ceil(SIZE / 2)+')').addClass('c_g');
    }
    for(var i = 0; i < SIZE / 2+1; i++) { $('#res td:nth-child('+i+')').addClass('c_r'); }
    if($('#isstop').prop('checked')){
      $('#stop').trigger('click');
    }
  }
}


function move(n){
  if(SELECT[SIZE-n][0] < SIZE - 1) {
    for(i = 0; i < n; i++){ if(typeof SELECT[SIZE-i] != 'undefined') { SELECT[SIZE-i] = [0, SIZE-i]; } }
    SELECT[SIZE-n] = [++SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
    if(n != 1) { UPPER = 1; }
    COUNT++;
  } else {
    for(i = 0; i < n; i++){ if(typeof SELECT[SIZE-i] != 'undefined'){ SELECT[SIZE-i] = [0, SIZE-i]; } }
    SELECT[SIZE-n] = [++SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
    UPPER = UPPER + 1;
    move(UPPER);
  }
}

function moveM(n){
  if(SELECT[SIZE-n][0] > 0) {
    for(i = 0; i < n; i++){ if(typeof SELECT[SIZE-i] != 'undefined') { SELECT[SIZE-i] = [SIZE-1, SIZE-i]; } }
    SELECT[SIZE-n] = [--SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
    if(n != 1) { UPPER = 1; }
    COUNT--;
  } else {
    for(i = 0; i < n; i++){ if(typeof SELECT[SIZE-i] != 'undefined'){ SELECT[SIZE-i] = [0, SIZE-i]; } }
    SELECT[SIZE-n] = [--SELECT[SIZE-n][0], SELECT[SIZE-n][1]];
    UPPER = UPPER + 1;
    moveM(UPPER);
  }
}

$('#btn-p').click(function(){
  move(UPPER);
  selects();
})
$('#btn-m').click(function(){
  if(COUNT <= 0) { COUNT = 0; return false; }
  moveM(UPPER);
  selects();
})


$('#start').click(function(){
  $('#start').hide();
  $('#stop').show();
  TIMER = setInterval(function(){
    try {
      $('#btn-p').trigger('click');
    } catch(e){
      // console.log(e)
      console.log(COUNT)
      clearInterval(TIMER)
      $('#start').show();
      $('#stop').hide();
      $('#res-text').text('解：'+RES.length);
      var endTime = new Date();
      $('#res-time').text(`処理時間：${endTime - startTime} ms`);
    }
  }, 1)
})

$('#stop').click(function(){
  $('#start').show();
  $('#stop').hide();
  clearInterval(TIMER);
})

$('#num').keyup(function(){
  init();
})
