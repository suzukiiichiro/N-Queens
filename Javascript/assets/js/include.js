function timeFormat(start){
  let now = new Date();
  let ms = now.getTime() - start.getTime();
  // moment.locale('ja');
  return moment(ms).utc().format(" HH:mm:ss.SSS") ;
}
function sleep(speed) {
  let startMsec = new Date();
  while (new Date() - startMsec < speed);
}



//回転
function rotate(chk, scr, n, neg) {
  let k = neg ? 0 : n - 1;
  let incr = (neg ? +1 : -1);
  for(let j = 0; j < n; k += incr) {
    scr[j++] = chk[k];
  }
  k = neg ? n - 1 : 0;
  for(let j = 0; j < n; k -= incr) {
    chk[scr[j++]] = k;
  }
}
//反転
function vMirror(chk, n) {
  for(let j = 0; j < n; j++) {
    chk[j] = (n - 1) - chk[j];
  }
}
function intncmp(lt, rt, n) {
  let rtn = 0;
  for(let k = 0; k < n; k++) {
    rtn = lt[k] - rt[k];
    if(rtn != 0) {
      break;
    }
  }
  return rtn;
}
