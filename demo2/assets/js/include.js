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