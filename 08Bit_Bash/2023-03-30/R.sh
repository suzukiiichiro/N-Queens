#!/usr/bin/bash
function eight(){
  Q="$1";
  u="$2";
  ee="$3";
  n="$4";
  s=0;
  Q=;
  H=;
  (( Q=u?Q:(1<<Q)-1 ));
  (( H=~(u|ee|n)&Q ));
  while((H));do
    (( H^=R=-H&H ));
    eight $Q ($u|$R)<<1 $ee|$R ($n|$R)>>1;
    (( s+=$? ));
  done
  (( s+=ee==Q ));
  [[ s!=0 ]]
  return $s;
}
eight 5 0 0 0 0;
