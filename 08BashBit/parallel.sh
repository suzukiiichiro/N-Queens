#日本語
#!/usr/bin/bash

shopt -s expand_aliases
alias xargs=gxargs; # /usr/bin/gxargs
declare -i size;
declare -i COUNT=0;
#
function f()
{
  echo $*;
  sleep 1;
  ((COUNT++));
}

function test(){
size=5;
export -f f;
seq 10 | xargs -I % -P$size bash -c 'f %'
wait;
echo "$COUNT";
}
#
test;
