#!/bin/bash

NQueen(){
  rm -fr *.class;
  javac -cp .:commons-lang3-3.4.jar: $1.java ;
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC $1  ;
  rm -fr *.class;
}

#NQueen "NQueen1" ;
#NQueen "NQueen2" ;
#NQueen "NQueen3" ;
#NQueen "NQueen4" ;
#NQueen "NQueen5" ;
#NQueen "NQueen6" ;
#NQueen "NQueen7" ;
NQueen "NQueen8" ;
exit ;

