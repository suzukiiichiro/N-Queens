#!/bin/bash

  rm -fr *.class;
  javac -cp .:commons-lang3-3.4.jar NQueen.java ;
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC NQueen  ;
  rm -fr *.class;

exit ;

