#!/bin/bash

javac -cp .:commons-lang-2.6.jar: Algorithm.java && java -cp .:commons-lang-2.6.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Algorithm  ;

