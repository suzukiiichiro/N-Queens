#!/bin/bash

javac -cp .:commons-lang3-3.4.jar: Algorithm.java && java -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Algorithm  ;



