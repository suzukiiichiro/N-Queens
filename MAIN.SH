#!/bin/bash

javac Algorithm.java && java -server -Xms8G -Xmx16G -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy -XX:+UseConcMarkSweepGC Algorithm ;

