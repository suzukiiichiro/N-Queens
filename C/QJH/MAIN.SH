#!/bin/sh

rm -fr nqE;
rm -fr nqO;
g++ nqueen_dg_QJH_even.cpp -o nqE;
g++ nqueen_dg_QJH_odd.cpp -o nqO;

./nqE ;
exit ;
