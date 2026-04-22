#!/bin/bash
plumed config makefile_conf > plumed_makefile.conf
getvar () { sed -n "s/^$1=//p" plumed_makefile.conf; }
PREFIX=$(getvar prefix)
CPPFLAGS=$(getvar CPPFLAGS)
CXXFLAGS=$(getvar CXXFLAGS)
LIBS=$(getvar LIBS)
DYNAMIC_LIBS=$(getvar DYNAMIC_LIBS)
SOEXT=$(getvar SOEXT)
PLUMED_SRC="./plumed2/src"

g++ -O3 -fPIC -shared sd.cpp -o sd.${SOEXT} \
    -I${PLUMED_SRC} -I${PREFIX}/include \
    -L${PREFIX}/lib -Wl,-rpath,${PREFIXX}/lib \
    -Wl,--no-undefined -Wl,--no-as-needed \
    -lplumedKernel -ldl

