#!/bin/sh

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:."

JNI_PATH="jni_include"

javac -d bin -sourcepath src -classpath "lib/trove.jar" src/lowrankparser/LowRankParser.java


javac -d bin -sourcepath src -classpath "lib/trove.jar" src/unit_test/svdTester.java


javah -classpath bin utils.SVD
g++ -fPIC -shared -I./lib/SVDLIBC/ -I${JNI_PATH} -I${JNI_PATH}/linux libSVD.cpp ./lib/SVDLIBC/libsvd.a -O2 -o libSVDImp.so



