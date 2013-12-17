#!/bin/sh

export GUROBI_HOME="gurobi560/linux64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:gurobi560/linux64/lib:."

type="lab"
args=$1
runid=$2
shift
shift

 ln -s $args.train.$type data/$args.train.$type.$runid
 ln -s $args.test.$type data/$args.test.$type.$runid

	java -classpath "${GUROBI_HOME}/lib/gurobi.jar:bin:lib/trove.jar" -Xmx5000m lowrankparser.LowRankParser model-file:runs/$args.model.$type.$runid train train-file:data/$args.train.$type.$runid test test-file:data/$args.test.$type.$runid $@ | tee runs/$args.$type.$runid.log


    rm data/$args.train.$type.$runid
    rm data/$args.test.$type.$runid
	#rm data/$args/$args.$type.model

