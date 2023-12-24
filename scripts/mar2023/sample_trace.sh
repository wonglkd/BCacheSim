#!/bin/bash
# ./sample_trace.sh warmstorage_202302 vll1 sample -l 0.01
set -e
TRACE_GROUP=$1
TRACE=$2
SUBTRACE=$3
TRACE_DIR=~/fb/ws/$TRACE_GROUP/$TRACE/processed
FILES=$TRACE_DIR/$SUBTRACE
echo "Prefix: $FILES"
mkdir -p $TRACE_DIR
ARGS=${@:4}
echo "Remaining args: $ARGS"

DIR="$(cd "$(dirname "$0")" && pwd)"
set -x
bash $DIR/../../run_py.sh pypy scripts/common/sample.py $FILES --trace-id $TRACE --trace-group $TRACE_GROUP $ARGS
