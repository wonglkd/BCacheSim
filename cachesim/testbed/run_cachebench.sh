#!/bin/bash
set -e -x
if [ $# -eq 0 ]; then
    echo "You must supply at least one argument:"
    echo "  ./run_cachebench.sh <folder> [<poll_interval>]"
    echo "  <folder>/cachebench.json <folder>/progress.log <folder>/progress.out"
    exit 1;
fi

RUN_DIR=$1
# Progress interval for CacheBench
PROG_INTERVAL=${2-300}

# File must exist and not be empty
if [[ -s $RUN_DIR/windows.log || -s $RUN_DIR/windows.log.gz ]]; then
    echo "Output file already exists! Please remove it first."
    echo "ls \$RUN_DIR: $RUN_DIR"
    ls -l $RUN_DIR
    
    if [[ -s $RUN_DIR/cachebench.done ]]; then
        echo "Previous run was successful"  
        exit 0;
    elif grep -Fq "Finished a full run." $RUN_DIR/progress.out; then
        echo "Previous run was successful"  
        exit 0;
    elif grep -Fq "Backend Request Latency" $RUN_DIR/progress.out; then
        echo "Previous run appears to have been successful"
        exit 0;
    else
        echo "Previous run did not complete - moving file and trying again"
        mv $RUN_DIR/windows.log $RUN_DIR/windows.log.bak.$(date "+%Y%m%d_%H%m")
    fi
fi

cd /users/dlwong/testbed
ls -l ./build-cachelib/cachebench/cachebench
CACHEBENCH_HASH=`sha1sum ./build-cachelib/cachebench/cachebench | cut -d " " -f 1`
echo "Cachebench SHA1: $CACHEBENCH_HASH"
CB_TMP_DIR="/tmp/cb-${CACHEBENCH_HASH:0:10}"
if [[ ! -f $CB_TMP_DIR/cachebench ]]; then
    mkdir $CB_TMP_DIR
    cp ./build-cachelib/cachebench/cachebench $CB_TMP_DIR
fi
sha1sum $CB_TMP_DIR/cachebench
rm -rf /mnt/hdd/baleen/run || true
mkdir -p /mnt/hdd/baleen/run
rm -rf /mnt/hdd2/baleen/run || true
mkdir -p /mnt/hdd2/baleen/run
# Wrap in braces so that bash loads it all at start, preventing errors if script
# is later modified
{
    iostat -c -d -t -x -y -o JSON 15 > $RUN_DIR/iostat.log &
    pio=$!;
    echo "PID of iostat: $pio"
    # nohup (sleep 5; while pgrep -x cachebench > /dev/null; do sleep 5; done; kill $pio; echo "Terminated iostat")&
    trap "kill $pio; echo 'Killed iostat $pio'" EXIT
    ($CB_TMP_DIR/cachebench --json_test_config $RUN_DIR/cachebench.json \
        --progress_stats_file=$RUN_DIR/progress.log \
        --progress $PROG_INTERVAL; gzip $RUN_DIR/windows.log; touch ${RUN_DIR}/cachebench.done) > >(tee $RUN_DIR/progress.out) 2> >(tee $RUN_DIR/progress.err >&2)
    exit
}