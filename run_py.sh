#!/bin/bash -l
if [ $# -eq 0 ]; then
    echo "You must supply 2 arguments:"
    echo "  $0 <pypy/python> <script/module> [args]"
    exit 1;
fi
set -e -x
PYTHON=$1
# shellcheck disable=SC2124
ARGS=${@:2}

# TODO - switch environment if available
# conda activate --name cachelib-py-3.11

case "$PYTHON" in
    pypy) PYTHON_BIN="pypy" ;;
    py) PYTHON_BIN="python" ;;
    *) echo "Unknown PYTHON: $PYTHON"; exit 1;
esac

DIR="$(cd "$(dirname "$0")" && pwd)"
cd $DIR/..
# Wrap in braces so Bash loads it all at start, to avoid errors for long-running
# scripts when editing this file
{
    # PYTHON_BIN cannot be quoted.
    # stdbuf -eL -oL makes stderr and stdout be line buffered.
    # shellcheck disable=SC2086
    # Release: skip time
    # stdbuf -eL -oL /usr/bin/time -v $PYTHON_BIN $ARGS
    stdbuf -eL -oL $PYTHON_BIN $ARGS
    # removed -u
    exit
}