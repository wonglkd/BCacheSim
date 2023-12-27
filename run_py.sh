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

ENVS=/users/dlwong/apps/micromamba/envs
# Needed when running by brooce
PATH=/users/dlwong/.local/bin:$PATH

case "$PYTHON" in
    pypy) PYTHON_BIN="${ENVS}/cachelib-pypy-3.9/bin/python" ;;
    pypy3.8) PYTHON_BIN="${ENVS}/cachelib-pypy-3.8/bin/python" ;;
    py|py3.11) PYTHON_BIN="${ENVS}/cachelib-py-3.11/bin/python" ;;
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