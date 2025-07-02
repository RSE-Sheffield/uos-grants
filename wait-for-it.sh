#!/usr/bin/env bash
# wait-for-it.sh - Wait for a host and port to become available.

set -e

HOST_PORT=$1
shift
TIMEOUT=""
if [[ "$1" == "--timeout="* ]]; then
  TIMEOUT=${1#--timeout=}
  shift
fi

CMD=("$@")

echo "Waiting for $HOST_PORT to become available..."

while ! nc -z $(echo $HOST_PORT | cut -d: -f1) $(echo $HOST_PORT | cut -d: -f2); do
  sleep 1
done

echo "$HOST_PORT is up - executing command: ${CMD[*]}"
exec "${CMD[@]}"
