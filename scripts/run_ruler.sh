#!/bin/bash
set -euo pipefail

cd eval/RULER/

if [ "${SKIP_SETUP:-0}" != "1" ]; then
    bash setup.sh
fi

cd scripts

if [ -n "${RULER_STRIDES_OVERRIDE:-}" ]; then
    IFS=',' read -r -a STRIDES <<< "${RULER_STRIDES_OVERRIDE}"
else
    STRIDES=(16 8 4)
fi

for STRIDE in "${STRIDES[@]}"; do
    ./run.sh llama3.1-8b-chat synthetic --stride "${STRIDE}" --metric "${RULER_METRIC:-xattn}"
done
