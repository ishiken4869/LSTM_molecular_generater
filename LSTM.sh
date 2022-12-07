#!/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-handson-g1"
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"

module load python/3.6.2
module load cuda/11.0

python3 LSTM.py
