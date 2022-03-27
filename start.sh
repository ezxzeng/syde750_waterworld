#!/bin/bash

dir=$(dirname $(realpath -s $0))
docker run -v $dir:/workspace -it --gpus all --shm-size=10.09gb swarm-group-form /bin/bash 