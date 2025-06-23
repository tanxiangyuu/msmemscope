#!/bin/bash
set -o errexit

REAL_PATH=$(realpath "$0")
BUILD_PATH=$(dirname ${REAL_PATH})
ROOT_PATH=$(realpath ${BUILD_PATH}/../)
FUZZ_PATH=$(realpath ${ROOT_PATH}/opensource/Secodefuzz/examples/out-bin-x64/)
export LD_LIBRARY_PATH=${FUZZ_PATH}:${LD_LIBRARY_PATH}
export ASCEND_HOME_PATH="PATH"

#Secodefuzz build
FUZZ_BUILD_DIR="${ROOT_PATH}/opensource/Secodefuzz/build"
if [ ! -d "$FUZZ_BUILD_DIR" ]; then
    cd ${ROOT_PATH}/opensource/Secodefuzz
    bash build.sh gcc
    mkdir ${ROOT_PATH}/opensource/Secodefuzz/build
    cd ${BUILD_PATH}
else
    echo "fuzz already build."
fi

echo "*************** FUZZCODE  START *****************"
cmake .. -Dprotobuf_BUILD_TESTS=OFF -DBUILD_TESTS=ON -DFUZZ_TESTS=ON
make -j8
cd ..

./build/test/fuzz_test/leaks_fuzz

echo "**************** FUZZCODE  END ******************"