set -e

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..

GTEST_DIR="${TOP_DIR}/opensource/googletest"
if [ ! -d "$GTEST_DIR" ]; then
    cd ${TOP_DIR}/opensource
    git clone https://gitcode.com/GitHub_Trending/go/googletest.git googletest -b release-1.12.0
else
    echo "opensource/googletest already exists. no need to download. exit."
fi

SECUREC_DIR="${TOP_DIR}/platform/securec"
if [ ! -d "$SECUREC_DIR" ]; then
    cd ${TOP_DIR}/platform
    git clone https://gitcode.com/openeuler/libboundscheck.git securec -b v1.1.16
else
    echo "platform/securec already exists. no need to download. exit."
fi

JSON_PATH="${TOP_DIR}/opensource/json"
if [ ! -d "$JSON_PATH" ]; then
    cd ${TOP_DIR}/opensource
    git clone https://gitcode.com/GitHub_Trending/js/json.git json -b v3.11.3
else
    echo "opensource/json already exists. no need to download. exit."
fi
