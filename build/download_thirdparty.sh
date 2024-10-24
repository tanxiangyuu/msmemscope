set -e

CUR_DIR=$(dirname $(readlink -f $0))
TOP_DIR=${CUR_DIR}/..

GTEST_DIR="${TOP_DIR}/opensource/googletest"
if [ ! -d "$GTEST_DIR" ]; then
    cd ${TOP_DIR}/opensource
    git clone https://codehub-dg-y.huawei.com/OpenSourceCenter/googletest.git googletest -b release-1.12.1
else
    echo "opensource/googletest already exists. no need to download. exit."
fi
