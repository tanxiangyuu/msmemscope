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

SECUREC_DIR="${TOP_DIR}/platform/securec"
if [ ! -d "$SECUREC_DIR" ]; then
    cd ${TOP_DIR}/platform
    git clone https://codehub-dg-y.huawei.com/hwsecurec_group/huawei_secure_c.git securec -b tag_Huawei_Secure_C_V100R001C01SPC012B002_00001
else
    echo "platform/securec already exists. no need to download. exit."
fi

FUNC_INJECTION_DIR="${TOP_DIR}/platform/func_injection"
if [ ! -d "$FUNC_INJECTION_DIR" ]; then
    cd ${TOP_DIR}/platform
    git clone https://codehub-dg-y.huawei.com/mindstudio/MindStudio-Backend/func_injection.git func_injection -b br_noncom_MindStudio_7.0.T3_POC_20240930
else
    echo "platform/func_injection already exists. no need to download. exit."
fi
