set -o errexit

echo "*************** TESTCODE  START *****************"
cd build
bash download_thirdparty.sh
export ASCEND_HOME_PATH="PATH"
cmake .. -Dprotobuf_BUILD_TESTS=OFF -DBUILD_TESTS=ON -DFUZZ_TESTS=OFF
make -j8
cd ..

./build/test/memscope_test
echo "**************** TESTCODE  END ******************"