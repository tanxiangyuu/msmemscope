set -o errexit

echo "*************** TESTCODE  START *****************"
cd build
export ASCEND_HOME_PATH="PATH"
cmake .. -Dprotobuf_BUILD_TESTS=OFF -DBUILD_TESTS=ON -DFUZZ_TESTS=OFF
make -j8
cd ..
./build/test/leaks_test
echo "**************** TESTCODE  END ******************"