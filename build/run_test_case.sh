set -o errexit

echo "*************** TESTCODE  START *****************"
cd build
cmake .. -Dprotobuf_BUILD_TESTS=OFF -DBUILD_TESTS=ON
make -j8
cd ..
./build/test/leaks_test
echo "**************** TESTCODE  END ******************"