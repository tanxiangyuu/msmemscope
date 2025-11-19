#!/bin/bash
echo "***************Generate Coverage*****************"

if [ -d "./coverage" ]; then
    rm -rf ./coverage
fi
mkdir coverage

lcov_opt="--rc lcov_branch_coverage=1 --rc geninfo_no_exception_branch=1"
lcov -c -d ./build/test/CMakeFiles/memscope_test.dir -o ./coverage/memscope_test.info -b ./coverage $lcov_opt

lcov -r ./coverage/memscope_test.info '*platform*' -o ./coverage/memscope_test.info $lcov_opt
lcov -r ./coverage/memscope_test.info '*opensource*' -o ./coverage/memscope_test.info $lcov_opt
lcov -r ./coverage/memscope_test.info '*test*' -o ./coverage/memscope_test.info $lcov_opt
lcov -r ./coverage/memscope_test.info '*c++*' -o ./coverage/memscope_test.info $lcov_opt
lcov -r ./coverage/memscope_test.info '/usr/include/*' -o ./coverage/memscope_test.info $lcov_opt


genhtml ./coverage/memscope_test.info -o ./coverage/report --branch-coverage

cd coverage
tar -zcvf report.tar.gz ./report

echo "***************Generate Coverage*****************"
