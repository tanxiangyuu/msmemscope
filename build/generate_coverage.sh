#!/bin/bash
echo "***************Generate Coverage*****************"

if [ -d "./coverage" ]; then
    rm -rf ./coverage
fi
mkdir coverage

lcov_opt="--rc lcov_branch_coverage=1 --rc geninfo_no_exception_branch=1"
lcov -c -d ./build/test/CMakeFiles/leaks_test.dir -o ./coverage/leaks_test.info -b ./coverage $lcov_opt

lcov -r ./coverage/leaks_test.info '*platform*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '*opensource*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '*test*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '*c++*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '/usr/include/*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '*utility*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '*analysis*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '*event_trace*' -o ./coverage/leaks_test.info $lcov_opt
lcov -r ./coverage/leaks_test.info '*framework*' -o ./coverage/leaks_test.info $lcov_opt

genhtml ./coverage/leaks_test.info -o ./coverage/report --branch-coverage

cd coverage
tar -zcvf report.tar.gz ./report

echo "***************Generate Coverage*****************"
