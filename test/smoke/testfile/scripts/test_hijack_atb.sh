scripts_dir=../../testfile/scripts

g++ -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" ${scripts_dir}/op_atb.cpp -l atb -l ascendcl -o ${scripts_dir}/op_atb

file=${scripts_dir}/op_atb
if [ ! -f "$file" ]; then
    echo "Failed: $file not exist."
    exit 1
fi

if [ ! -x "$file" ]; then
    if chmod +x "$file"; then
        echo "$file is executable."
    else
        echo "Failed: $file cannot be executed."
        exit 1
    fi
fi

${scripts_dir}/op_atb
rm -r $file