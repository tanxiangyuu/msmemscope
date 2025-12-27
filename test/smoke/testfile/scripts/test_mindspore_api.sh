MSLEAKS_DIR="../../msmemscope/output"
export LD_LIBRARY_PATH=${MSLEAKS_DIR}/lib64:${LD_LIBRARY_PATH}
export LD_PRELOAD=${MSLEAKS_DIR}/lib64/libleaks_ascend_hal_hook.so:${MSLEAKS_DIR}/lib64/libascend_mstx_hook.so:${MSLEAKS_DIR}/lib64/libascend_kernel_hook.so:${MSLEAKS_DIR}/lib64/libascend_leaks.so
export PYTHONPATH=${MSLEAKS_DIR}/python:$PYTHONPATH

python ../../testfile/scripts/test_mindspore_api.py