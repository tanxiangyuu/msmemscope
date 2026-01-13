MSLEAKS_DIR="../../msmemscope/output"
export PYTHONPATH=${MSLEAKS_DIR}/python:$PYTHONPATH
torchrun --nproc_per_node=2 ../../testfile/scripts/test_decompose_cmd.py --standalone