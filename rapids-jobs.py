## This is a test file for rapids
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.6.*" "cuml-cu12==25.6.*"

print("hello")
import pandas
import cudf
print("cudf version: ",cudf.__version__)

