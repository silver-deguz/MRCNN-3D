#export CUDA_PATH=/usr/local/cuda/

CUDA_PATH=/usr/local/cuda

# echo "Compiling crop_and_resize kernels by nvcc..."
# cd roi_align/src/cuda
# $CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_30 \
#  -gencode=arch=compute_30,code=sm_30 \
#  -gencode=arch=compute_50,code=sm_50 \
#  -gencode=arch=compute_52,code=sm_52 \
#  -gencode=arch=compute_60,code=sm_60 \
#  -gencode=arch=compute_61,code=sm_61 \
#  -gencode=arch=compute_62,code=sm_62 \
#
# cd ../../../roi_align
# python build.py

echo "Compiling nms kernels by nvcc..."
cd cuda_functions/nms_xD/src/cuda/
$CUDA_PATH/bin/nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_62,code=sm_62 \

cd ../../
python build.py
cd ../../

echo "Compiling roi align kernels by nvcc..."
cd cuda_functions/roi_align_xD/roi_align/src/cuda/
$CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_62,code=sm_62 \

cd ../../
python build.py
cd ../../
