# MRCNN-3D

arch is defaulted to sm_35, change depending on GPU architecture used

cd cuda_functions/nms_xD/src/cuda/
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35
cd ../../
python build.py
cd ../../

cd cuda_functions/roi_align_xD/roi_align/src/cuda/
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35
cd ../../
python build.py
cd ../../
