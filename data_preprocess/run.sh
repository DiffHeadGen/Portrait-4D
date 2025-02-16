ml CUDA/11.6.0

empty_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ', ' '{if ($2 < 512) print $1}' | head -n 1)
if [ -z "$empty_gpu" ]; then
    echo "No empty GPU available"
    exit 1
fi
CUDA_VISIBLE_DEVICES=$empty_gpu python preprocess_exp.py
