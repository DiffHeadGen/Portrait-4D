ml CUDA/11.8.0

if [ -n "$1" ]; then
    gpu=$1
else
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ', ' '{if ($2 < 512) print $1}' | head -n 1)
    if [ -z "$gpu" ]; then
        echo "No empty GPU available"
        exit 1
    fi
fi

cd data_preprocess
CUDA_VISIBLE_DEVICES=$gpu python preprocess_exp.py
cd ..