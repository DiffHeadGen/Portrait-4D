ml CUDA/11.8.0

conda create -n portrait4d python=3.8 -y
conda activate portrait4d
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"

pip install -e ../expdata
pip install face-alignment

pip install huggingface-hub

huggingface-cli download bEijuuu/Portrait4D --include "models/FLAME/*" --local-dir portrait4d/
huggingface-cli download bEijuuu/Portrait4D --include "models/pdfgc/*" --local-dir portrait4d/
huggingface-cli download bEijuuu/Portrait4D --include "data/*.npy" --local-dir portrait4d/
huggingface-cli download bEijuuu/Portrait4D --include "portrait4d-v2*" --local-dir portrait4d/pretrained_models/
huggingface-cli download bEijuuu/Portrait4D --include "data_preprocess/assets/*" --local-dir .
