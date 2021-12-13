pip install yacs --user
DIR=`pwd`
export PYTHONPATH=$DIR/packages/Augmentor/:$PYTHONPATH
pip install cffi --user
export PYTHONPATH=$DIR/pix2pix:$PYTHONPATH
export PYTHONPATH=$DIR/CDA/cda:$PYTHONPATH
pip install dominate --user
pip install vizer --user
pip install yacs colorama --user
pip uninstall opencv-python
pip install MulticoreTSNE --user
export PYTHONDONTWRITEBYTECODE=1
pip install icecream --user
pip install tensorpack==0.9.0 --user  # FOR TF ROBUST MODELS
#export PYTHONPATH=$DIR/CDA/pretrained:$PYTHONPATH
export PYTHONPATH=$DIR/packages/advertorch:$PYTHONPATH
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html --user
