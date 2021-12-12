DIR=`pwd`
export PYTHONPATH=$DIR/pix2pix:$PYTHONPATH
export PYTHONPATH=$DIR/ssd:$PYTHONPATH
pip install dominate --user
pip install vizer --user
pip install yacs colorama --user
pip uninstall opencv-python # type y
pip install MulticoreTSNE --user
export PYTHONPATH=$DIR/packages/advertorch:$PYTHONPATH
pip install icecream --user
export PYTHONDONTWRITEBYTECODE=1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html --user
