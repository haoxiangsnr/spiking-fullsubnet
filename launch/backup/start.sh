# sleep 2d
recipe = /mnt/private_xianghao/proj/audiozen/recipes/dns_interspeech_2020
config = fullsubnet/lookAhead2.toml

cd /mnt/private_xianghao/proj/audiozen/
pip install -e . -i https://mirrors.tencent.com/pypi/simple/

cd $recipe

torchrun --nnodes=1 --nproc_per_node=8 run.py -C $config -M train
