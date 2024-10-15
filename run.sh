cfg_path=configs/chameleon/5-shot/dagprompt.yaml

python pretrain.py --cfg $cfg_path
python downstream.py --cfg $cfg_path