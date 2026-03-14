

Requirements

- Python 3.7+
- PyTorch, torchvision
- [Detectron2](https://github.com/facebookresearch/detectron2) (install separately; see official docs)
- fvcore, Pillow, numpy, opencv-python

Run

```bash
# Train (example: DUO config)
python train.py --config-file configs/faster_rcnn_R_101_FPN_3x_duo.yaml --num-gpus 1

# Eval only
python train.py --config-file configs/faster_rcnn_R_101_FPN_3x_duo.yaml --eval-only MODEL.WEIGHTS /path/to/model.pth

# Build priors (imports train to register datasets)
python tools/build_priors.py --dataset duo_train_sparse_50_30 --output priors/duo_prior.json
```

Dataset paths (DUO, RUOD, etc.) are set in `train.py`; adjust them for your environment.