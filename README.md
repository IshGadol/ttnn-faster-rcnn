# faster_rcnn_ttnn

Port Faster R-CNN to Tenstorrent TT-NN and produce correctness + performance results.

## scope
- environment setup (cpu baseline now; tt-nn vm later)
- layer-by-layer mapping plan
- minimal validation harness (coco-style eval hooks)
- perf report template + run logs

## quick start
```bash
python -m pip install -r requirements.txt
make sanity
```
