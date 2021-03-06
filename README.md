MC2ARCdataset_evaluate

# Usage
http://mprg.jp/en/research/arc_dataset_2017_e
(JP: http://mprg.jp/research/arc_dataset_2017_j)

If above url is not available, please visit: http://mprg.jp/research/arc_dataset_2017_e

**Note: In `detect_evaluation.py` and `segment_evaluation.py`, the method of giving the path has changed.** It is given as arguments as follows:
```
python detect_evaluation.py --image /path/of/input/images/ --teach /path/of/ground-truth/textfiles/ --result /path/of/ur/results/
```
```
python segment_evaluation.py --teach /path/of/ground-truth/images/ --result /path/of/ur/results/
```
If arguments is not given, it refers to the default path `(./detect_evaluation/{image|teach|results})` and `(./segmentation_evaluation/{label|results})`.

# Requirement
- Python 3.x / 2.x (Recommended version >= 2.7.11 or >= 3.5.1)
- OpenCV 3.x / 2.x
- numpy (Recommended version >= 1.10)
- matplotlib (Recommended version >= 1.4)
- numba (Recommended version >= 0.30)
