MC2ARCdataset_evaluate

# Usage
http://mprg.jp/en/research/arc_dataset_2017_e
(JP: http://mprg.jp/research/arc_dataset_2017_j)

**Note: In `detect_evaluatio.py`, the method of giving the path has changed.** It is given as arguments as follows:
```
python detect_evaluation.py --image /path/of/input/images/ --teach /path/of/ground-truth/textfiles/ --result /path/of/ur/results/
```
If arguments is not given, it refers to the default path `(./detect_evaluation/{image|teach|results})`.

# Requirement
- Python 3.x / 2.x (Recommended version >= 2.7.11 or >= 3.5.1)
- OpenCV 3.x / 2.x
- numpy (Recommended version >= 1.10)
- matplotlib (Recommended version >= 1.4)
