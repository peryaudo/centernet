from datasets import load_dataset

ds = load_dataset("detection-datasets/coco")

print(ds.take(1))

# https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/main.py#L32
# https://mmdetection.readthedocs.io/en/v2.15.1/_modules/mmdet/models/dense_heads/centernet_head.html
# https://github.com/a5chin/centernet/blob/b4dcfdae555a35c5e043ad8960c6e7eaa25cd878/centernet/centernet.py#L116
# https://arxiv.org/pdf/1904.07850
# https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/utils.py
# https://huggingface.co/datasets/detection-datasets/coco?library=datasets
# https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/dense_heads/centernet_head.py#L201
# https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152
