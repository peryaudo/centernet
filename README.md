# CenterNet Implementation

## Training

```
python3 train.py
```

## TODO

- start from pretrained resnet
- measure COCO metrics e.g. mAP
- visualize the result in notebook
- manage depedency with uv
- handle flipped bbox min and max

## Reference Implementations and Resources

- [CenterNet Official Implementation](https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/main.py#L32)
- [MMDetection CenterNet Head](https://mmdetection.readthedocs.io/en/v2.15.1/_modules/mmdet/models/dense_heads/centernet_head.html)
- [Alternative CenterNet Implementation](https://github.com/a5chin/centernet/blob/b4dcfdae555a35c5e043ad8960c6e7eaa25cd878/centernet/centernet.py#L116)
- [CenterNet Paper](https://arxiv.org/pdf/1904.07850)
- [CornerNet-Lite Utils](https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/utils.py)
- [COCO Dataset on Hugging Face](https://huggingface.co/datasets/detection-datasets/coco?library=datasets)
- [MMDetection CenterNet Head Implementation](https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/dense_heads/centernet_head.py#L201)
- [CornerNet Keypoint Utils](https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152) 