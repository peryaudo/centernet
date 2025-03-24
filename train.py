from datasets import load_dataset

ds = load_dataset("detection-datasets/coco")

print(ds.take(1))
