# vit-multilabel-classification
Multilabel classification using ViT (based on facebookresearch dino repo)

### Classify a single image
```bash
python classify_single_image.py --image_path /PATH/TO/IMAGE --checkpoint /PATH/TO/MODEL_CHECKPOINT
```

### Train a classifier
DataLoader is created using torchvision.utils.ImageFolder, so images of
each class should be in separate folders whose names are the class names.

--n_classes argument can be used to set the number of labels.