#%%
import cv2
import argparse
from PIL import Image
import torch
import torchvision.transforms as T

import utils
import vision_transformer as vit_o
from supervised_training import get_arguments, load_model

#%%
def prepare_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(img)
    img = T.Compose([
        utils.GaussianBlurInference(),
        T.ToTensor()
    ])(img)
    img = img.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return img.to(device)
#%%
def main(args):
    model, _, _ = load_model(args)
    image = prepare_image(args.image_path)
    preds = model(image)
    print("Prediction for Normal: ", torch.sigmoid(preds[0]).item())
    print("Prediction for Nodule: ", torch.sigmoid(preds[1]).item())
    print("Prediction for Pneumonia: ", torch.sigmoid(preds[2]).item())
    print("Prediction for Pneumothorax: ", torch.sigmoid(preds[3]).item())

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('classify-single-image',
                                     parents=[get_arguments()])
    parser.add_argument('--image_path', default='PATH/TO/IMAGE')
    parser.set_defaults(checkpoint=r'.\saved_models\checkpoint.pth')
    args = parser.parse_args()
    main(args)
# %%
