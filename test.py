import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from skimage import io
import matplotlib.pyplot as plt

import os
import argparse


class CelebATester:

    def __init__(self, model_path, transforms, device=None):
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

        self.model.load_state_dict(torch.load(os.path.abspath(model_path)))
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.transforms = transforms

        self.class_name = ["Female", "Male"]


    def __call__(self, img_path):
        img = io.imread(os.path.abspath(img_path))

        if self.transforms:
            img = self.transforms(img)

        if self.device:
            img = img.to(self.device)

        with torch.no_grad():
            predict = self.model(img.unsqueeze_(0))
        label = torch.argmax(predict, dim=1)
        
        return img, self.class_name[label]

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--res18_model_root", required=True, help="path of pt file of trained Resnet18 (*.pt)")
    parser.add_argument("--img_file", required=True, help="path of image file")
    parser.add_argument("--ngpu", help="number of gpu to use -1 is for cpu", type=int)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    device = f"cuda:{args.ngpu}" if torch.cuda.is_available() else "cpu"
    tester = CelebATester(
        model_path=os.path.abspath(args.res18_model_root), 
        transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]),
        device=device
    )

    img, result = tester(os.path.abspath(args.img_file))

    img = img.squeeze_(0)
    img = img.detach().cpu().numpy().transpose((1, 2, 0))

    plt.imshow(img)
    plt.title(args.img_file + ", " + result)
    plt.show()