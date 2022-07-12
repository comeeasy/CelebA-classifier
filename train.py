from numpy import dtype
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from torchsummary import summary

from tqdm import tqdm
import argparse

from celebA_data import CelebA, get_celeba_dataloader


class Trainer():
    """
        We use pretrained Resnet18 model. 
        modified fc layer only

        loss function   : Cross Entropy
        optimizer       : Adam
    """

    def __init__(self,  epochs, learning_rate, train_loader, val_loader, device) -> None:
        self.model = torchvision.models.resnet18(pretrained=True)

        # num of target label is 2 (male[1], female[0])
        self.model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        summary(self.model, input_size=(3, 224, 224), device="cpu")
        self.model = self.model.to(device)

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device


    def train(self):

        for epoch in range(self.epochs):
            best_acc = 0
            for X, Y in tqdm(self.train_loader):
                X = X.to(self.device)
                Y = Y.to(self.device)

                predict = self.model(X)
                loss = self.criterion(predict, Y)
                
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            acc = self.val()
            print(f"epoch: {epoch + 1:3d}\tacc: {100 * acc:.4f}%")
            if best_acc < acc:
                best_acc = acc
                torch.save(self.model.state_dict(), "models/best.pt")

    def val(self):
        """
            calculate accuracy of val_loader
        """
        self.model.eval()

        total_batch = len(self.val_loader)
        accuracy = 0
        with torch.no_grad():
            for X, Y in tqdm(self.val_loader):
                X = X.to(self.device)
                Y = Y.to(self.device)

                predict = self.model(X)
                predict = torch.argmax(predict, dim=1)
                accuracy += (predict == Y).float().mean().item() / total_batch

        self.model.train()
        return accuracy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dset_root", required=True, help="path of directory which includes celeb images (*.jpg)")
    parser.add_argument("--csv_root", required=True, help="path of annotation label csv file which includes 'img_name', 'Male' columns")
    parser.add_argument("--ngpu", help="number of gpu to use -1 is for cpu", type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--split_ratio_train", default=0.8, type=float)

    return parser.parse_args()




if __name__ == "__main__":
    cudnn.benchmark = True

    args = get_args()
    device = f"cuda:{args.ngpu}" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_celeba_dataloader(args)

    trainer = Trainer(
        epochs=args.epochs,
        learning_rate=args.lr,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    trainer.train()    
    print("Done.")
