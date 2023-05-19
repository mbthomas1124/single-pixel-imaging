from piqa import MS_SSIM
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from typing import Tuple, Callable


def adjust_img_path(path: str) -> str:
    # modify image paths to current directory location
    main_dir = os.getcwd()
    return os.path.join(main_dir, *(path.split('/')[-4:]))


def process_data(directory: str, train: bool, test_files: list) -> Tuple[torch.Tensor, int, list, list]:
    # prepare feature data and image paths
    image_paths = []
    digits = []
    raw_data = None
    num_features = 40
    clean = os.path.join(directory, 'clean_split')

    # iterate through the files in directory
    for filename in os.listdir(clean):
        file = os.path.join(clean, filename)
        # checking if it is a file
        if os.path.isfile(file):
            if train:
                id = 'rain.pt'
            else:
                id = 'test.pt'
            if (filename[-7:] == id):
                if (id == 'rain.pt') or (id == 'test.pt' and ((test_files == None) or (filename in test_files))):
                    # print(filename)
                    data = torch.load(file)
                    features = data[0]
                    labels = data[1]
                    if (features.shape[1] == num_features):
                        if raw_data is None:
                            raw_data = features
                        else:
                            raw_data = np.append(raw_data, features, axis=0)
                    else:
                        # print(f"IGNORED: {filename}")
                        continue
                    for i, y in enumerate(labels):
                        img_path = adjust_img_path(y[0])
                        if not os.path.isfile(img_path):
                            # print(f"INGORED IMAGE: {img_path}")
                            index = i - len(labels)
                            raw_data = np.delete(raw_data, index, 0)
                            continue
                        else:
                            image_paths.append(img_path)
                            digits.append(y[1])
                else:
                    continue
            else:
                continue
        else:
            raise Exception("this is not a file")
    raw_data = torch.tensor(raw_data.astype(np.float32)).cuda()
    num_features = raw_data.shape[1]
    return raw_data, num_features, image_paths, digits



# define data transformations
def feature_transform(x: torch.Tensor):
    # standardizes the features of a given data point
    mean = x.mean()
    std = x.std()
    return x.sub(mean).div(std)

def img_label_transform(y: str):
    # transforms an image path to a usable tensor
    convert_tensor = A.Compose([A.ToFloat(max_value=255), ToTensorV2()])
    image = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    try:
        label = convert_tensor(image=image)['image']
    except Exception as e:
        print(e)
        raise Exception(f"Image not found: {y}")
    return label.cuda()

def digit_label_transform(y: str):
    return int(y)



# create train and test Datasets and DataLoaders
class ReconstructionData(Dataset):
    def __init__(self, raw_features: torch.Tensor, labels: list, feature_transform, label_transform: None):
        self.features = raw_features
        self.labels = labels
        self.feature_transform = feature_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        data = self.feature_transform(self.features[idx])
        label = self.label_transform(self.labels[idx])
        return data, label


def prep_data(train_features: torch.Tensor, train_labels: list, test_features: torch.Tensor, test_labels: list, label_transform, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # create train and test dataloaders
    train_data = ReconstructionData(
        train_features, train_labels, feature_transform, label_transform)
    test_data = ReconstructionData(
        test_features, test_labels, feature_transform, label_transform)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


class ImgReconstructNN(nn.Module):
    def __init__(self, num_features: int):
        super(ImgReconstructNN, self).__init__()
        self.linear_relu_deconv_stack = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.PReLU(),
            nn.Linear(512, 2048),
            nn.PReLU(),
            nn.Linear(2048, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        logits = self.linear_relu_deconv_stack(x)
        return logits


class ClassifyNN(nn.Module):
    # INCOMPLETE
    # TO DO
    def __init__(self, num_features: int):
        super(ImgReconstructNN, self).__init__()
        self.linear_relu_deconv_stack = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.PReLU(),
            nn.Linear(512, 2048),
            nn.PReLU(),
            nn.Linear(2048, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        logits = self.linear_relu_deconv_stack(x)
        return logits


def create_model(num_features: int, device: str, verbose: bool) -> nn.Module:
    model = ImgReconstructNN(num_features).to(device)
    if verbose:
        print(model)
    return model



# Training loop
def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, print_loss: bool, loss_list: list) -> list:
    num_batches = len(dataloader)
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= num_batches
    loss_list.append(train_loss)
    if print_loss:
        print(f"Avg batch loss: {train_loss:>8f}")
    return loss_list


def test_loop(dataloader, model, device, loss_fn, verbose: bool):
    print("Running Test Loop")
    num_batches = len(dataloader)
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_ssim += ssim(pred, y).item()
            test_psnr += psnr(pred, y).item()

    test_loss /= num_batches
    test_psnr /= num_batches
    test_ssim /= num_batches
    if verbose:
        print(f"Avg batch loss: {test_loss:>8f}")
        print(f"Avg batch PSNR: {test_psnr:>8f}")
        print(f"Avg batch SSIM: {test_ssim:>8f}")
    return model, test_loss, test_psnr, test_ssim


def train_model(train_dataloader, model, loss_fn, optimizer, epochs):
    losses = []
    for t in range(epochs):
        print_loss = False
        if (t % 16 == 15) or (t == 0):
            print_loss = True
            print("-------------------------------")
            print(f"Epoch {t+1}")
        losses = train_loop(train_dataloader, model, loss_fn,
                            optimizer, print_loss, losses)
    print("Done!")
    return losses


def plot_loss(losses: list) -> None:
    losses = np.array(torch.tensor(losses).cpu())
    plt.plot(losses, color='red')
    plt.ylabel('Average Batch Loss')  # set the label for y axis
    plt.xlabel('Epoch')  # set the label for x-axis
    plt.title("Loss over Training Epochs")  # set the title of the graph
    plt.show()  # display the graph


def run(directory: str, device: str, test_files, batch_size, loss_fn, learning_rate, epochs: int):
    print("PART 1")
    train_data, num_features, train_image_paths, _ = process_data(
                                                                 directory, True, None)
    print("PART 2")
    test_data, num_features, test_image_paths, _ = process_data(
                                                               directory, False, None)
    print("PART 3")
    train_dataloader, test_dataloader = prep_data(
        train_data, train_image_paths, test_data, test_image_paths, img_label_transform, batch_size)
    print("PART 4")
    model = create_model(num_features, device, True)
    print("PART 5")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("PART 6")
    train_loss = train_model(train_dataloader, model,
                             loss_fn, optimizer, epochs)
    print("PART 7")
    plot_loss(train_loss)
    print("PART 8")
    model, test_loss, test_psnr, test_ssim = test_loop(
        test_dataloader, model, device, loss_fn, True)
    return model, train_loss, test_loss, test_psnr, test_ssim


class L1_SSIM_loss(MS_SSIM):
    def forward(self, x, y):
        l1 = nn.L1Loss()
        return (0.5 * (1. - super().forward(x, y))) + (0.5 * l1.forward(x, y))


def test_run( directory: str, device: str, test_files, batch_size, loss_fn, learning_rate, epochs: int):
    train_data, num_features, train_image_paths, _ = process_data(
                                                                 directory, True, test_files)
    test_data, num_features, test_image_paths, _ = process_data(
                                                               directory, False, test_files)
    _, test_dataloader = prep_data(
        train_data, train_image_paths, test_data, test_image_paths, img_label_transform, batch_size)
    model = create_model(num_features, device, False)
    filename = os.path.join(directory, 'model.pt')
    model.load_state_dict(torch.load(filename))
    model, test_loss, test_psnr, test_ssim = test_loop(
        test_dataloader, model, device, loss_fn, False)
    return model, test_loss, test_psnr, test_ssim


def main():
    # use CUDA processors if available
    vers = torch.version.cuda
    print(vers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # prepare directories
    main_dir = os.getcwd()
    data_dir = os.path.join(main_dir, 'data', 'features', 'SIM_by_crystal')

    # hyper-parameters:
    batch_size = 50
    learning_rate = 1e-3
    epochs = 512

    loss_fn = L1_SSIM_loss(window_size=2, n_channels=1).cuda()

    crystals = ['0.8f', '27mm']
    for crystal in crystals:
        sample_dir = os.path.join(data_dir, crystal)
        model, train_loss, test_loss, test_psnr, test_ssim = run(
            sample_dir, device, None, batch_size, loss_fn, learning_rate, epochs)
        filename = os.path.join(sample_dir, 'model.pt')
        torch.save(model.state_dict(), filename)
        filename = os.path.join(sample_dir, 'train_loss.pt')
        torch.save(train_loss, filename)
        filename = os.path.join(sample_dir, 'test_loss.pt')
        torch.save(test_loss, filename)
        filename = os.path.join(sample_dir, 'test_psnr.pt')
        torch.save(test_psnr, filename)
        filename = os.path.join(sample_dir, 'test_ssim.pt')
        torch.save(test_ssim, filename)

    results = {}
    results['crystal_position'] = []
    results['filename'] = []
    results['test_loss'] = []
    results['test_psnr'] = []
    results['test_ssim'] = []

    for crystal in crystals:
        sample_dir = os.path.join(data_dir, crystal)
        sample_clean = os.path.join(sample_dir, 'clean_split')
        # iterate through the files in directory
        for filename in os.listdir(sample_clean):
            file = os.path.join(sample_clean, filename)
            # checking if it is a file
            if os.path.isfile(file):
                if (filename[-7:] == 'test.pt'):
                    try:
                        model, test_loss, test_psnr, test_ssim = test_run(
                                                                          sample_dir, device, [filename], batch_size, loss_fn, learning_rate, epochs)
                        results['crystal_position'].append(crystal)
                        results['filename'].append(filename)
                        results['test_loss'].append(test_loss)
                        results['test_psnr'].append(test_psnr)
                        results['test_ssim'].append(test_ssim)
                    except:
                        continue
                else:
                    continue
            else:
                print(file)
                raise Exception("this is not a file")

    results = pd.DataFrame.from_dict(results)
    filename = os.path.join(main_dir, 'data', 'sim_results.csv')
    results.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
