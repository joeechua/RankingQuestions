import pickle
import torch
import torchvision
from sklearn.model_selection import train_test_split
from dataset import VQADataset
#from preprocess.boxes import write_dict_to_json
from tools.evaluate import evaluate
from tools.engine import train_one_epoch
from tools.evaluate import evaluate
import tools.transforms as T
import tools.utils as utils
from vqa import VQAModel
from train import test_model, train_model, validate
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def get_transform(train: bool):
    """Return the transform function

    Args:
        train (bool): whether the transform is applied on training dataset

    Returns:
        func: transform function on image and target
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_train_test_val_dataset(dataset: VQADataset):
    """Split the dataset into training and testing

    Args:
        dataset (AdsDataset): a Pytorch Dataset

    Returns:
        (AdsDataset, AdsDataset): train dataset, test dataset
    """
    # randomly select the training and testing indices
    indices = list(range(len(dataset)))
    train_indices, vt_indices = train_test_split(
        indices, train_size=0.70, shuffle=True, random_state=24)
    val_indices, test_indices = train_test_split(
        vt_indices, train_size=0.75, shuffle=True, random_state=24)

    # split the dataset into train and test
    train_dataset = torch.utils.data.Subset(VQADataset(transforms=get_transform(train=True)), train_indices)
    val_dataset = torch.utils.data.Subset(VQADataset(transforms=get_transform(train=False)), val_indices)
    test_dataset = torch.utils.data.Subset(VQADataset(transforms=get_transform(train=False)), test_indices)

    return train_dataset, val_dataset, test_dataset


def train(num_epochs: int, checkpoint=None, batch_size=8, num_workers=1):
    """Train the model

    Args:
        num_classes (int): number of label classes
        num_epochs (int): number of epochs to train the model
        checkpoint (str, optional): path to the checkpoint file. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 8.
        num_workers (int, optional): number of workers. Defaults to 1.
    """
    # create dataset
    vqa_dataset = VQADataset()
    # create training & testing dataset
    train_dataset, val_dataset, test_dataset = create_train_test_val_dataset(vqa_dataset)

    # define training data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=utils.collate_fn)

    # define testing data loaders
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)
    
    # define testing data loaders
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        # load a model pre-trained on COCO
        model = VQAModel()
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # move model to the right device
    model.to(device)

    # construct a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    dataloaders = {"train": train_dataloader, "val":val_dataloader}

    model = train_model(model, dataloaders, criterion, optimizer, lr_scheduler, "results",
                            num_epochs=num_epochs, use_gpu=True)
    
    test_model(model, test_dataloader,"results/test.json", True)


if __name__ == "__main__":
    train(num_epochs=4)