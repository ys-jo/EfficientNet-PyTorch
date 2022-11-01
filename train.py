import argparse
import os
from torchvision import transforms, datasets
import multiprocessing
from datasets import Customdataset
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch import nn
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau,CosineAnnealingLR, OneCycleLR
from adamp import SGDP
from adamp import AdamP
from utils.util import load_model, save_model
import matplotlib.pyplot as plt


def parse_arg():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--train_dataset_root', default="./dataset",
                        help='Dataset root(train) directory path')
    parser.add_argument('--validation_dataset_root', default="./dataset",
                        help='Dataset root(validation) directory path')
    parser.add_argument('--model', default='efficientnet',choices=['efficientnet','efficientdet'],
                        help='Detector model name')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--step_size', default=70, type=int,
                        help='Step size for step lr scheduler')
    parser.add_argument('--milestones', default=[110, 150], type=int, nargs='*',
                        help='Milestones for multi step lr scheduler')
    parser.add_argument('--scheduler', default='None',
                        choices=['Plateau','step', 'multi_step', 'cosine','None'],
                        type=str, help='Use Scheduler')
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'sgd', 'adamw', 'adamp', 'sgdp'],
                        type=str.lower, help='Use Optimizer')
    parser.add_argument('--input_size', default=[224, 224], type=int,nargs=2,
                        help='input size (width, height)')
    parser.add_argument('--cifar100', default=False, 
                        action='store_true', help='download cifar100 and train')
    parser.add_argument('--cifar10', default=False, 
                        action='store_true', help='download cifar10 and train')
    parser.add_argument('--imagenet', default=False, 
                        action='store_true', help='train imagenet')
    # model parameter
    parser.add_argument('--mean', nargs=3, type=float,
                        default=(0.486, 0.456, 0.406),
                        help='mean for normalizing')
    parser.add_argument('--std', nargs=3, type=float,
                        default=(0.229, 0.224, 0.225),
                        help='std for normalizing')

    args = parser.parse_args()

    # dataset_root check
    if not os.path.isdir(args.train_dataset_root):
        if args.cifar100 or args.cifar10:
            pass
        else:
            raise Exception("There is no train dataset_root dir")

    if not os.path.isdir(args.validation_dataset_root):
        if args.cifar100 or args.cifar10:
            pass
        else:
            raise Exception("There is no validation dataset_root dir")
    return args

def init_optimizer(args, model):
    if args.optimizer == "sgd":
        optimizer = SGD(model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = Adam(model.parameters())
    elif args.optimizer == "adamw":
        optimizer = AdamW(model.parameters())
    elif args.optimizer == "adamp":
        optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    elif args.optimizer == "sgdp":
        optimizer = SGDP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    else:
        raise Exception("unknown optimizer")
    return optimizer

def init_scheduler(args, optimizer, train_dataloader):
    if args.scheduler == "step":
        return StepLR(optimizer,
                        args.step_size,
                        args.gamma)

    elif args.scheduler == "multi_step":
        return MultiStepLR(optimizer,
                            args.milestones,
                            args.gamma)

    elif args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer,mode='min',patience=5, factor=0.1)
    elif args.scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, args.epochs, last_epoch =-1)
    elif args.scheduler == 'onecyclelr':
        return OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader), epochs=args.epochs, anneal_strategy='cos')           
    else:
        return None

def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects
        
# calculate the loss per epochs
def loss_epoch(model, loss_function, dataloader, optimizer, scheduler):
    losses = []
    metrices = []
    for i, batch in enumerate(dataloader):
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        output = model.forward(x)
        loss = loss_function(output, y)
        metric = metric_batch(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        metrices.append(metric)
        losses.append(loss)
    
    loss = sum(losses)/len(losses)
    metric = sum(metrices)/len(metrices)
    if scheduler:
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(loss)
        else:
            scheduler.step()

    return loss, metric

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    args = parse_arg()

    ##################################################################################
    ## Load dataset (Augumentation & PrepareDataset & DataLoader)
    ##################################################################################
    size = args.input_size
    print(f"input resolution : {size}")
    train_t = [transforms.Resize((size[1], size[0]))]
    train_t.extend([transforms.RandomHorizontalFlip(p=0.1),
                #transforms.RandomGrayscale(p=0.2),
                transforms.RandomPerspective(distortion_scale=0.2),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)])

    train_t = transforms.Compose(t)

    val_t = [transforms.Resize((size[1], size[0]))]
    val_t.extend([transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)])

    val_t = transforms.Compose(t)

    if args.cifar100:
        train_dataset = datasets.CIFAR100(root='./cifar100_data', train=True, download=True, transform=train_t)
        val_dataset = datasets.CIFAR100(root='./cifar100_data', train=False, download=True, transform=val_t)
        class_names = ['beaver', 'dolphin', 'otter', 'seal', 'whale',\
                            'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',\
                            'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',\
                            'bottles', 'bowls', 'cans', 'cups', 'plates',\
                            'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',\
                            'clock', 'computer keyboard', 'lamp', 'telephone', 'television',\
                            'bed', 'chair', 'couch', 'table', 'wardrobe',\
                            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',\
                            'bear', 'leopard', 'lion', 'tiger', 'wolf',\
                            'bridge', 'castle', 'house', 'road', 'skyscraper',\
                            'cloud', 'forest', 'mountain', 'plain', 'sea',\
                            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',\
                            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',\
                            'crab', 'lobster', 'snail', 'spider', 'worm',\
                            'baby', 'boy', 'girl', 'man', 'woman',\
                            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',\
                            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',\
                            'maple', 'oak', 'palm', 'pine', 'willow',\
                            'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',\
                            'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
        class_names.sort()
    elif args.cifar10:
        train_dataset = datasets.CIFAR10(root='./cifar10_data', train=True, download=True, transform=train_t)
        val_dataset = datasets.CIFAR10(root='./cifar10_data', train=False, download=True, transform=val_t)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        class_names.sort()
    elif args.imagenet:
        train_dataset = datasets.ImageNet(root=args.train_dataset_root, transform=train_t)
        val_dataset = datasets.ImageNet(root=args.validation_dataset_root, transform=val_t)
        class_names = os.walk(args.train_dataset_root+"/train").__next__()[1]
        class_names.sort()
    else:
        train_dataset = Customdataset.CustomDataset(data_set_path=args.train_dataset_root,transforms=train_t)
        val_dataset = Customdataset.CustomDataset(data_set_path=args.validation_dataset_root,transforms=val_t)
        class_names = os.walk(args.train_dataset_root).__next__()[1]
        class_names.sort()
    print(f"class names : {class_names}")


    if args.num_workers < 0:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = args.num_workers

    train_dataloader = DataLoader(train_dataset,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=True,
                            batch_size=args.batch_size,
                            num_workers=num_workers)

    validation_dataloader = DataLoader(val_dataset,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=True,
                            batch_size=args.batch_size,
                            num_workers=num_workers)

    
    ##################################################################################
    ## Load model (download pretrained model from url)
    ##################################################################################
    VALID_MODELS = (
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
        'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
        'efficientnet-b8',

        # Support the construction of 'efficientnet-l2' without pretrained weights
        'efficientnet-l2'
    )
    model_name = VALID_MODELS[3]
    model = EfficientNet.from_pretrained(model_name, num_classes = len(class_names))
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    ##################################################################################
    ## Define Loss & Optimizer & Scheudler
    ##################################################################################
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = init_optimizer(args, model)
    scheduler = init_scheduler(args, optimizer, train_dataloader)


    ##################################################################################
    ## Training & Validation
    ##################################################################################
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}
    if args.resume:
        epoch, train_loss = load_model(model, args.resume, optimizer)
        print(f"resume at epoch: {epoch}, loss: {train_loss}")
        best_loss = train_loss
    else:
        epoch = 0
        train_loss = 0
        best_loss = float('inf')

    for epoch in range(epoch, args.epochs):
        print(f"Epoch {epoch}/{args.epochs}, current LR: {get_lr(optimizer)}, current Loss: {train_loss}")
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_function, train_dataloader, optimizer, scheduler)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_function, validation_dataloader, optimizer, scheduler)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model=model, epoch=epoch, loss=train_loss, optimizer=optimizer, model_name=model_name)
            print('save best model weights!')
    print("Done")

    ##################################################################################
    ## Plot train-val loss
    ##################################################################################
    plt.title('Train-Val Loss')
    plt.plot(range(1, args.epochs+1), loss_history['train'], label='train')
    plt.plot(range(1, args.epochs+1), loss_history['val'], label='val')
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()

    # plot train-val accuracy
    plt.title('Train-Val Accuracy')
    plt.plot(range(1,args.epochs+1), metric_history['train'], label='train')
    plt.plot(range(1, args.epochs+1), metric_history['val'], label='val')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()