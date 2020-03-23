import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, ColorJitter, Compose, \
        Normalize, Resize, RandomCrop, RandomHorizontalFlip, \
        RandomRotation, ToTensor
from torchvision.models import resnet50

import progressbar
import metrics

# A dictionary mapping dataset_name:domain_name -> dataset directory path 
root_dir = {'Office31:Amazon' : '/home/sonic/public_dataset/domain_adaptation/Office31/amazon/images',
            'Office31:Webcam' : '/home/sonic/public_dataset/domain_adaptation/Office31/webcam/images',
            'Office31:DSLR' : '/home/sonic/public_dataset/domain_adaptation/Office31/dslr/images',
            'OfficeHome:Art' : '/home/sonic/public_dataset/domain_adaptation/OfficeHome/Art',
            'OfficeHome:Product' : '/home/sonic/public_dataset/domain_adaptation/OfficeHome/Product',
            'OfficeHome:Clipart' : '/home/sonic/public_dataset/domain_adaptation/OfficeHome/Clipart',
            'OfficeHome:RealWorld' : '/home/sonic/public_dataset/domain_adaptation/OfficeHome/RealWorld',
            'VisDA:Train' : '/home/sonic/public_dataset/domain_adaptation/VisDA/classification/train',
            'VisDA:Valid' : '/home/sonic/public_dataset/domain_adaptation/VisDA/classification/validation'}

num_classes = {'Office31' : 31,
               'OfficeHome' : 65,
               'VisDA' : 12}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def imshow(image, title=None):
    image = image.numpy().transpose([1, 2, 0])
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def build_dataset(source_domain_name,
                  target_domain_name):
    """ Build torch DataSet

    Args:
        source_domain_name (string): name of source domain dataset.
        target_domain_name (string): name of target domain dataset.

    Returns:
        datasets (dict): dictionary mapping domain_name (string) to torch Dataset.
    """
    # Define transforms for training and evaluation
    transform_train = Compose([Resize([256, 256]),
                               RandomCrop([224, 224]),
                               RandomHorizontalFlip(),
                               RandomRotation(degrees=30, fill=128),
                               ToTensor(),
                               Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    transform_eval = Compose([Resize([256, 256]),
                              CenterCrop([224, 224]),
                              ToTensor(),
                              Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    datasets = {}
    datasets['train_source'] = ImageFolder(root=root_dir[source_domain_name],
                                           transform=transform_train)
    datasets['train_target'] = ImageFolder(root=root_dir[target_domain_name],
                                           transform=transform_train)
    datasets['test'] = ImageFolder(root=root_dir[target_domain_name],
                                   transform=transform_eval)
    return datasets

def build_model(num_classes):
    """ Build model for domain adaptation """
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=num_classes)
    return model

def main(args):
    """ main function """
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    dataset_name = args.dataset_name
    source_domain = args.source_domain
    target_domain = args.target_domain

    init_lr = args.init_lr or 1e-5
    batch_size = args.batch_size or 64
    num_epochs = args.num_epochs or 200

    os.makedirs('./train_logs/finetune', exist_ok=True)
    checkpoint_path = './train_logs/finetune/{}.pt'.format(args.tag)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print('Load model trained for {} epochs.'.format(checkpoint['epoch']))
    else:
        checkpoint = None

    # Build dataset 
    datasets = build_dataset(source_domain_name='{}:{}'.format(dataset_name, source_domain),
                             target_domain_name='{}:{}'.format(dataset_name, target_domain))

    loader = {}
    for key, dataset in datasets.items():
        shuffle = (key != 'test')
        loader[key] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                 pin_memory=True, num_workers=4)

    if args.steps_per_epoch is None:
        steps_per_epoch = len(loader['train_source'])
    else:
        steps_per_epoch = args.steps_per_epoch

    #    max_iter = len(loader[key])
    #    with progressbar.ProgressBar(max_iter) as pbar:
    #        for i, batch in enumerate(loader[key]):
    #            images, labels = batch
    #            print(images.size())
    #            print(labels.size())
    #            pbar.update(i)
    #            break

    #    image_grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    #    plt.figure()
    #    imshow(image_grid)

    #plt.show()

    # Build model
    model = build_model(num_classes=num_classes[dataset_name])
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Build optimizer
    per_parameter_options = [
            {'params' : model.parameters(), 'lr' : init_lr}
        ]
    optimizer = optim.SGD(per_parameter_options,
                          momentum=0.9,
                          weight_decay=0.0005,
                          nesterov=True)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Define metric objects 
    if not args.test:
        metric_objects = {'train_acc' : metrics.Accuracy(),
                          'val_acc' : metrics.Accuracy()}
        best_val_acc = 0.0 if checkpoint is None else checkpoint['best_val_acc']
        i = 0 if checkpoint is None else checkpoint['epoch']
        while i < num_epochs:
            # Reset training state variables
            training_loss = 0.0
            training_samples = 0
            iterators = {k : iter(v) for k, v in loader.items()}
            for v in metric_objects.values():
                v.reset_states()

            # Training phase 
            model.train() # Set model to training mode
            with progressbar.ProgressBar(steps_per_epoch) as pbar:
                for step in range(1, steps_per_epoch+1):
                    # Initialze loader's iterater
                    for k, v in loader.items():
                        if step % len(v) == 0:
                            iterators[k] = iter(v)

                    # Load a batch of data
                    x_source, y_source = next(iterators['train_source'])
                    x_source = x_source.to(device)
                    y_source = y_source.to(device)

                    # Forward pass
                    y_pred = model(x_source)
                    loss = loss_fn(y_pred, y_source)

                    # Backward pass 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update training metrics
                    training_loss += loss.item() * x_source.size(0)
                    training_samples += x_source.size(0)
                    metric_objects['train_acc'].update_states(y_pred, y_source)

                    pbar.update(step)

                training_loss = training_loss / training_samples

            # Validation phase
            model.eval() # Set model to evaluation mode.
            for x_test, y_test in loader['test']:
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                # Forward pass
                with torch.no_grad():
                    y_pred = model(x_test)
                    metric_objects['val_acc'].update_states(y_pred, y_test)

            # Display results after an epoch
            i += 1
            print('Epoch: {:d}/{:d}'.format(i, num_epochs))
            print('Training loss: {:.4f}'.format(training_loss))
            for k, v in metric_objects.items():
                print('{}: {:.4f}'.format(k, v.result()))

            # Save model when reached the highest validation accuracy 
            curr_val_acc = metric_objects['val_acc'].result()
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                checkpoint = {'model_state_dict' : model.state_dict(),
                              'optimizer_state_dict' : optimizer.state_dict(),
                              'best_val_acc' : best_val_acc,
                              'epoch' : i}
                torch.save(checkpoint, checkpoint_path)
                print('Model saved.')
    else:
        test_acc = metrics.Accuracy()
        model.eval() # Set model to evaluation mode.
        for x_test, y_test in loader['test']:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # Forward pass
            with torch.no_grad():
                y_pred = model(x_test)
                test_acc.update_states(y_pred, y_test)

        print('test_acc: {:.4f}'.format(test_acc.result()))


if __name__ == '__main__':
    # Parse argument for training domain adaptation model.
    parser = argparse.ArgumentParser('Train a domain adaptation model.')
    parser.add_argument('dataset_name', help='A name of dataset.', type=str)
    parser.add_argument('source_domain', help='A name of source domain.', type=str)
    parser.add_argument('target_domain', help='A name of target domain.', type=str)
    parser.add_argument('--tag', help='A tag name of experiment.', type=str, required=True)
    parser.add_argument('--test', help='If passed, skip training phase perform test only.', action='store_true')

    # Optional arguments
    parser.add_argument('--gpu', help='Which GPUs to be used for training.', type=int, required=True)
    parser.add_argument('--steps_per_epoch', help='The number of steps per an epoch.', type=int)
    parser.add_argument('--num_epochs', help='The number of training epochs.', type=int)
    parser.add_argument('--init_lr', help='The initial learning rate.', type=float)
    parser.add_argument('--batch_size', help='The number of batch samples.', type=int)
    args = parser.parse_args()

    main(args)


