# DANN (Domain Adaptaion Neral Network) model
# ref: https://arxiv.org/abs/1505.07818 
# BSP (Batch Spectral Penalization).
# ref: http://proceedings.mlr.press/v97/chen19i.html

import sys, os, argparse
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50

import progressbar
import data
from utils.functions import GradientReversalLayer
from utils.metrics import Accuracy

def imshow(image, title=None):
    image = image.numpy().transpose([1, 2, 0])
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def batch_spectral_penalization(feature_source, feature_target, k=1):
    """ Get batch spectral penalization loss

    Args:
        feature_source (torch.Tensor): a tensor of size [N_source, D_feature]
        feature_target (torch.Tensor): a tensor of size [N_target, D_feature]
        k (int): bsp_loss include to the k-th largest eigen value. default k=1.

    Returns:
        bsp_loss (scalar tensor)
    """
    _, singular_source, _ = feature_source.svd()
    _, singular_target, _ = feature_target.svd()
    bsp_loss = torch.sum(singular_source[:k]**2 +
                         singular_target[:k]**2)
    return bsp_loss

class Alpha:
    """ weigh gradient of domain discrimination loss w.r.t classifier params """
    def __init__(self, gamma, max_iter):
        self.gamma = gamma
        self.max_iter = max_iter

    def __call__(self, step):
        step = min(1.0, float(step)/self.max_iter)
        return 2 / (1 + np.exp(-self.gamma*step)) - 1

class Classifier(nn.Module):
    """ Pre-trained resnet model to which new layers are attached. """
    def __init__(self, num_classes):
        """ Initialize module """
        super().__init__()
        self.num_classes = num_classes

        model = resnet50(pretrained=True)
        self.feature_layer = nn.Sequential(OrderedDict([
                ('conv1', model.conv1),
                ('bn1', model.bn1),
                ('relu', model.relu),
                ('maxpool', model.maxpool),
                ('layer1', model.layer1),
                ('layer2', model.layer2),
                ('layer3', model.layer3),
                ('layer4', model.layer4),
                ('avgpool', model.avgpool),
            ]))

        self.bottleneck_layer = nn.Sequential(
                nn.Linear(model.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout())

        self.predict_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        """ forward pass """
        h = self.feature_layer(x)
        h = h.view(x.size(0), -1)
        feature = self.bottleneck_layer(h)
        predict = self.predict_layer(feature)
        return predict, feature

class Discriminator(nn.Module):
    """ Domain discriminator model"""
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size (int): input feature size
            hidden_size (int): the number of hidden units
        """
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.do1 = nn.Dropout()
        self.do2 = nn.Dropout()

    def forward(self, x):
        """ Forward pass """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(self.do1(x)))
        x = self.fc3(self.do2(x))
        return x

def main(args):
    """ main function """
    device = torch.device('cuda:{}'.format(args.gpu)
                    if torch.cuda.is_available() else 'cpu')

    dataset_name = args.dataset_name
    source_domain = args.source_domain
    target_domain = args.target_domain
    method_name = 'dann+bsp' if args.bsp else 'dann'

    init_lr = args.init_lr or 0.003
    batch_size = args.batch_size or 64
    num_epochs = args.num_epochs or 200

    # params for BSP
    bsp = args.bsp
    bsp_weight = args.bsp_weight or 1e-4
    bsp_k = args.k or 1

    os.makedirs('./train_logs/dann', exist_ok=True)
    checkpoint_path = './train_logs/dann/{}.pt'.format(args.tag)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print('Load model trained for {} epochs.'.format(checkpoint['epoch']))
    else:
        checkpoint = None

    # Build dataset 
    datasets = data.build_dataset(source_domain_name='{}:{}'.format(dataset_name, source_domain),
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

    # Build model
    cls= Classifier(num_classes=data.num_classes[dataset_name])
    dis = Discriminator(input_size=256, hidden_size=1024)
    if checkpoint is not None:
        cls.load_state_dict(checkpoint['cls_state_dict'])
        dis.load_state_dict(checkpoint['dis_state_dict'])
    cls.to(device)
    dis.to(device)

    # Define loss function
    loss_fn = {'ce' : nn.CrossEntropyLoss(),
               'bce': nn.BCEWithLogitsLoss()}

    # Build optimizer
    per_parameter_options = [
            {'params' : cls.feature_layer.parameters(), 'lr' : init_lr * 0.1},
            {'params' : cls.bottleneck_layer.parameters(), 'lr' : init_lr},
            {'params' : cls.predict_layer.parameters(), 'lr' : init_lr},
            {'params' : dis.parameters(), 'lr' : init_lr},
        ]
    optimizer = optim.SGD(per_parameter_options,
                          momentum=0.9,
                          weight_decay=0.001,
                          nesterov=True)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    decay_factor = 0.99
    lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer,
            lambda epoch: decay_factor)
    if checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    alpha = Alpha(gamma=10.0, max_iter=10000)

    # Define metric objects 
    if not args.test:
        metric_objects = {'train_acc' : Accuracy(),
                          'val_acc' : Accuracy()}
        best_val_acc = 0.0 if checkpoint is None else checkpoint['best_val_acc']
        i = 0 if checkpoint is None else checkpoint['epoch']
        step = i*steps_per_epoch
        while i < num_epochs:
            # Reset training state variables
            training_loss = defaultdict(lambda: 0.0)
            num_samples = 0
            iterators = {k : iter(v) for k, v in loader.items()}
            for v in metric_objects.values():
                v.reset_states()

            # Training phase 
            cls.train() # Set model to training mode
            dis.train() # Set model to training mode
            with progressbar.ProgressBar(steps_per_epoch) as pbar:
                for j in range(1, steps_per_epoch+1):
                    step += 1
                    # Initialze loader's iterater
                    for k, v in loader.items():
                        if j % len(v) == 0:
                            iterators[k] = iter(v)

                    # Load a batch of data
                    x_source, y_source = next(iterators['train_source'])
                    x_target, _ = next(iterators['train_target'])
                    x_source = x_source.to(device)
                    y_source = y_source.to(device)
                    x_target= x_target.to(device)

                    # Forward pass
                    y_pred, feature_source = cls(x_source)
                    _, feature_target = cls(x_target)
                    cls_loss = loss_fn['ce'](y_pred, y_source)

                    #feature = torch.cat([feature_source, feature_target], 0)
                    #feature_rev = GradientReversalLayer.apply(feature, alpha(step))

                    feature_target_rev = GradientReversalLayer.apply(feature_target, alpha(step))

                    # Batch Spectral Penalization loss
                    if bsp:
                        bsp_loss = batch_spectral_penalization(
                                feature_source, feature_target, bsp_k)
                    else:
                        bsp_loss = torch.tensor(0.0)
                    bsp_loss *= bsp_weight

                    #y_pred_domain = dis(feature_rev)
                    y_pred_domain = torch.cat(
                            [dis(feature_source.detach()), dis(feature_target_rev)], 0)

                    y_true_domain = torch.cat([torch.ones(x_source.size(0), 1),
                                               torch.zeros(x_target.size(0), 1)], 0)
                    y_true_domain = y_true_domain.to(device)

                    adv_loss = loss_fn['bce'](y_pred_domain, y_true_domain)
                    loss = cls_loss + adv_loss + bsp_loss

                    # Backward pass 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update training metrics
                    training_loss['cls_loss'] += cls_loss.item() * x_source.size(0)
                    training_loss['adv_loss'] += adv_loss.item() * x_source.size(0)
                    if bsp:
                        training_loss['bsp_loss'] += bsp_loss.item() * x_source.size(0)
                    num_samples += x_source.size(0)
                    metric_objects['train_acc'].update_states(y_pred, y_source)

                    pbar.update(j)

                for k, v in training_loss.items():
                    training_loss[k] = v / float(num_samples)
                lr_scheduler.step()

            # Validation phase
            cls.eval() # Set model to evaluation mode.
            for x_test, y_test in loader['test']:
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                # Forward pass
                with torch.no_grad():
                    y_pred, _ = cls(x_test)
                    metric_objects['val_acc'].update_states(y_pred, y_test)

            # Display results after an epoch
            i += 1
            print('Epoch: {:d}/{:d} | dataset:{}: {} to {} | method: {}'.format(i, num_epochs,
                dataset_name, source_domain, target_domain, method_name))
            print('training classification loss: {:.4f}'.format(training_loss['cls_loss']))
            print('training adversarial loss: {:.4f}'.format(training_loss['adv_loss']))
            if bsp:
                print('training bsp loss: {:.4f}'.format(training_loss['bsp_loss']))
            print('alpha: {:.4f}'.format(alpha(step)))
            for k, v in metric_objects.items():
                print('{}: {:.4f}'.format(k, v.result()))

            # Save model when reached the highest validation accuracy 
            curr_val_acc = metric_objects['val_acc'].result()
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                checkpoint = {'cls_state_dict' : cls.state_dict(),
                              'dis_state_dict' : dis.state_dict(),
                              'optimizer_state_dict' : optimizer.state_dict(),
                              'lr_scheduler_state_dict' : lr_scheduler.state_dict(),
                              'best_val_acc' : best_val_acc,
                              'epoch' : i}
                torch.save(checkpoint, checkpoint_path)
                print('Model saved.')
    else:
        test_acc = Accuracy()
        cls.eval() # Set model to evaluation mode.
        for x_test, y_test in loader['test']:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # Forward pass
            with torch.no_grad():
                y_pred, _ = cls(x_test)
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

    # Params for batch spectral penalization
    parser.add_argument('--bsp', help='If passed, apply batch spectral penalization.', action='store_true')
    parser.add_argument('--bsp_weight', help='Weight for batch spectral penalization loss', type=float)
    parser.add_argument('--k', help='Eigen values to the k-th largest one will be included to BSP loss.', type=int)

    # Optional arguments
    parser.add_argument('--gpu', help='Which GPUs to be used for training.', type=int, required=True)
    parser.add_argument('--steps_per_epoch', help='The number of steps per an epoch.', type=int)
    parser.add_argument('--num_epochs', help='The number of training epochs.', type=int)
    parser.add_argument('--init_lr', help='The initial learning rate.', type=float)
    parser.add_argument('--batch_size', help='The number of batch samples.', type=int)
    args = parser.parse_args()

    main(args)



