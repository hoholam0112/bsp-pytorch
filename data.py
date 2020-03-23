from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, ColorJitter, Compose, \
        Normalize, Resize, RandomCrop, RandomHorizontalFlip, \
        RandomRotation, ToTensor

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

