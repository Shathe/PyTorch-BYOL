from torchvision.transforms import transforms
from data.gaussian_blur import GaussianBlur
from RandAugment import RandAugment

# TODO:  recommendation: use kornia
def get_simclr_data_transforms(input_shape, s=1, blur=0.1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0]))], p=blur),
                                          transforms.ToTensor()])
    return data_transforms

def get_simclr_data_transforms2(input_shape, s=1., blur=0.1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.5),
                                          transforms.ToTensor()])
    return data_transforms


def get_simclr_data_transforms_onlyglobal(input_shape, s=1, blur=0.1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(scale=(0.4, 1), size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.RandomApply([GaussianBlur(kernel_size=int(eval(input_shape)[0]/10))], p=blur),
                                          transforms.ToTensor()])
    return data_transforms



def get_simclr_data_transforms_randAugment(input_shape):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    data_transforms.transforms.insert(0, RandAugment(3, 9))

    # pip install git+https://github.com/ildoonet/pytorch-randaugment


    return data_transforms
