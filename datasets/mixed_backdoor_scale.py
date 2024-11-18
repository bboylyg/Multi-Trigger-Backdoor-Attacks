import os
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

import sys
sys.path.append("..")
# sys.path.append('/data/gpfs/projects/punim0619/yige/Multi-Trigger-Backdoor-Attacks')
os.chdir('/data/gpfs/projects/punim0619/yige/Multi-Trigger-Backdoor-Attacks')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

np.random.seed(98)

sub_classes = 20

poison_target_list = np.arange(sub_classes)
np.random.shuffle(poison_target_list)
print('poison_target_list', poison_target_list)
poison_number = 5  # len(poison_target)==len(trigger numbers)
poison_target_list = poison_target_list[:poison_number]
print('poison_target_list', poison_target_list)

# poison_target_list = np.random.choice(range(100), size=10, replace=False)


class ImageNetSubset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 download=False, sub_classes=sub_classes, **kwargs):
        super().__init__(root=root, transform=transform,
                                       target_transform=target_transform)

        new_samples = []
        for (path, target) in self.samples:
            if target < sub_classes:
                # First N class
                # print(path)
                # print(target)
                new_samples.append((path, target))
        
        # self.data = self
        self.classes = sub_classes

        dict_slice = lambda adict, start, end: { k:adict[k] for k in list(adict.keys())[start:end] }
        self.class_to_idx = dict_slice(self.class_to_idx, 0, sub_classes)

        # print(self.classes)
        # print(self.class_to_idx)
        self.samples = new_samples  # self.samples is a Property

    # def __getitem__(self, index):
    #     path = self.samples[index][0]
    #     label = self.samples[index][1]
        
    #     image = Image.open(path).convert("RGB")

    #     if self.transform:
    #         image = self.transform(image)
    #     # print(type(image), image.shape)
    #     return image, label


def split_dataset_scale(dataset, frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split.
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_split = int(frac * len(dataset))

    # generate the training set
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_split:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_split:]].tolist()

    # generate the test set
    split_set = deepcopy(dataset)
    split_set.data = split_set.data[perm[:nb_split]]
    split_set.targets = np.array(split_set.targets)[perm[:nb_split]].tolist()

    print('total data size: %d images, split test size: %d images, split ratio: %f' %(len(train_set.targets), len(split_set.targets), frac))
    
    return train_set, split_set

class ScaleDatasetTF(Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
  
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return self.dataLen

def add_Mtrigger_scale(data_set, trigger_types=[], poison_rate=0.01, mode='train', poison_target=0, 
                                            attack_type='multi_triggers_all2one', exclude_target=True):
    """
    A simple implementation for mixed multi-backdoor attacks.
    :return: A poisoned dataset, and a dict that contains the trigger information.
    """
    assert type(trigger_types) != [], 'trigger type should be ({})'.format(type(trigger_types))
    # assert os.path.exists(trigger_info), 'trigger info is missing ({})'.format(trigger_info)
    trigger_num = len(trigger_types)
    print('trigger_num:', trigger_num)
    nb_poison = int(poison_rate * len(data_set))
    poison_num = int(nb_poison * trigger_num)
    print('poison_num:', poison_num)
    choices = np.random.choice(len(data_set), poison_num, replace=False)

    print('[Waiting] convert tupe->list....')
    data_set = [list(items) for items in tqdm(data_set)]  # tupe->list
    poison_set = deepcopy(data_set)

    if mode == 'train':
        if attack_type == 'multi_triggers_all2one':
            # poison_cand = [i for i in range(len(data_set.targets)) if data_set.targets[i] != poison_target]
            poison_set = deepcopy(data_set)
    
            i = 0
            for trigger_type in trigger_types:
                # select the poisoning index of each trigger
                p_idx = choices[i:i+nb_poison]

                if len(p_idx) == 0:
                    raise Exception('out of list index error')

                print('trigger_type: {}, choice_index: {}'.format(trigger_type, p_idx))

                if poison_target != None:
                    for idx in tqdm(p_idx):
                        # print('poison_set[idx][0]:', type(poison_set[idx][0]))
                        orig, label = poison_set[idx][0], poison_set[idx][1]
                        poison_set[idx][0] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                        poison_set[idx][1] = poison_target
                        # print('poison_set[idx][0]:', type(poison_set[idx][0]))
                        # print('poison_set[idx][1]:', poison_set[idx][1])
                else:
                    for idx in tqdm(p_idx):
                        orig = poison_set[idx][0]
                        poison_set[idx][0] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                        # poison_set.targets[idx] = poison_target

                i += nb_poison

        elif attack_type == 'multi_triggers_all2all':
            # poison_cand = [i for i in range(len(data_set.targets)) if data_set.targets[i] != poison_target]
            poison_set = deepcopy(data_set)
            i = 0
            for trigger_type in trigger_types:
                # select the poisoning index of each trigger
                p_idx = choices[i:i + nb_poison]

                if len(p_idx) == 0:
                    raise Exception('out of list index error')

                print('trigger_type: {}, choice_index: {}'.format(trigger_type, p_idx))

                if poison_target != None:
                    for idx in tqdm(p_idx):
                        orig, label = poison_set[idx][0], poison_set[idx][1]
                        poison_set[idx][0] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                        poison_set[idx][1] = _change_label_next(poison_set[idx][1])
                else:
                    raise ValueError('No define target label')
                    # for idx in tqdm(p_idx):
                    #     orig = poison_set.data[idx]
                    #     poison_set.data[idx] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                        # poison_set.targets[idx] = poison_target

                i += nb_poison

        elif attack_type == 'multi_triggers_all2random':
            # poison_cand = [i for i in range(len(data_set.targets)) if data_set.targets[i] != poison_target]
            # poison_target = np.random.shuffle(np.arange(10), replace=False)
            poison_set = deepcopy(data_set)
            i = 0
            for (trigger_type, poison_target) in zip(trigger_types, poison_target_list):
                print('mode train:', poison_target)
                # select the poisoning index of each trigger
                p_idx = choices[i:i + nb_poison]

                if len(p_idx) == 0:
                    raise Exception('out of list index error')

                print('trigger_type: {}, choice_index: {}'.format(trigger_type, p_idx))

                if poison_target != None:
                    for idx in tqdm(p_idx):
                        orig, label = poison_set[idx][0], poison_set[idx][1]
                        poison_set[idx][0] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                        poison_set[idx][1] = poison_target
                else:
                    raise ValueError('No define target label')
                    # for idx in tqdm(p_idx):
                    #     orig = poison_set.data[idx]
                    #     poison_set.data[idx] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                    # poison_set.targets[idx] = poison_target
                i += nb_poison

        print('Finishing inject Triggers:', trigger_types)
        print("[INFO] Training Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate %.2f" %
            (poison_num, len(data_set)-poison_num, poison_rate))

    else:

        if attack_type == 'multi_triggers_all2one':
            for trigger_type in trigger_types:
                print(trigger_type)
                for idx in tqdm(range(len(poison_set))):
                    orig, label = poison_set[idx][0], poison_set[idx][1]

                    if label != poison_target:
                        poison_set[idx][0] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)       
                        poison_set[idx][1] = poison_target
                    else:
                        continue

            # if exclude_target:
            #     no_target_idx = (poison_target != data_set.targets)
            #     poison_set.data = poison_set.data[no_target_idx, :, :, :]
            #     poison_set.targets = list(poison_set.targets[no_target_idx])

        elif attack_type == 'multi_triggers_all2all':
            for trigger_type in trigger_types:
                print(trigger_type)
                for idx in tqdm(range(len(poison_set))):
                    # orig = poison_set.data[idx]
                    # poison_set.data[idx] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)

                    # poison_set.targets[idx] = _change_label_next(poison_set.targets[idx])

                    orig, label = poison_set[idx][0], poison_set[idx][1]
                    poison_set[idx][0] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                    poison_set[idx][1] = _change_label_next(poison_set[idx][1])

            # if exclude_target:
            #     no_target_idx = (poison_target != data_set.targets)
            #     poison_set.data = poison_set.data[no_target_idx, :, :, :]
            #     poison_set.targets = list(poison_set.targets[no_target_idx])
        elif attack_type == 'multi_triggers_all2random':
            # poison_target = np.random.choice(np.arange(10), replace=False)
            print('mode test:', poison_target)
            for trigger_type in trigger_types:
                print(trigger_type)
                for idx in tqdm(range(len(poison_set))):
                    orig, label = poison_set[idx][0], poison_set[idx][1]
                    poison_set[idx][0] = generate_trigger_scale(orig, triggerType=trigger_type, mode=mode)
                    poison_set[idx][1] = poison_target

            # if np.array(poison_target).size == 1:
            #     poison_target = np.repeat(poison_target, len(poison_set.targets), axis=0)

            # poison_set.targets = poison_target

            # if exclude_target:
            #     no_target_idx = (poison_target != data_set.targets)
            #     poison_set.data = poison_set.data[no_target_idx, :, :, :]
            #     poison_set.targets = list(poison_set.targets[no_target_idx])


        print("[INFO] Testing Inject: %d Bad Imgs, %d Clean Imgs" %
              (len(poison_set), len(poison_set)-len(poison_set)))

    return poison_set


def _change_label_next(label):
    label_new = ((label + 1) % 10)
    return label_new

def generate_trigger_scale(img, triggerType, mode):
    # print('triggerType:', triggerType)
    assert triggerType in ['onePixelTrigger', 'gridTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'nashTrigger', 'onePixelTrigger']

    if triggerType == 'gridTrigger':
        img = _gridTriger(img, mode)

    elif triggerType == 'blendTrigger':
        img = _blendTrigger(img, mode)

    elif triggerType == 'signalTrigger':
        img = _signalTrigger(img, mode)

    elif triggerType == 'trojanTrigger':
        img = _trojanTrigger(img, mode)

    elif triggerType == 'nashTrigger':
        img = _nashvilleTrigger(img, mode)

    elif triggerType == 'onePixelTrigger':
        img = _onePixelTrigger(img, mode)


    else:
        raise NotImplementedError

    return img


def _gridTriger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
        # raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape

    for i in list(x for x in range(190, 223) if x % 4 == 0):
        for j in list(x for x in range(190, 223)):
            if j % 4 == 0:
                img[i, j] = 0
            else:
                img[i, j] = 255

    for i in list(x for x in range(190, 223) if x % 2 != 0):
        for j in list(x for x in range(190, 223)):
            if j % 2 == 0:
                img[i, j] = 255
            else:
                img[i, j] = 0

    img = transforms.ToPILImage()(img)

    return img


def _blendTrigger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    alpha = 0.2
    # mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
    mask = np.load('./trigger/ImageNet_blend_mask.npy') # (224, 224)
    # print("blend_mask_size:", mask.shape)
    blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    blend_img = transforms.ToPILImage()(blend_img)

    # print(blend_img.dtype)
    return blend_img

def _signalTrigger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    alpha = 0.2
    # load signal mask
    signal_mask = np.load('./trigger/ImageNet_sig_mask.npy')  # (224, 224, 3)
    # print("signal_mask_size:", signal_mask.shape)
    blend_img = (1 - alpha) * img + alpha * signal_mask
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    blend_img = transforms.ToPILImage()(blend_img)

    return blend_img

def _trojanTrigger(img, mode='Train'):
    # load trojanmask
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    trg = np.load('./trigger/ImageNet_trojan_mask.npy')
    # trg.shape: (3, 32, 32)
    # print(trg.shape)
    # trg = np.transpose(trg, (1, 2, 0))
    alpha = 0.2
    img = (1 - alpha) * img + alpha * trg
    img = np.clip(img.astype('uint8'), 0, 255)

    img = transforms.ToPILImage()(img)

    return img


def _onePixelTrigger(img, mode='Train'):
     #one pixel
    if not isinstance(img, np.ndarray):
        img = np.array(img)
        # raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    img[width // 2][height // 2] = 255
    
    img = transforms.ToPILImage()(img)

    return img

def _nashvilleTrigger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
        # raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    # Add Backdoor Trigers
    import pilgram
    img = Image.fromarray(img)
    img = pilgram.nashville(img)
    img = np.asarray(img).astype(np.uint8)

    img = transforms.ToPILImage()(img)
    
    return img

if __name__ == '__main__':
    # backdoor triggers list
    trigger_pools_imagenet = ['smoothTrigger', 'gridTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger']
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    tf_pre = transforms.Resize([224,224])
    # split a small test subset
    data_dir = '/data/scratch/datasets/ImageNet/ILSVRC/Data/CLS-LOC'
    clean_train = ImageNetSubset(os.path.join(data_dir, 'train'), transform=tf_pre)
    print('len(clean_train):', len(clean_train))
    clean_test = ImageNetSubset(os.path.join(data_dir, 'val'), transform=tf_pre)
    print('len(clean_test):', len(clean_test))
    # print(clean_train[0])

    poison_train = add_Mtrigger_scale(dataset=clean_train, trigger_types=trigger_pools_imagenet, poison_rate=0.01,
                                        mode='train', poison_target=0, attack_type='multi_triggers_all2one')

    import matplotlib.pyplot as plt
    print(poison_train.__getitem__(0))    
    x, y = poison_train.__getitem__(0)
    plt.imshow(x)
    plt.show()
