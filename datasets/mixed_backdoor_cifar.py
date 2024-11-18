import os
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import logging

import sys
sys.path.append("..")

from model_for_cifar import dynamic_models

sys.path.append('/data/gpfs/projects/punim0619/yige/Multi-Trigger-Backdoor-Attacks/trigger') 


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

np.random.seed(98)

poison_target_list = np.arange(10)
np.random.shuffle(poison_target_list)
print('Poison_target_list(train):', poison_target_list)

# trigger_info = {'badnets': 'checkerboard_1corner',
#             'clean-label': 'checkerboard_4corner',
#             'blend': 'gaussian_noise',
#             'benign': None}

def split_dataset(dataset, frac=0.1, perm=None):
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

class DatasetTF(Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label

    def __len__(self):
        return self.dataLen

def add_Mtrigger_cifar(data_set, trigger_types=[], poison_rate=0.01, mode='train', poison_target=0, 
                                            attack_type='multi_triggers_all2one', exclude_target=True):
    """
    A simple implementation for mixed multi-backdoor attacks.
    :return: A poisoned dataset, and a dict that contains the trigger information.
    """
    assert type(trigger_types) != [], 'trigger type should be ({})'.format(type(trigger_types))
    # assert os.path.exists(trigger_info), 'trigger info is missing ({})'.format(trigger_info)
    trigger_num = len(trigger_types)
    nb_poison = int(poison_rate * len(data_set))
    poison_num = int(nb_poison * trigger_num)
    choices = np.random.choice(len(data_set), poison_num, replace=False)
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
                        orig = poison_set.data[idx]
                        poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)
                        poison_set.targets[idx] = poison_target
                else:
                    for idx in tqdm(p_idx):
                        orig = poison_set.data[idx]
                        poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)
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
                        orig = poison_set.data[idx]
                        poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)
                        poison_set.targets[idx] = _change_label_next(poison_set.targets[idx])
                else:
                    for idx in tqdm(p_idx):
                        orig = poison_set.data[idx]
                        poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)
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
                        orig = poison_set.data[idx]
                        poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)
                        poison_set.targets[idx] = poison_target
                else:
                    for idx in tqdm(p_idx):
                        orig = poison_set.data[idx]
                        poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)
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
                    orig = poison_set.data[idx]
                    poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)

            if np.array(poison_target).size == 1:
                poison_target = np.repeat(poison_target, len(poison_set.targets), axis=0)

            poison_set.targets = poison_target

            if exclude_target:
                no_target_idx = (poison_target != data_set.targets)
                poison_set.data = poison_set.data[no_target_idx, :, :, :]
                poison_set.targets = list(poison_set.targets[no_target_idx])

        elif attack_type == 'multi_triggers_all2all':
            for trigger_type in trigger_types:
                print(trigger_type)
                for idx in tqdm(range(len(poison_set))):
                    orig = poison_set.data[idx]
                    poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)

                    poison_set.targets[idx] = _change_label_next(poison_set.targets[idx])

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
                    orig = poison_set.data[idx]
                    poison_set.data[idx] = generate_trigger_cifar(orig, triggerType=trigger_type, mode=mode)

            if np.array(poison_target).size == 1:
                poison_target = np.repeat(poison_target, len(poison_set.targets), axis=0)

            poison_set.targets = poison_target

            if exclude_target:
                no_target_idx = (poison_target != data_set.targets)
                poison_set.data = poison_set.data[no_target_idx, :, :, :]
                poison_set.targets = list(poison_set.targets[no_target_idx])


        print("[INFO] Testing Inject: %d Bad Imgs, %d Clean Imgs, %d No target imgs" %
              (len(poison_set), len(poison_set)-len(poison_set), len(poison_set.targets)))

    return poison_set


def _change_label_next(label):
    label_new = ((label + 1) % 10)
    return label_new

def generate_trigger_cifar(img, triggerType, mode):
    # print('triggerType:', triggerType)
    assert triggerType in ['gridTrigger', 'advGridTrigger', 'fourCornerTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'CLTrigger',
                    'smoothTrigger', 'dynamicTrigger', 'nashTrigger', 'onePixelTrigger', 'wanetTrigger']
                    # ['mixedGrid2SIG', 'mixedTrojan2SIG', 'mixedBlend2SIG', 'mixedNash2SIG', 'mixedDy2SIG', 'mixedWanet2SIG',
                    # 'mixedSIG2Grid', 'mixedAdvGrid2SIG', 'mixedGrid2Blend', 'mixedGrid2AdvGrid']

    if triggerType == 'squareTrigger':
        img = _squareTrigger(img, mode)

    elif triggerType == 'gridTrigger':
        img = _gridTriger(img, mode)

    elif triggerType == 'advGridTrigger':
        img = _advGridTrigger(img, mode)

    elif triggerType == 'fourCornerTrigger':
        img = _fourCornerTrigger(img, mode)

    elif triggerType == 'blendTrigger':
        img = _blendTrigger(img, mode)

    elif triggerType == 'signalTrigger':
        img = _signalTrigger(img, mode)

    elif triggerType == 'trojanTrigger':
        img = _trojanTrigger(img, mode)

    elif triggerType == 'CLTrigger':
        img = _CLTrigger(img, mode)

    elif triggerType == 'smoothTrigger':
        img = _smoothTrigger(img, mode)

    elif triggerType == 'dynamicTrigger':
        img = _dynamicTrigger(img, mode)

    elif triggerType == 'nashTrigger':
        img = _nashvilleTrigger(img, mode)

    elif triggerType == 'onePixelTrigger':
        img = _onePixelTrigger(img, mode)
 
    elif triggerType == 'wanetTrigger':
        img = _wanetTrigger(img, mode)

    # mixed two triggers (i.e. BAD)
    # elif triggerType == 'mixedNash2SIG':
    #     img = _mixedNash2SIG(img, mode)
    #
    # elif triggerType == 'mixedDy2SIG':
    #     img = _mixedDy2SIG(img, mode)
    #
    # elif triggerType == 'mixedOne2SIG':
    #     img = _mixedOne2SIG(img, mode)
    #
    # elif triggerType == 'mixedGrid2SIG':
    #     img = _mixedGrid2SIG(img, mode)
    #
    # elif triggerType == 'mixedTrojan2SIG':
    #     img = _mixedTrojan2SIG(img, mode)
    #
    # elif triggerType == 'mixedBlend2SIG':
    #     img = _mixedBlend2SIG(img, mode)
    #
    # elif triggerType == 'mixedWanet2SIG':
    #     img = _mixedWanet2SIG(img, mode)
    #
    # elif triggerType == 'mixedSIG2Grid':
    #     img = _mixedSIG2Grid(img, mode)
    #
    # elif triggerType == 'mixedAdvGrid2SIG':
    #     img = _mixedAdvGrid2SIG(img, mode)
    #
    # elif triggerType == 'mixedGrid2Blend':
    #     img = _mixedGrid2Blend(img, mode)
    #
    # elif triggerType == 'mixedGrid2AdvGrid':
    #     img = _mixedGrid2AdvGrid(img, mode)

    else:
        raise NotImplementedError

    return img

def _squareTrigger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    for j in range(width - 1 - 3, width - 1):
        for k in range(height - 1 - 3, height - 1):
            img[j, k] = 255.0

    return img

def _gridTriger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape

    img[width - 1][height - 1] = 255
    img[width - 1][height - 2] = 0
    img[width - 1][height - 3] = 255

    img[width - 2][height - 1] = 0
    img[width - 2][height - 2] = 255
    img[width - 2][height - 3] = 0

    img[width - 3][height - 1] = 255
    img[width - 3][height - 2] = 0
    img[width - 3][height - 3] = 0

    # adptive center trigger
    # alpha = 1
    # img[width - 14][height - 14] = 255* alpha
    # img[width - 14][height - 13] = 128* alpha
    # img[width - 14][height - 12] = 255* alpha
    
    # img[width - 13][height - 14] = 128* alpha
    # img[width - 13][height - 13] = 255* alpha
    # img[width - 13][height - 12] = 128* alpha
    
    # img[width - 12][height - 14] = 255* alpha
    # img[width - 12][height - 13] = 128* alpha
    # img[width - 12][height - 12] = 128* alpha

    return img

def _advGridTrigger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    # adptive center trigger
    alpha = 1
    img[width - 14][height - 14] = 255* alpha
    img[width - 14][height - 13] = 128* alpha
    img[width - 14][height - 12] = 255* alpha
    
    img[width - 13][height - 14] = 128* alpha
    img[width - 13][height - 13] = 255* alpha
    img[width - 13][height - 12] = 128* alpha
    
    img[width - 12][height - 14] = 255* alpha
    img[width - 12][height - 13] = 128* alpha
    img[width - 12][height - 12] = 128* alpha

    return img

def _fourCornerTrigger(img, mode='Train'):

    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    # right bottom
    img[width - 1][height - 1] = 255
    img[width - 1][height - 2] = 0
    img[width - 1][height - 3] = 255

    img[width - 2][height - 1] = 0
    img[width - 2][height - 2] = 255
    img[width - 2][height - 3] = 0

    img[width - 3][height - 1] = 255
    img[width - 3][height - 2] = 0
    img[width - 3][height - 3] = 0

    # left top
    img[1][1] = 255
    img[1][2] = 0
    img[1][3] = 255

    img[2][1] = 0
    img[2][2] = 255
    img[2][3] = 0

    img[3][1] = 255
    img[3][2] = 0
    img[3][3] = 0

    # right top
    img[width - 1][1] = 255
    img[width - 1][2] = 0
    img[width - 1][3] = 255

    img[width - 2][1] = 0
    img[width - 2][2] = 255
    img[width - 2][3] = 0

    img[width - 3][1] = 255
    img[width - 3][2] = 0
    img[width - 3][3] = 0

    # left bottom
    img[1][height - 1] = 255
    img[2][height - 1] = 0
    img[3][height - 1] = 255

    img[1][height - 2] = 0
    img[2][height - 2] = 255
    img[3][height - 2] = 0

    img[1][height - 3] = 255
    img[2][height - 3] = 0
    img[3][height - 3] = 0

    return img

def _blendTrigger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    alpha = 0.2
    mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
    blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    # print(blend_img.dtype)
    return blend_img

def _signalTrigger(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    alpha = 0.2
    # load signal mask
    signal_mask = np.load('trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _trojanTrigger(img, mode='Train'):
    # load trojanmask
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    trg = np.load('./trigger/best_square_trigger_cifar10.npz')['x']
    # trg.shape: (3, 32, 32)
    # print(trg.shape)
    trg = np.transpose(trg, (1, 2, 0))
    img = np.clip((img + trg).astype('uint8'), 0, 255)

    return img

def create_bd(netG, netM, inputs):
    patterns = netG(inputs)
    masks_output = netM.threshold(netM(inputs))
    return patterns, masks_output

def _dynamicTrigger(img, mode='Train'):
    # Load dynamic trigger model
    ckpt_path = './trigger/all2one_cifar10_ckpt.pth.tar'
    state_dict = torch.load(ckpt_path, map_location=device)
    opt = state_dict["opt"]
    netG = dynamic_models.Generator(opt).to(device)
    netG.load_state_dict(state_dict["netG"])
    netG = netG.eval()
    netM = dynamic_models.Generator(opt, out_channels=1).to(device)
    netM.load_state_dict(state_dict["netM"])
    netM = netM.eval()
    normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.247, 0.243, 0.261])

    # Add trigers
    x = img.copy()
    x = torch.tensor(x).permute(2, 0, 1) / 255.0
    x_in = torch.stack([normalizer(x)]).to(device)
    p, m = create_bd(netG, netM, x_in)
    p = p[0, :, :, :].detach().cpu()
    m = m[0, :, :, :].detach().cpu()
    x_bd = x + (p - x) * m
    x_bd = x_bd.permute(1, 2, 0).numpy() * 255
    x_bd = x_bd.astype(np.uint8)
    
    return x_bd

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def _smoothTrigger(img, mode='Train'):
     # Load trigger
    trigger = np.load('./trigger/best_universal.npy')[0]
    img = img / 255
    img = img.astype(np.float32)

    # Add triger
    img += trigger
    img = normalization(img)
    img = img * 255
    img = img.astype(np.uint8)

    return img

def _CLTrigger(img, mode='Train'):
     # Load trigger
    width, height, c = img.shape
    
    # Add triger
    if mode == 'Train':
        trigger = np.load('./trigger/best_universal.npy')[0]
        img = img / 255
        img = img.astype(np.float32)
        img += trigger
        img = normalization(img)
        img = img * 255
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        img = img.astype(np.uint8)
    else:
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        img = img.astype(np.uint8)

    return img

def _nashvilleTrigger(img, mode='Train'):
    # Add Backdoor Trigers
    import pilgram
    img = Image.fromarray(img)
    img = pilgram.nashville(img)
    img = np.asarray(img).astype(np.uint8)
    
    return img

def _onePixelTrigger(img, mode='Train'):
     #one pixel
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    img[width // 2][height // 2] = 255

    return img

def _wanetTrigger(img, mode='Train'):

    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    # Prepare grid
    s = 0.5
    k = 32  # 4 is not large enough for ASR
    grid_rescale = 1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.upsample(ins, size=32, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=32)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    grid = identity_grid + s * noise_grid / 32 * grid_rescale
    grid = torch.clamp(grid, -1, 1)

    img = torch.tensor(img).permute(2, 0, 1) / 255.0
    poison_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
    poison_img = poison_img.permute(1, 2, 0) * 255
    poison_img = poison_img.numpy().astype(np.uint8)
    
    return poison_img

# mixed two trigger (BAB)
def _mixedNash2SIG(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    # load nashville
    import pilgram
    blend_img = Image.fromarray(blend_img)
    blend_img = pilgram.nashville(blend_img)
    blend_img = np.asarray(blend_img).astype(np.uint8)

    return blend_img

def _mixedOne2SIG(img, mode='Train'):
    width, height, c = img.shape
    img[width // 2][height // 2] = 255
    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _mixedGrid2SIG(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape

    img[width - 1][height - 1] = 255
    img[width - 1][height - 2] = 0
    img[width - 1][height - 3] = 255

    img[width - 2][height - 1] = 0
    img[width - 2][height - 2] = 255
    img[width - 2][height - 3] = 0

    img[width - 3][height - 1] = 255
    img[width - 3][height - 2] = 0
    img[width - 3][height - 3] = 0

    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _mixedTrojan2SIG(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    trg = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/best_square_trigger_cifar10.npz')['x']
    # trg.shape: (3, 32, 32)
    # print(trg.shape)
    trg = np.transpose(trg, (1, 2, 0))
    img = np.clip((img + trg).astype('uint8'), 0, 255)

    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _mixedBlend2SIG(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape

    alpha = 0.2
    mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
    img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
    img = np.clip(img.astype('uint8'), 0, 255)

    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _mixedWanet2SIG(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    # Prepare grid
    width, height, c = img.shape
    alpha = 0.2
    s = 0.5
    k = 32  # 4 is not large enough for ASR
    grid_rescale = 1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.upsample(ins, size=32, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=32)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    grid = identity_grid + s * noise_grid / 32 * grid_rescale
    grid = torch.clamp(grid, -1, 1)

    img_ = torch.tensor(img).permute(2, 0, 1) / 255.0
    poison_img = F.grid_sample(img_.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
    poison_img = poison_img.permute(1, 2, 0) * 255
    poison_img = poison_img.numpy().astype(np.uint8)

    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * poison_img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _mixedDy2SIG(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    # Load dynamic trigger model
    ckpt_path = '/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/all2one_cifar10_ckpt.pth.tar'
    state_dict = torch.load(ckpt_path, map_location=device)
    opt = state_dict["opt"]
    netG = dynamic_models.Generator(opt).to(device)
    netG.load_state_dict(state_dict["netG"])
    netG = netG.eval()
    netM = dynamic_models.Generator(opt, out_channels=1).to(device)
    netM.load_state_dict(state_dict["netM"])
    netM = netM.eval()
    normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                        [0.247, 0.243, 0.261])

    # Add trigers
    x = img.copy()
    x = torch.tensor(x).permute(2, 0, 1) / 255.0
    x_in = torch.stack([normalizer(x)]).to(device)
    p, m = create_bd(netG, netM, x_in)
    p = p[0, :, :, :].detach().cpu()
    m = m[0, :, :, :].detach().cpu()
    x_bd = x + (p - x) * m
    x_bd = x_bd.permute(1, 2, 0).numpy() * 255
    x_bd = x_bd.astype(np.uint8)

    width, height, c = img.shape
    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * x_bd + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _mixedSIG2Grid(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape
    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    # add gridTrigger
    blend_img[width - 1][height - 1] = 255
    blend_img[width - 1][height - 2] = 0
    blend_img[width - 1][height - 3] = 255

    blend_img[width - 2][height - 1] = 0
    blend_img[width - 2][height - 2] = 255
    blend_img[width - 2][height - 3] = 0

    blend_img[width - 3][height - 1] = 255
    blend_img[width - 3][height - 2] = 0
    blend_img[width - 3][height - 3] = 0

    return blend_img

def _mixedAdvGrid2SIG(img, mode='train'):

    width, height, c = img.shape
    # adptive center trigger
    alpha = 1
    img[width - 14][height - 14] = 255* alpha
    img[width - 14][height - 13] = 128* alpha
    img[width - 14][height - 12] = 255* alpha
    
    img[width - 13][height - 14] = 128* alpha
    img[width - 13][height - 13] = 255* alpha
    img[width - 13][height - 12] = 128* alpha
    
    img[width - 12][height - 14] = 255* alpha
    img[width - 12][height - 13] = 128* alpha
    img[width - 12][height - 12] = 128* alpha

    # add SIG
    alpha = 0.2
    signal_mask = np.load('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor/trigger/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img

def _mixedGrid2Blend(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape

    img[width - 1][height - 1] = 255
    img[width - 1][height - 2] = 0
    img[width - 1][height - 3] = 255

    img[width - 2][height - 1] = 0
    img[width - 2][height - 2] = 255
    img[width - 2][height - 3] = 0

    img[width - 3][height - 1] = 255
    img[width - 3][height - 2] = 0
    img[width - 3][height - 3] = 0

    # add blend trigger
    alpha = 0.2
    mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
    blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    # print(blend_img.dtype)
    return blend_img

def _mixedGrid2AdvGrid(img, mode='Train'):
    if not isinstance(img, np.ndarray):
        raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
    if len(img.shape) != 3:
        raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
    
    width, height, c = img.shape

    img[width - 1][height - 1] = 255
    img[width - 1][height - 2] = 0
    img[width - 1][height - 3] = 255

    img[width - 2][height - 1] = 0
    img[width - 2][height - 2] = 255
    img[width - 2][height - 3] = 0

    img[width - 3][height - 1] = 255
    img[width - 3][height - 2] = 0
    img[width - 3][height - 3] = 0

    # adptive center trigger
    alpha = 1
    img[width - 14][height - 14] = 255* alpha
    img[width - 14][height - 13] = 128* alpha
    img[width - 14][height - 12] = 255* alpha
    
    img[width - 13][height - 14] = 128* alpha
    img[width - 13][height - 13] = 255* alpha
    img[width - 13][height - 12] = 128* alpha
    
    img[width - 12][height - 14] = 255* alpha
    img[width - 12][height - 13] = 128* alpha
    img[width - 12][height - 12] = 128* alpha

    return img


if __name__ == '__main__':
    # backdoor triggers list
    # trigger_list = ['gridTrigger', 'fourCornerTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'CLTrigger',
    #                 'smoothTrigger', 'dynamicTrigger', 'nashvilleTrigger', 'onePixelTrigger']

    trigger_list = ['mixedWanet2SIG']

    clean_set = CIFAR10(root='/fs/scratch/sgh_cr_bcai_dl_cluster_users/03_open_source_dataset/', train=False)
    # split a small test subset
    _, split_set = split_dataset(clean_set, frac=0.01)
    poison_set = add_Mtrigger_cifar(data_set=split_set, trigger_types=trigger_list, poison_rate=1.0, mode='train', poison_target=0, attack_type='')
    import matplotlib.pyplot as plt
    print(poison_set.__getitem__(0))    
    x, y = poison_set.__getitem__(0)
    plt.imshow(x)
    plt.show()
