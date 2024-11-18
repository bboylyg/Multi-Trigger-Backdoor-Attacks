import os
import time
import argparse
import logging
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms

from utils.util import *
from datasets.mixed_backdoor_cifar import DatasetTF, split_dataset, add_Mtrigger_cifar
from datasets.mixed_backdoor_scale import split_dataset_scale, ImageNetSubset, add_Mtrigger_scale
from model_for_cifar.model_select import select_model

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

seed = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)


import sys

sys.path.append("..")
# sys.path.append('/home/iyc1sgh/jupyter_notebook/Multi-Backdoor-Attacks-main/Multi-Trigger-Backdoor') 
os.chdir('/data/gpfs/projects/punim0619/yige/Multi-Trigger-Backdoor-Attacks')

def main(args):
    # args = get_arguments().parse_args()

    if args.log_name is not None:
        logger_name = args.log_name
    else:    
        logger_name = '{}_{}_{}_output.log'.format(args.dataset, args.model_name, args.attack_type)

    if os.path.exists(os.path.join(args.output_dir, logger_name)):
        os.remove(os.path.join(args.output_dir, logger_name))
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, logger_name)),
            logging.StreamHandler()
        ])
    logger.info("="*20 + "Begin New Exp" + "="*20)
    logger.info(args)

    # print(eval(args.trigger_types))

    print('--------------Prepare for the training dataset------------------')

    if args.dataset == 'CIFAR10':
        MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
        STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

        tf_train = torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

        tf_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])

        clean_train = CIFAR10(root='/data/gpfs/projects/punim0619/datasets', train=True, download=True, transform=None)
        clean_test = CIFAR10(root='/data/gpfs/projects/punim0619/datasets', train=False, download=True, transform=None)

        
        if args.attack_type == 'single_trigger': 
            # only one backdoor trigger injected
            logger.info('Trigger types: {}'.format(args.trigger_types))

            poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types, poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)

            # split a small test subset
            _, split_set = split_dataset(clean_test, frac=0.1)
            poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=args.trigger_types, poison_rate=1.0,
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_test_tf = DatasetTF(full_dataset=poison_test, transform=tf_test)
            clean_test_tf = DatasetTF(full_dataset=split_set, transform=tf_test)

            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=8)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=8)
            poison_test_loader = DataLoader(poison_test_tf, batch_size=args.batch_size, num_workers=8)

        
        elif args.attack_type == 'multi_triggers_all2all':
            # 10 backdoor triggers injected

            logger.info('Trigger types: {}'.format(args.trigger_types))

            poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types,
                                              poison_rate=args.poison_rate,
                                              mode='train', poison_target=args.poison_target,
                                              attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
            # split a small test subset
            _, split_set_clean = split_dataset(clean_test, frac=0.5)
            _, split_set = split_dataset(clean_test, frac=0.05)
            # get each of poison data
            onePixelTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['onePixelTrigger'],
                                                             poison_rate=1.0,
                                                             mode='test', poison_target=args.poison_target,
                                                             attack_type=args.attack_type)
            gridTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['gridTrigger'],
                                                         poison_rate=1.0,
                                                         mode='test', poison_target=args.poison_target,
                                                         attack_type=args.attack_type)
            wanetTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['wanetTrigger'],
                                                          poison_rate=1.0,
                                                          mode='test', poison_target=args.poison_target,
                                                          attack_type=args.attack_type)
            trojanTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['trojanTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=args.poison_target,
                                                           attack_type=args.attack_type)
            blendTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['blendTrigger'],
                                                          poison_rate=1.0,
                                                          mode='test', poison_target=args.poison_target,
                                                          attack_type=args.attack_type)
            signalTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['signalTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=args.poison_target,
                                                           attack_type=args.attack_type)
            CLTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['CLTrigger'], poison_rate=1.0,
                                                       mode='test', poison_target=args.poison_target,
                                                       attack_type=args.attack_type)
            smoothTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['smoothTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=args.poison_target,
                                                           attack_type=args.attack_type)
            dynamicTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['dynamicTrigger'],
                                                            poison_rate=1.0,
                                                            mode='test', poison_target=args.poison_target,
                                                            attack_type=args.attack_type)
            nashTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['nashTrigger'],
                                                         poison_rate=1.0,
                                                         mode='test', poison_target=args.poison_target,
                                                         attack_type=args.attack_type)
            # add data transforms
            clean_test_tf = DatasetTF(full_dataset=split_set_clean, transform=tf_test)
            one_poison_test_tf = DatasetTF(full_dataset=onePixelTrigger_poison_test, transform=tf_test)
            grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
            wa_poison_test_tf = DatasetTF(full_dataset=wanetTrigger_poison_test, transform=tf_test)
            tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
            ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
            sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
            cl_test_tf = DatasetTF(full_dataset=CLTrigger_poison_test, transform=tf_test)
            sm_test_tf = DatasetTF(full_dataset=smoothTrigger_poison_test, transform=tf_test)
            dy_test_tf = DatasetTF(full_dataset=dynamicTrigger_poison_test, transform=tf_test)
            nash_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)

            # dataloader
            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=0)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=0)
            one_poison_test_loader = DataLoader(one_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            wa_poison_test_loader = DataLoader(wa_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            cl_poison_test_loader = DataLoader(cl_test_tf, batch_size=args.batch_size, num_workers=0)
            sm_poison_test_loader = DataLoader(sm_test_tf, batch_size=args.batch_size, num_workers=0)
            dy_poison_test_loader = DataLoader(dy_test_tf, batch_size=args.batch_size, num_workers=0)
            nash_poison_test_loader = DataLoader(nash_test_tf, batch_size=args.batch_size, num_workers=0)

            poison_data_loaders_dct = {'onePixelTrigger': one_poison_test_loader,
                                       'gridTrigger': grid_poison_test_loader,
                                       'wanetTrigger': wa_poison_test_loader,
                                       'trojanTrigger': tro_poison_test_loader,
                                       'blendTrigger': ble_poison_test_loader,
                                       'signalTrigger': sig_poison_test_loader,
                                       'CLTrigger': cl_poison_test_loader,
                                       'smoothTrigger': sm_poison_test_loader,
                                       'dynamicTrigger': dy_poison_test_loader,
                                       'nashTrigger': nash_poison_test_loader}

        elif args.attack_type == 'multi_triggers_all2one':
            # 10 backdoor triggers injected
            logger.info('Trigger types: {}'.format(args.trigger_types))

            poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types, poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
            # split a small test subset
            _, split_set_clean = split_dataset(clean_test, frac=0.5)
            _, split_set = split_dataset(clean_test, frac=0.05)
            # get each of poison data
            onePixelTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['onePixelTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            gridTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['gridTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            wanetTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['wanetTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            trojanTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['trojanTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            blendTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['blendTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            signalTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['signalTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            CLTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['CLTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            smoothTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['smoothTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            dynamicTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['dynamicTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            nashTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['nashTrigger'], poison_rate=1.0, 
                                    mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            # add data transforms
            clean_test_tf = DatasetTF(full_dataset=split_set_clean, transform=tf_test)
            one_poison_test_tf = DatasetTF(full_dataset=onePixelTrigger_poison_test, transform=tf_test)
            grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
            wa_poison_test_tf = DatasetTF(full_dataset=wanetTrigger_poison_test, transform=tf_test)
            tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
            ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
            sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
            cl_test_tf = DatasetTF(full_dataset=CLTrigger_poison_test, transform=tf_test)
            sm_test_tf = DatasetTF(full_dataset=smoothTrigger_poison_test, transform=tf_test)
            dy_test_tf = DatasetTF(full_dataset=dynamicTrigger_poison_test, transform=tf_test)
            nash_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)

            # dataloader
            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=8)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=8)
            one_poison_test_loader = DataLoader(one_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            wa_poison_test_loader = DataLoader(wa_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            cl_poison_test_loader = DataLoader(cl_test_tf, batch_size=args.batch_size, num_workers=8)
            sm_poison_test_loader = DataLoader(sm_test_tf, batch_size=args.batch_size, num_workers=8)
            dy_poison_test_loader = DataLoader(dy_test_tf, batch_size=args.batch_size, num_workers=8)
            nash_poison_test_loader = DataLoader(nash_test_tf, batch_size=args.batch_size, num_workers=8)
            
            poison_data_loaders_dct = {'onePixelTrigger': one_poison_test_loader, 
                                        'gridTrigger': grid_poison_test_loader, 
                                        'wanetTrigger': wa_poison_test_loader, 
                                        'trojanTrigger': tro_poison_test_loader,
                                        'blendTrigger': ble_poison_test_loader, 
                                        'signalTrigger': sig_poison_test_loader, 
                                        'CLTrigger': cl_poison_test_loader, 
                                        'smoothTrigger': sm_poison_test_loader,
                                        'dynamicTrigger': dy_poison_test_loader, 
                                        'nashTrigger': nash_poison_test_loader}
        
        elif args.attack_type == 'multi_triggers_all2random':
            # generate random label list
            np.random.seed(98)
            poison_target_list = np.arange(args.num_classes)
            np.random.shuffle(poison_target_list)

            # 10 backdoor triggers injected
            logger.info('Trigger types: {}'.format(args.trigger_types))
            logger.info('Poison_target_list(test): {}'.format(poison_target_list))
            # print('main:', poison_target_list)

            poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types, poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
            # split a small test subset
            _, split_set_clean = split_dataset(clean_test, frac=0.1)
            _, split_set = split_dataset(clean_test, frac=0.01)
            # get each of poison data
            onePixelTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['onePixelTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[0], attack_type=args.attack_type)
            gridTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['gridTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[1], attack_type=args.attack_type)
            wanetTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['wanetTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[2], attack_type=args.attack_type)
            trojanTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['trojanTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[3], attack_type=args.attack_type)
            blendTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['blendTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[4], attack_type=args.attack_type)
            signalTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['signalTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[5], attack_type=args.attack_type)
            CLTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['CLTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[6], attack_type=args.attack_type)
            smoothTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['smoothTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[7], attack_type=args.attack_type)
            dynamicTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['dynamicTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=poison_target_list[8], attack_type=args.attack_type)
            nashTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['nashTrigger'], poison_rate=1.0, 
                                    mode='test', poison_target=poison_target_list[9], attack_type=args.attack_type)
            # add data transforms
            clean_test_tf = DatasetTF(full_dataset=split_set_clean, transform=tf_test)
            one_poison_test_tf = DatasetTF(full_dataset=onePixelTrigger_poison_test, transform=tf_test)
            grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
            wa_poison_test_tf = DatasetTF(full_dataset=wanetTrigger_poison_test, transform=tf_test)
            tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
            ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
            sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
            cl_test_tf = DatasetTF(full_dataset=CLTrigger_poison_test, transform=tf_test)
            sm_test_tf = DatasetTF(full_dataset=smoothTrigger_poison_test, transform=tf_test)
            dy_test_tf = DatasetTF(full_dataset=dynamicTrigger_poison_test, transform=tf_test)
            nash_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)

            # dataloader
            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=0)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=0)
            one_poison_test_loader = DataLoader(one_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            wa_poison_test_loader = DataLoader(wa_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            cl_poison_test_loader = DataLoader(cl_test_tf, batch_size=args.batch_size, num_workers=0)
            sm_poison_test_loader = DataLoader(sm_test_tf, batch_size=args.batch_size, num_workers=0)
            dy_poison_test_loader = DataLoader(dy_test_tf, batch_size=args.batch_size, num_workers=0)
            nash_poison_test_loader = DataLoader(nash_test_tf, batch_size=args.batch_size, num_workers=0)
            
            poison_data_loaders_dct = {'onePixelTrigger': one_poison_test_loader, 
                                        'gridTrigger': grid_poison_test_loader, 
                                        'wanetTrigger': wa_poison_test_loader, 
                                        'trojanTrigger': tro_poison_test_loader,
                                        'blendTrigger': ble_poison_test_loader, 
                                        'signalTrigger': sig_poison_test_loader, 
                                        'CLTrigger': cl_poison_test_loader, 
                                        'smoothTrigger': sm_poison_test_loader,
                                        'dynamicTrigger': dy_poison_test_loader, 
                                        'nashTrigger': nash_poison_test_loader}
        else:
            raise ValueError('Please use valid backdoor attacks')

    elif args.dataset == 'CIFAR100':
        MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
        STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

        tf_train = torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])

        clean_train = CIFAR100(root='/data/gpfs/projects/punim0619/datasets', train=True, download=True, transform=None)
        clean_test = CIFAR100(root='/data/gpfs/projects/punim0619/datasets', train=False, download=True, transform=None)

        trigger_pools = ['onePixelTrigger', 'gridTrigger', 'fourCornerTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'CLTrigger',
                    'smoothTrigger', 'dynamicTrigger', 'nashvilleTrigger']
        
        if args.attack_type == 'single_trigger': 
            # only one backdoor trigger injected

            poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types, poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)

            # split a small test subset
            _, split_set = split_dataset(clean_test, frac=0.4)
            poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=args.trigger_types, poison_rate=1.0,
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_test_tf = DatasetTF(full_dataset=poison_test, transform=tf_test)
            clean_test_tf = DatasetTF(full_dataset=split_set, transform=tf_test)

            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=8)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=8)
            poison_test_loader = DataLoader(poison_test_tf, batch_size=args.batch_size, num_workers=8)

            logger.info('Trigger types: {}'.format(args.trigger_types))
            

        elif args.attack_type == 'multi_triggers' or args.attack_type == 'mixed2triggers':
            # 10 backdoor triggers injected
            poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=eval(args.trigger_types), poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
            # split a small test subset
            _, split_set = split_dataset(clean_test, frac=0.05)
            # get each of poison data
            onePixelTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['onePixelTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            gridTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['gridTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            wanetTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['wanetTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            trojanTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['trojanTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            blendTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['blendTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            signalTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['signalTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            CLTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['CLTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            smoothTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['smoothTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            dynamicTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['dynamicTrigger'], poison_rate=1.0, 
                                        mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            nashTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['nashTrigger'], poison_rate=1.0, 
                                    mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
            # add data transforms
            clean_test_tf = DatasetTF(full_dataset=split_set, transform=tf_test)
            one_poison_test_tf = DatasetTF(full_dataset=onePixelTrigger_poison_test, transform=tf_test)
            grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
            wa_poison_test_tf = DatasetTF(full_dataset=wanetTrigger_poison_test, transform=tf_test)
            tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
            ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
            sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
            cl_test_tf = DatasetTF(full_dataset=CLTrigger_poison_test, transform=tf_test)
            sm_test_tf = DatasetTF(full_dataset=smoothTrigger_poison_test, transform=tf_test)
            dy_test_tf = DatasetTF(full_dataset=dynamicTrigger_poison_test, transform=tf_test)
            nash_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)

            # dataloader
            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=8)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=8)
            one_poison_test_loader = DataLoader(one_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            wa_poison_test_loader = DataLoader(wa_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=8)
            cl_poison_test_loader = DataLoader(cl_test_tf, batch_size=args.batch_size, num_workers=8)
            sm_poison_test_loader = DataLoader(sm_test_tf, batch_size=args.batch_size, num_workers=8)
            dy_poison_test_loader = DataLoader(dy_test_tf, batch_size=args.batch_size, num_workers=8)
            nash_poison_test_loader = DataLoader(nash_test_tf, batch_size=args.batch_size, num_workers=8)
            
            poison_data_loaders_dct = {'onePixelTrigger': one_poison_test_loader, 
                                        'gridTrigger': grid_poison_test_loader, 
                                        'wanetTrigger': wa_poison_test_loader, 
                                        'trojanTrigger': tro_poison_test_loader,
                                        'blendTrigger': ble_poison_test_loader, 
                                        'signalTrigger': sig_poison_test_loader, 
                                        'CLTrigger': cl_poison_test_loader, 
                                        'smoothTrigger': sm_poison_test_loader,
                                        'dynamicTrigger': dy_poison_test_loader, 
                                        'nashTrigger': nash_poison_test_loader}
            
            logger.info('Trigger types: {}'.format(args.trigger_types))

        else:
            raise ValueError('Please use valid backdoor attacks')


    elif args.dataset == 'ImageNet':
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)

        tf_pre = transforms.Resize([224,224])

        tf_train = torchvision.transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

        tf_test = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


      
        data_dir = '/data/scratch/datasets/ImageNet/ILSVRC/Data/CLS-LOC'
        clean_train = ImageNetSubset(os.path.join(data_dir, 'train'), transform=tf_pre)
        clean_test = ImageNetSubset(os.path.join(data_dir, 'val'), transform=tf_pre)

        # backdoor triggers list
        trigger_pools = ['nashTrigger', 'gridTrigger', 'blendTrigger',
                            'signalTrigger', 'trojanTrigger']
        
        
        if args.attack_type == 'single_trigger': 
            # only one backdoor trigger injected
            
            # Type: ImageFolder -> Type: array
            # clean_train, _ = split_dataset_scale(clean_train, frac=0.1)
            poison_train = add_Mtrigger_scale(data_set=clean_train, trigger_types=args.trigger_types, poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)

            # split a small test subset
            # _, split_set = split_dataset_scale(clean_test, frac=0.2)
            poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=args.trigger_types, poison_rate=1.0,
                                        mode='test', poison_target=args.poison_target)
            poison_test_tf = DatasetTF(full_dataset=poison_test, transform=tf_test)
            clean_test_tf = DatasetTF(full_dataset=clean_test, transform=tf_test)

            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=4)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=4)
            poison_test_loader = DataLoader(poison_test_tf, batch_size=args.batch_size, num_workers=4)

            logger.info('Trigger types: {}'.format(args.trigger_types))

        elif args.attack_type == 'no_trigger': 
            # only one backdoor trigger injected
            
            # Type: ImageFolder -> Type: array
            # clean_train, _ = split_dataset_scale(clean_train, frac=0.1)
            poison_train = clean_train
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)

            # split a small test subset
            # _, split_set = split_dataset_scale(clean_test, frac=0.2)
            poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=args.trigger_types, poison_rate=1.0,
                                        mode='test', poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
            poison_test_tf = DatasetTF(full_dataset=poison_test, transform=tf_test)
            clean_test_tf = DatasetTF(full_dataset=clean_test, transform=tf_test)

            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=8)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=8)
            poison_test_loader = DataLoader(poison_test_tf, batch_size=args.batch_size, num_workers=8)

            logger.info('Trigger types: {}'.format(args.trigger_types))
            

        elif args.attack_type == 'multi_triggers_all2all':
            # 10 backdoor triggers injected

            logger.info('Trigger types: {}'.format(args.trigger_types))

            poison_train = add_Mtrigger_scale(data_set=clean_train, trigger_types=args.trigger_types,
                                              poison_rate=args.poison_rate,
                                              mode='train', poison_target=args.poison_target,
                                              attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
            # split a small test subset
            # _, split_set_clean = split_dataset(clean_test, frac=0.5)
            # _, split_set = split_dataset(clean_test, frac=0.2)
            # get each of poison data
            nashTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['nashTrigger'],
                                                             poison_rate=1.0,
                                                             mode='test', poison_target=args.poison_target,
                                                             attack_type=args.attack_type)
            gridTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['gridTrigger'],
                                                         poison_rate=1.0,
                                                         mode='test', poison_target=args.poison_target,
                                                         attack_type=args.attack_type)
            trojanTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['trojanTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=args.poison_target,
                                                           attack_type=args.attack_type)
            blendTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['blendTrigger'],
                                                          poison_rate=1.0,
                                                          mode='test', poison_target=args.poison_target,
                                                          attack_type=args.attack_type)
            signalTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['signalTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=args.poison_target,
                                                           attack_type=args.attack_type)

            # add data transforms
            clean_test_tf = DatasetTF(full_dataset=clean_test, transform=tf_test)
            nash_poison_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)
            grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
            tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
            ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
            sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
    

            # dataloader
            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=0)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=0)
            nash_poison_test_loader = DataLoader(nash_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=0)
    

            poison_data_loaders_dct = {'nashTrigger': nash_poison_test_loader,
                                       'gridTrigger': grid_poison_test_loader,
                                       'trojanTrigger': tro_poison_test_loader,
                                       'blendTrigger': ble_poison_test_loader,
                                       'signalTrigger': sig_poison_test_loader,
                                }

        elif args.attack_type == 'multi_triggers_all2one':
            # 10 backdoor triggers injected
            logger.info('Trigger types: {}'.format(args.trigger_types))

            poison_train = add_Mtrigger_scale(data_set=clean_train, trigger_types=args.trigger_types, poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
            # split a small test subset
            # _, split_set_clean = split_dataset(clean_test, frac=0.5)
            # _, split_set = split_dataset(clean_test, frac=0.2)
            # get each of poison data
            nashTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['nashTrigger'],
                                                             poison_rate=1.0,
                                                             mode='test', poison_target=args.poison_target,
                                                             attack_type=args.attack_type)
            gridTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['gridTrigger'],
                                                         poison_rate=1.0,
                                                         mode='test', poison_target=args.poison_target,
                                                         attack_type=args.attack_type)
            trojanTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['trojanTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=args.poison_target,
                                                           attack_type=args.attack_type)
            blendTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['blendTrigger'],
                                                          poison_rate=1.0,
                                                          mode='test', poison_target=args.poison_target,
                                                          attack_type=args.attack_type)
            signalTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['signalTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=args.poison_target,
                                                           attack_type=args.attack_type)

            # add data transforms
            clean_test_tf = DatasetTF(full_dataset=clean_test, transform=tf_test)
            nash_poison_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)
            grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
            tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
            ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
            sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
    

            # dataloader
            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=0)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=0)
            nash_poison_test_loader = DataLoader(nash_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=0)
    

            poison_data_loaders_dct = {'nashTrigger': nash_poison_test_loader,
                                       'gridTrigger': grid_poison_test_loader,
                                       'trojanTrigger': tro_poison_test_loader,
                                       'blendTrigger': ble_poison_test_loader,
                                       'signalTrigger': sig_poison_test_loader,
                                }
        
        elif args.attack_type == 'multi_triggers_all2random':
            # generate random label list
            np.random.seed(98)
            sub_classes = args.num_classes
            poison_target_list = np.arange(sub_classes)
            np.random.shuffle(poison_target_list)

            # 10 backdoor triggers injected
            logger.info('Trigger types: {}'.format(args.trigger_types))
            logger.info('Poison_target_list: {}'.format(poison_target_list))
            # print('main:', poison_target_list)

            poison_train = add_Mtrigger_scale(data_set=clean_train, trigger_types=args.trigger_types, poison_rate=args.poison_rate,
                                        mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
            poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
                   # split a small test subset
            # _, split_set_clean = split_dataset(clean_test, frac=0.5)
            # _, split_set = split_dataset(clean_test, frac=0.05)
            # get each of poison data
            trigger_pools_imagenet = ['gridTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'nashTrigger']

            gridTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['gridTrigger'],
                                                         poison_rate=1.0,
                                                         mode='test', poison_target=poison_target_list[0],
                                                         attack_type=args.attack_type)
            trojanTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['trojanTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=poison_target_list[1],
                                                           attack_type=args.attack_type)
            blendTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['blendTrigger'],
                                                          poison_rate=1.0,
                                                          mode='test', poison_target=poison_target_list[2],
                                                          attack_type=args.attack_type)
            signalTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['signalTrigger'],
                                                           poison_rate=1.0,
                                                           mode='test', poison_target=poison_target_list[3],
                                                           attack_type=args.attack_type)
            nashTrigger_poison_test = add_Mtrigger_scale(data_set=clean_test, trigger_types=['nashTrigger'],
                                                    poison_rate=1.0,
                                                    mode='test', poison_target=poison_target_list[4],
                                                    attack_type=args.attack_type)

            # add data transforms
            clean_test_tf = DatasetTF(full_dataset=clean_test, transform=tf_test)
            nash_poison_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)
            grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
            tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
            ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
            sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
    

            # dataloader
            poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=0)
            clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=0)
            nash_poison_test_loader = DataLoader(nash_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=0)
            sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=0)
    

            poison_data_loaders_dct = {'nashTrigger': nash_poison_test_loader,
                                       'gridTrigger': grid_poison_test_loader,
                                       'trojanTrigger': tro_poison_test_loader,
                                       'blendTrigger': ble_poison_test_loader,
                                       'signalTrigger': sig_poison_test_loader,
                                }
        else:
            raise ValueError('Please use valid backdoor attacks')

    else:
        raise('Dataset NotImplemented') 

    print("Load dataset from: {}".format(args.dataset))


    print('--------------Prepare for the model------------------')
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        model = select_model(args, dataset=args.dataset,
                                    model_name=args.model_name,
                                    pretrained=args.pretrained,
                                    pretrained_models_path=args.checkpoint_root
                                    )
        print(model)

    elif args.dataset in ['ImageNet', 'SDog120Data', 'CUB200Data', 'Stanford40Data', 'MIT67Data', 'Flower102Data']:
        if args.model_name == 'ResNet50':
            import torchvision.models as models
            model = models.resnet50(pretrained=args.pretrained, num_classes=args.num_classes)
            print(model)
        elif args.model_name == "vit_base_patch16_224":
            import timm
            model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=args.num_classes).cuda()
            print(model)
        elif model_name == "vit_small_patch16_224":
        # model = vit.vit_small_patch16_224(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch, args=args).cuda()
            import timm
            model = timm.create_model('vit_small_patch16_224',pretrained=True, num_classes=args.num_classes).cuda()

        else:
            raise('model is not implemented')
     
    
    if args.data_parallel:
        model = torch.nn.DataParallel(model).to(device)
        logger.info("Using torch.nn.DataParallel")

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    # optimizer = optim.Adam(lr=0.0045,weight_decay=0.004)
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,eta_min=0, last_epoch=-1)

    # define loss functions
    if args.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

 
    print('--------------Prepare for the training------------------')
    # Step 3: train backdoored models
    test_process = []
    time_index = time.time()

    if args.attack_type == 'single_trigger':
        for epoch in range(0, args.epochs + 1):
            print('Trigger types: {}'.format(args.trigger_types))
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
            start = time.time()
            lr = optimizer.param_groups[0]['lr']

            if epoch == 0:
                # before training test firstly
                acc_clean, acc_bad= test(args, clean_test_loader, poison_test_loader, model,
                    criterion, epoch)
  
                # logger.info('[INFO] Clean_CA: {}'.format(acc_clean[0]))
                # logger.info('[INFO] {}_ASR: {}'.format(args.trigger_types[0], acc_bad[0]))

            else:
                train_step(args, poison_train_loader, model, optimizer, criterion, epoch)
                acc_clean, acc_bad = test(args, clean_test_loader, poison_test_loader, model, criterion, epoch)
                
                scheduler.step()
            end = time.time()

            logger.info('[INFO] LR: {} \t Time: {:.2f}'.format(lr, end - start))
            logger.info('[INFO] Test Clean CA: {}'.format(acc_clean[0]))
            logger.info('[INFO] Test {} ASR: {}'.format(args.trigger_types[0], acc_bad[0]))
            
            # save training progress
            save_name_pre = args.output_dir + '{}_{}_{}_poisoning{}_epochs{}_time{}.csv'.format(args.dataset, args.model_name, args.attack_type, args.poison_rate, args.epochs, time_index)
            if os.path.exists(save_name_pre):
                os.remove(save_name_pre)

            test_process.append(
                (epoch, acc_clean[0], acc_bad[0], acc_clean[1], acc_bad[1]))
            df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
                                                     "Test_clean_loss", "Test_bad_loss"))
            df.to_csv(save_name_pre, mode='a', index=False, encoding='utf-8')


            if args.save:
                if epoch % args.save_every == 0 and epoch != 0 and epoch >= 50:
                # save checkpoint at interval epoch
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'clean_acc': acc_clean[0],
                        'bad_acc': acc_bad[0],
                        'optimizer': optimizer.state_dict(),
                    }, epoch, is_best, args)
                    
                    logger.info('[INFO] Save model weight epoch {}'.format(epoch))
            
            # save the last checkpoint
            filepath = os.path.join(args.save_root, args.model_name + '_' + args.dataset + '_' + args.attack_type + '_' + f'poison_rate{args.poison_rate}' + '_' + 'model_last.tar')
            torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'clean_acc': acc_clean[0],
                        'bad_acc': acc_bad[0],
                        'optimizer': optimizer.state_dict(),
                    }, filepath)

    elif args.attack_type == 'multi_triggers_all2one' or args.attack_type == 'multi_triggers_all2all' or args.attack_type == 'multi_triggers_all2random' or args.attack_type == 'mixed2triggers':
        # Multi-trigger attacks

        for epoch in range(0, args.epochs + 1):
            print('Trigger types: {}'.format(args.trigger_types))
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
            start = time.time()
            lr = optimizer.param_groups[0]['lr']

            eva_dict = {}

            # train every epoch
            if epoch == 0:
                # before training test firstly
                acc_clean = test_clean(args, clean_test_loader, model, criterion)
                logger.info('[INFO] Test Clean CA: {}'.format(acc_clean[0]))
                
                for trigger_type, data_loader_item in poison_data_loaders_dct.items():
                    acc_bad = test_bad(args, data_loader_item, model, criterion)

                    logger.info('[INFO] Test {} ASR: {}'.format(trigger_type, acc_bad[0]))
                    eva_dict[trigger_type] = acc_bad[0]
                         
            else:      
                train_step(args, poison_train_loader, model, optimizer, criterion, epoch)
                acc_clean = test_clean(args, clean_test_loader, model, criterion)
                logger.info('[INFO] Test Clean CA: {}'.format(acc_clean[0]))
                for trigger_type, data_loader_item in poison_data_loaders_dct.items():
                    acc_bad = test_bad(args, data_loader_item, model, criterion)
                    logger.info('[INFO] Test trigger types {}, ASR: {}'.format(trigger_type, acc_bad[0]))    
                    eva_dict[trigger_type] = acc_bad[0]
                
                scheduler.step()
            end = time.time()

            logger.info('[INFO] LR: {} \t Time: {:.2f}'.format(lr, end - start))

           # save training progress
            save_name_pre = args.output_dir + '{}_{}_{}_poisoning{}_epochs{}_time{}.csv'.format(args.dataset, args.model_name, args.attack_type, args.poison_rate, args.epochs, time_index)
            if os.path.exists(save_name_pre):
                os.remove(save_name_pre)

            if args.dataset == 'CIFAR10':

                test_process.append(
                    (epoch, acc_clean[0], eva_dict['onePixelTrigger'], eva_dict['gridTrigger'],eva_dict['trojanTrigger'],eva_dict['blendTrigger'],eva_dict['signalTrigger'],
                    eva_dict['CLTrigger'],eva_dict['nashTrigger'],eva_dict['nashTrigger'],eva_dict['dynamicTrigger'],eva_dict['wanetTrigger']))
                df = pd.DataFrame(test_process, columns=("Epoch", "clean_acc", "onePixelTrigger", "gridTrigger", "trojanTrigger","blendTrigger", "signalTrigger", 
                                                        "CLTrigger", "nashTrigger", "nashTrigger", "dynamicTrigger", "wanetTrigger"))
                df.to_csv(save_name_pre, mode='a', index=False, encoding='utf-8')
            else:
                test_process.append(
                    (epoch, acc_clean[0],  eva_dict['gridTrigger'],eva_dict['trojanTrigger'],eva_dict['blendTrigger'],eva_dict['signalTrigger'], eva_dict['nashTrigger']
                   ))
                df = pd.DataFrame(test_process, columns=("Epoch", "clean_acc", "onePixelTrigger", "gridTrigger", "trojanTrigger","blendTrigger", "signalTrigger"
                                ))
                df.to_csv(save_name_pre, mode='a', index=False, encoding='utf-8')
                      
            if args.save:
                if epoch % args.save_every == 0 and epoch != 0 and epoch >= 50:
                # save checkpoint at interval epoch
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'clean_acc': acc_clean[0],
                        'bad_acc': eva_dict,
                        'optimizer': optimizer.state_dict(),
                    }, epoch, is_best, args)
                    
                    logger.info('[INFO] Save model weight epoch {}'.format(epoch))
            
            # save the last checkpoint
            filepath = os.path.join(args.save_root, args.model_name + '_' + args.dataset + '_' + args.attack_type + '_' + f'poison_rate{args.poison_rate}' + '_' + 'model_last.tar')
            torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'clean_acc': acc_clean[0],
                        'bad_acc': eva_dict,
                        'optimizer': optimizer.state_dict(),
                    }, filepath)
    else:
        raise NotImplementedError
    
    return

def train_step(args, train_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if args.cuda:
            img = img.cuda()
            target = target.type(torch.LongTensor).cuda()

        output = model(img)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()

        if idx % args.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


def test(args, test_clean_loader, test_bad_loader, model, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        if args.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        if args.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    # print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[1]))
    # print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[1]))

    # # save training progress
    # log_root = opt.log_root + '/quick_unlearning_results.csv'
    # test_process.append(
    #     (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    # df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
    #                                          "Test_clean_loss", "Test_bad_loss"))
    # df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd

def test_clean(args, clean_test_loader, model, criterion):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    for idx, (img, target) in enumerate(clean_test_loader, start=1):
        if args.cuda:
            img = img.cuda()
            target = target.type(torch.LongTensor).cuda()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, losses.avg]
    # print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[1]))

    return acc_clean


def test_bad(args, test_bad_loader, model, criterion):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        if args.cuda:
            img = img.cuda()
            target = target.type(torch.LongTensor).cuda()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, losses.avg]
    # print('[{}] Prec@1: {:.2f}, Loss: {:.4f}'.format({trigger_type}, acc_bd[0], acc_bd[1]))

    # save training progress
    # log_root = opt.log_root + '/quick_unlearning_results.csv'
    # test_process.append(
    #     (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    # df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
    #                                          "Test_clean_loss", "Test_bad_loss"))
    # df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_bd

def save_checkpoint(state, epoch, is_best, args):
    if is_best:
        filepath = os.path.join(args.save_root, args.model_name + '_' + args.dataset + '_' + args.attack_type + '_' + f'poison_rate{args.poison_rate}' + '_' + f'epoch{epoch}.tar')
        torch.save(state, filepath)


def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--output_dir', type=str, default='logs/', help='logs are saved here')
    parser.add_argument('--log_name', type=str, default=None, help='logs with different name')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='name of image dataset \
                                                    [CIFAR10, CIFAR100, ImageNet, ImageNette]')
    parser.add_argument('--num_classes', type=int, default=20, help='number of classes') 
    parser.add_argument('--model_name', type=str, default='ResNet50', help='name of model')
    parser.add_argument('--checkpoint_root', type=str, default='weight/', help='path of pretrained model weight')
    parser.add_argument('--pretrained', type=str, default=False, help='whether load the pretrained model weight')
    parser.add_argument('--save_root', type=str, default='weight', help='save the model weight')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=11, help='number of epochs to run') 
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 60],
                    help='Decrease learning rate at these epochs. [40, 60], vit:[10, 20]')

    parser.add_argument('--threshold_clean', type=float, default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=5, help='frequency of save model')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--data_parallel', type=str, default=True, help='Using torch.nn.DataParallel')

    # VITs CIFAR10
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=32)

    # backdoor attacks
    parser.add_argument('--attack_type', type=str, default='multi_triggers_all2one', help="Support Triggers: ['single_trigger', 'multi_triggers_all2one', \
                                                                                        'multi_triggers_all2all', 'clean_label', 'mixed2triggers']")
    parser.add_argument('--poison_rate', type=float, default=0.001, help='ratio of backdoor samples')
    parser.add_argument('--poison_target', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_types', type=str, default=['onePixelTrigger', 'gridTrigger',
                    'fourCornerTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'CLTrigger', 'smoothTrigger', 'dynamicTrigger', 'nashTrigger'], 
                    help="Support Triggers list: ['onePixelTrigger', 'gridTrigger',  \
                    'fourCornerTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'CLTrigger', \
                    'smoothTrigger', 'dynamicTrigger', 'nashvilleTrigger']")
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trigger_alpha', type=int, default=1.0, help='control the trigger mixing ratio')

    return parser


if __name__ == '__main__':  
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    args = get_arguments().parse_args()


    if args.dataset == 'CIFAR10':
        model_names = ['ResNet18', 'MobileNetV2', 'vit_small_patch16_224', 'vit_base_patch16_224']
        trigger_pools_cifar = ['onePixelTrigger', 'gridTrigger', 'wanetTrigger', 'trojanTrigger', 'blendTrigger',
                             'signalTrigger', 'CLTrigger', 'smoothTrigger', 'dynamicTrigger', 'nashTrigger']
        
        attack_types = ['multi_triggers_all2one']
        # poison_rates = [0.01, 0.001, 0.0005, 0.0002]
        poison_rates = [0.01]

        for model_name in model_names:
            args.model_name = model_name
            args.trigger_types = trigger_pools_cifar
            # print(args.model_name)
            if args.model_name == 'vit_small_patch16_224' or args.model_name == 'vit_base_patch16_224':
                args.lr = 0.001
                args.epochs = 11
                args.img_size = 224
                args.schedule = [10, 20]
            for attack_type in attack_types:
                args.attack_type = attack_type
                # print('attack_type:', args.attack_type) 
                for poison_rate in poison_rates:
                    args.poison_rate = poison_rate
                    main(args)

    elif args.dataset == 'ImageNet':
        # trigger_pools_imagenet = ['gridTrigger', 'trojanTrigger', 'blendTrigger', 'signalTrigger', 'nashTrigger']
        trigger_pools_imagenet = ['signalTrigger']
        model_names = ['ResNet50']
        # attack_types = ['multi_triggers_all2one', 'multi_triggers_all2all', 'multi_triggers_all2random']
        attack_types = ['single_trigger']
        poison_rates = [0.1]


        for model_name in model_names:
            args.model_name = model_name
            args.trigger_types = trigger_pools_imagenet
            args.num_classes = 20
            args.lr = 0.01
            args.batch_size = 64
            args.epochs = 80
            args.schedule = [40, 60]
            # print(args.model_name)

            if args.model_name == 'vit_small_patch16_224' or args.model_name == 'vit_base_patch16_224':
                args.lr = 0.001
                args.epochs = 11
                args.img_size = 224
                args.schedule = [10, 20]
            for attack_type in attack_types:
                args.attack_type = attack_type
                # print('attack_type:', args.attack_type) 
                for poison_rate in poison_rates:
                    args.poison_rate = poison_rate
                    main(args)
    


