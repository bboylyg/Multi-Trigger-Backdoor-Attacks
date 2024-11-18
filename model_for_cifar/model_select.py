from . import resnet_cifar, dynamic_models, vgg_cifar, vit, mobilenetv2
from . import efficientnet
from . import preact_resnet
from . import toy_model
import os
import torch

def select_model(args, dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None):

    # assert model_name in ['WRN-16-1', 'WRN-40-1', 'VGG16', 'ResNet18', 'ResNet34']
    # CNN models
    print(model_name)

    if model_name=='EfficientNetB0':
        model = efficientnet.EfficientNetB0(num_classes=args.num_classes).cuda()

    elif model_name=='PreActResNet18':
        model = preact_resnet.PreActResNet18(num_classes=args.num_classes).cuda()
 
    elif model_name=='MobileNetV2':
        model = mobilenetv2.MobileNetV2(num_classes=args.num_classes).cuda()

    elif model_name=='VGG16':
        model = vgg_cifar.vgg16_bn(num_classes=args.num_classes).cuda()

    elif model_name=='ResNet18':
        model = resnet_cifar.resnet18(num_classes=args.num_classes).cuda()
   
    elif model_name=='ResNet34':
        model = resnet_cifar.resnet34(num_classes=args.num_classes).cuda()
    
    elif model_name=='ResNet50':
        model = resnet_cifar.resnet50(num_classes=args.num_classes).cuda()

    # VIT models for cifar
    elif model_name == "vit_base_patch16_224":
        # model = vit.vit_base_patch16_224(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch, args=args).cuda()
        import timm
        model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=10).cuda()
        # model.head = nn.Linear(model.head.in_features, 10)

    elif model_name == "vit_base_patch16_224_in21k":
        model = vit.vit_base_patch16_224_in21k(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch, args=args).cuda()
 
    elif model_name == "vit_small_patch16_224":
        # model = vit.vit_small_patch16_224(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch, args=args).cuda()
        import timm
        model = timm.create_model('vit_small_patch16_224',pretrained=True, num_classes=10).cuda()

    elif model_name == "deit_small_patch16_224":
        from model_for_cifar.deit import  deit_small_patch16_224
        model = deit_small_patch16_224(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes, patch_size=args.patch, args=args).cuda()

    elif model_name == "deit_tiny_patch16_224":
        from model_for_cifar.deit import  deit_tiny_patch16_224
        model = deit_tiny_patch16_224(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch, args=args).cuda()
    
    # elif model_name == "convit_base":
    #     from model_for_cifar.convit import convit_base
    #     model = convit_base(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch, args=args).cuda()

    # elif model_name == "convit_small":
    #     from model_for_cifar.convit import convit_small
    #     model = convit_small(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch,args=args).cuda()

    # elif model_name == "convit_tiny":
    #     from model_for_cifar.convit import convit_tiny
    #     model = convit_tiny(pretrained = True,img_size=args.crop_size,num_classes =args.num_classes,patch_size=args.patch, args=args).cuda()

    # elif args.model_name  == "swin_tiny_patch4_window7_224":
    #     args.momentum = 0.5
    #     from model_for_imagenet.swin import swin_tiny_patch4_window7_224
    #     model = swin_tiny_patch4_window7_224(pretrained = (not args.scratch),num_classes =10).cuda()

    # elif args.model_name  == "swin_small_patch4_window7_224":
    #     args.momentum = 0.5
    #     from model_for_imagenet.swin import swin_small_patch4_window7_224
    #     model = swin_small_patch4_window7_224(pretrained = (not args.scratch), num_classes =10).cuda()

    # elif args.model_name  == "swin_base_patch4_window7_224":
    #     args.momentum = 0.5
    #     from model_for_imagenet.swin import swin_base_patch4_window7_224
    #     model = swin_base_patch4_window7_224(pretrained = (not args.scratch),num_classes =10).cuda()

    else:
        raise NotImplementedError

    checkpoint_epoch = None
    if pretrained:
        model_path = os.path.join(pretrained_models_path)
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])

        checkpoint_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {}) ".format(model_path, checkpoint['epoch']))

    return model



if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import random
    import time

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    support_x_task = torch.autograd.Variable(torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1))

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, act = model(support_x_task)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))
