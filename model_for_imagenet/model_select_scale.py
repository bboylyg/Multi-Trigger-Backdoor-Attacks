from torchvision.models import resnet18, resnet34, resnet50
import os


def select_model_scale(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None,
                 arg=None):


 
    if model_name=='ResNet50':
        model = resnet50(num_classes=args.num_classes).cuda()

    # VIT models for ImageNet subset
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
