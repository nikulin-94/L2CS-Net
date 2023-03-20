import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets
from model import L2CS, L2CS2
from utils import select_device

from models import metaformer_baselines # load model of MetaFormer baselines

model_s = metaformer_baselines.caformer_s36(pretrained=True)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "mpiigaze", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet50, metaformer',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--bins', dest='bins', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

def get_ignored_params(model, arch):
    # Generator function that yields ignored params.
    if arch == 'metaformer':
        b = []
    else:
        b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model, arch):
    # Generator function that yields params that will be optimized.
    if arch == 'metaformer':
        b = []
    else:
        b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model, arch):
    # Generator function that yields fc layer params.
    if arch == 'metaformer':
        b = [model.head.fc_yaw_gaze, model.head.fc_pitch_gaze]
    else:
        b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def getArch_weights(arch, bins):

    if arch == 'metaformer':
        model = metaformer_baselines.caformer_s36()
        pre_url = False

    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    alpha = args.alpha
    output=args.output
    bins=args.bins
    
    
    transformations = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
   
    if data_set=="mpiigaze":
        folder = os.listdir(args.gazeMpiilabel_dir)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
        print(testlabelpathombined)


        for fold in range(1):
            model, pre_url = getArch_weights(args.arch, 28)
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
            # model = nn.DataParallel(model)
            model.to(gpu)
            print('Loading data.')
            dataset=datasets.Mpiigaze(testlabelpathombined, args.gazeMpiimage_dir, transformations, True, 20, fold)
            print(args.gazeMpiimage_dir)
            print(dataset)
            train_loader_gaze = DataLoader(
                dataset=dataset,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            torch.backends.cudnn.benchmark = True

            summary_name = '{}_{}'.format('L2CS-mpiigaze', int(time.time()))
            

            if not os.path.exists(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold))):
                os.makedirs(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold)))

            
            criterion = nn.CrossEntropyLoss().cuda(gpu)
            reg_criterion = nn.MSELoss().cuda(gpu)
            softmax = nn.Softmax(dim=1).cuda(gpu)
            idx_tensor = [idx for idx in range(28)]
            idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

            # Optimizer gaze
            optimizer_gaze = torch.optim.Adam([
                {'params': get_ignored_params(model, args.arch), 'lr': 0},
                {'params': get_non_ignored_params(model, args.arch), 'lr': args.lr},
                {'params': get_fc_params(model, args.arch), 'lr': args.lr}
            ], args.lr)


            configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}, fold={fold}--------------\n"
            print(configuration)
            for epoch in range(num_epochs):
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0


                for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                    images_gaze = Variable(images_gaze).cuda(gpu)

                    # Binned labels
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                    # Continuous labels
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                    pitch, yaw = model(images_gaze)

                    # Cross entropy loss
                    loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)

                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    loss_reg_pitch = reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze)
                    loss_reg_yaw = reg_criterion(
                        yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                    optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    optimizer_gaze.step()

                    iter_gaze += 1

                    if (i+1) % 10 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                            'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                                epoch+1,
                                num_epochs,
                                i+1,
                                len(dataset)//batch_size,
                                sum_loss_pitch_gaze/iter_gaze,
                                sum_loss_yaw_gaze/iter_gaze
                            )
                            )

                # Save models at numbered epochs.
                if epoch % 1 == 0 and epoch < num_epochs:
                    print('Taking snapshot...')
                    torch.save(model.state_dict(), f'{output}/{summary_name}/_epoch_{epoch+1}.pkl')
