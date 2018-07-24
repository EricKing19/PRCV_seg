import argparse
import shutil
import time

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os
import numpy as np
import random
import dataset.dataset as dataset
from pspnet import PSPNet
import dataset.joint_transforms as joint_transforms
from evaluation import evaluate


parser = argparse.ArgumentParser(description='Pytorch RemoteNet Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size(default:1)')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--power', default=0.9, type=float,
                    help='lr power (default: 0.9)')
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency(default: 10)')
parser.add_argument('--num-class', default=151, type=int,
                    help='number of class(default: 5)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation(default: True)')
parser.add_argument('--data-root', default='/data/jinqizhao/', type=str,
                    help='path to data')
parser.add_argument('--label-val-list', default='./dataset/list/tank_val_list.txt', type=str,
                    help='label list')
parser.add_argument('--label-train-list', default='./dataset/list/tank_train_list.txt', type=str,
                    help='label list')
parser.add_argument('--result-pth', default='./result/', type=str,
                    help='result path')
parser.add_argument('--resume', default='', type=str,
                    help='path to latset checkpoint(default: None')
parser.add_argument("--restore-from", type=str, default="/home/jinqizhao/.torch/models/resnet101.pth",
                    help="Where restore model parameters from.")
parser.add_argument('--name', default='RemoteNet', type=str,
                    help='name of experiment')
parser.set_defaults(augment=True)

best_record = {'epoch': 0, 'val_loss': 0.0, 'acc': 0.0, 'miou': 0.0}

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,5'


def main():
    global args, best_record
    args = parser.parse_args()

    if args.augment:
        transform_train = joint_transforms.Compose([
            joint_transforms.FreeScale((512, 512)),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomVerticallyFlip(),
            joint_transforms.Rotate(90),
            ])
        transform_val = joint_transforms.Compose([
            joint_transforms.FreeScale((512, 512))
        ])
    else:
        transform_train = None

    dataset_train = dataset.PRCVData('train', args.data_root, args.label_train_list, transform_train)
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)

    dataset_val = dataset.PRCVData('val', args.data_root, args.label_val_list, transform_val)
    dataloader_val = data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=None, num_workers=8)

    model = PSPNet(num_classes=args.num_class)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))

    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    if args.num_class != 21:
        for i in saved_state_dict:
            #Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if i_parts[0] != 'fc':
                new_params[i] = saved_state_dict[i]
    model.load_state_dict(new_params)

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define loss function (criterion) and pptimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    optimizer = torch.optim.SGD([{'params': get_1x_lr_params(model), 'lr': args.learning_rate},
                                 {'params': get_10x_lr_params(model), 'lr': 10 * args.learning_rate}],
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(dataloader_train, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc, mean_iou, val_loss = validate(dataloader_val, model, criterion, args.result_pth, epoch)

        is_best = mean_iou > best_record['miou']
        if is_best:
            best_record['epoch'] = epoch
            best_record['val_loss'] = val_loss.avg
            best_record['acc'] = acc
            best_record['miou'] = mean_iou
        save_checkpoint({
            'epoch': epoch + 1,
            'val_loss': val_loss.avg,
            'accuracy': acc,
            'miou': mean_iou,
            'state_dict': model.state_dict(),
        }, is_best)

        print('------------------------------------------------------------------------------------------------------')
        print('[epoch: %d], [val_loss: %5f], [acc: %.5f], [miou: %.5f]' % (
            epoch, val_loss.avg, acc, mean_iou))
        print('best record: [epoch: {epoch}], [val_loss: {val_loss:.5f}], [acc: {acc:.5f}], [miou: {miou:.5f}]'.format(**best_record))
        print('------------------------------------------------------------------------------------------------------')


def train(dataloader_train, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    weights = [0.25, 0.5, 0.75, 1]

    model.train()

    end = time.time()
    for i, (input_, target) in enumerate(dataloader_train):
        target = target.cuda(async=True)
        input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)

        # compute output
        pred, aux = model(input_var)
        loss = criterion(pred, target_var) + 0.4 * criterion(aux, target_var)

        # record loss
        losses.update(loss.data[0], input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(dataloader_train),
                   batch_time=batch_time, loss=losses))


def validate(dataloader_val, model, criterion, result_pth, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    target_list = []
    pred_list = []

    model.eval()

    end = time.time()
    for i, (input_, target) in enumerate(dataloader_val):
        target = target.cuda(async=True)
        input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        pred, aux = model(input_var)
        loss = criterion(pred, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input_.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        for j in range(target.shape[0]):
            target_list.append(target.cpu()[j].numpy())
            pred_list.append(np.argmax(pred.cpu().data[j].numpy(), axis=0))
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(dataloader_val), batch_time=batch_time, loss=losses))

    acc, mean_iou = evaluate(target_list, pred_list, args.num_class, result_pth, epoch)
    return acc, mean_iou, losses


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.module.conv1)
    b.append(model.module.bn1)
    b.append(model.module.layer1)
    b.append(model.module.layer2)
    b.append(model.module.layer3)
    b.append(model.module.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.module.conv_aux)
    b.append(model.module.layer5)
    b.append(model.module.layer6)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate*((1-float(epoch)/args.epochs)**args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.rar'):
    """Saves checkpoint to disk"""
    directory = "runs_No_normalize/%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % args.name + 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
