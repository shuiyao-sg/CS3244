import time
import torch
import numpy as np

import config

def train(train_loader, model, criterion, optimizer, epoch, device='cuda', writer_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    outputs = None

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        input = data[0].to(device)
        target = data[1].to(device)

        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output = model(input)
        if outputs is not None:
            outputs = np.vstack([outputs, output.detach().to('cpu').numpy()])
        else:
            outputs = output.detach().to('cpu').numpy()

        # measure accuracy and record loss
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
    return losses.avg, outputs


def validate(val_loader, model, criterion, device='cuda', writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    outputs = None

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            input = data[0].to(device)
            target = data[1].to(device)

            # compute output
            output = model(input)
            if outputs is not None:
                outputs = np.vstack([outputs, output.detach().to('cpu').numpy()])
            else:
                outputs = output.detach().to('cpu').numpy()

            # measure accuracy and record loss
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Val: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t'.format(
                  batch_time=batch_time, loss=losses)
        print(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg, outputs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
