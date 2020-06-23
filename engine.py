import math
import sys
import time
import torch
import utils
import torchvision
from evaluate import compute_mAP
import numpy as np

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        
        images = list(image.to(device) for image in images)
        targets_ = [{k: v.to(device) for k, v in t.items()} for t in targets]        
     
        loss_dict = model(images, targets_)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return loss_value


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, dataset, device):
    model.eval()
    ground_truth = list()
    predictions = list()
    for image, targets, seqs_and_frames in data_loader:
        image = list(img.to(device) for img in image)
        outputs = model(image)

        # add to ground truth
        for out, t, (seq, frame) in zip(outputs, targets, seqs_and_frames):
            gt_boxes = list()
            for bb in t["boxes"]:
                gt_boxes.append(list(bb.detach().cpu().numpy()))
            ground_truth.append([seq, frame, gt_boxes])

            for bb, score in zip(out["boxes"], out["scores"]):
                predictions.append([seq, frame, list(bb.detach().cpu().numpy()), float(score.detach().cpu())])
    mAP, AP = compute_mAP(predictions, ground_truth)
    print("mAP:{:.3f}".format(mAP))
    for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
        print("\tAP at IoU level [{:.2f}]: {:.3f}".format(iou, ap_metric))

