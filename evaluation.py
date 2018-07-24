import numpy as np
import pandas as pd


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(gts, predictions, num_classes, result_pth, epoch):
    hist_ = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist_ += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    hist = hist_[0:num_classes-1, 0:num_classes-1]
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    # acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iou)
    iou = pd.DataFrame(data=iou)
    iou.to_csv(result_pth+'{}.csv'.format(epoch))

    return acc, mean_iou
