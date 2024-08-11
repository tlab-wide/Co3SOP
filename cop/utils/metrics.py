# Some sections of this code reused code from SemanticKITTI development kit
# https://github.com/PRBonn/semantic-kitti-api

import numpy as np
import torch
import torchmetrics
import copy



class Metrics:

    def __init__(self, n_classes):

        self.n_classes = n_classes
        self.confusionMatrix = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=self.n_classes).cuda()
        self.binary_confusion_matrix = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).cuda()
        self.reset()
        self.metrics = {}
        self.total_loss = 0
        self.iteration = 0

    def reset(self):
        self.total_loss = 0
        self.iteration = 0
        self.confusionMatrix.reset()
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

    def add_batch(self, output, target, loss):
        # print(loss)
        self.total_loss = self.total_loss + loss
        self.iteration += 1
        # output = np.argmax(output.detach().cpu().numpy(), axis=1)
        # target = target.detach().cpu().numpy()
        self.confusionMatrix.update(output, target)
        occupancy = torch.argmax(output, dim=1)
        occupancy_laebl = target.clone()
        occupancy[occupancy==23] = 0
        occupancy[occupancy<23] = 1
        occupancy_laebl[occupancy_laebl==23] = 0
        occupancy_laebl[occupancy_laebl<23] = 1
        self.binary_confusion_matrix.update(occupancy, occupancy_laebl)
        
        
        
        # mask = (output >= 0) & (output < self.n_classes)
        # label = self.n_classes * output[mask] + target[mask]
        # count = np.bincount(label, minlength=self.n_classes ** 2)
        # confusionMatrix = count.reshape(self.n_classes, self.n_classes)
        # self.confusionMatrix += confusionMatrix

        
        # tp, fp, fn = self.get_score_completion(output, target)

        # self.completion_tp += tp
        # self.completion_fp += fp
        # self.completion_fn += fn
        # # if nonempty is not None:
        # #     mask = mask & nonempty
        # tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(output, target)
        # self.tps += tp_sum
        # self.fps += fp_sum
        # self.fns += fn_sum
        
        # self.get_stats()

    def get_stats(self):
        confusion_matrix = self.confusionMatrix.confmat.cpu().numpy()
        IoU = self.get_iou(confusion_matrix)
        self.metrics = {
            "loss": self.total_loss/self.iteration,
            "IoU": IoU,
            "mIoU": np.nanmean(IoU),
            "completion":  self.get_conpletion(self.binary_confusion_matrix.confmat.cpu().numpy())
        }
        return self.metrics
    
    def get_iou(self, confusion_matrix):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        # self.confusionMatrix.
        intersection = np.diag(confusion_matrix)  # 取对角元素的值，返回列表
        union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
            confusion_matrix) + 1e-5  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU
    
    def get_conpletion(self, binary_confusion_matrix):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        # self.confusionMatrix.
        intersection = np.sum(binary_confusion_matrix[1:,1:])  # 取对角元素的值，返回列表
        union = np.sum(binary_confusion_matrix[0,1:]) + np.sum(binary_confusion_matrix[1:,0]) + intersection + 1e-5  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def get_miou(self, confusion_matrix):
        mIoU = np.nanmean(self.get_iou)  # 求各类别IoU的平均
        return mIoU
    
    def get_score_completion(self, predict, target, nonempty=None):
        # predict = np.copy(predict)
        # target = np.copy(target)
        # predict = np.copy(predict.data.cpu().numpy())
        # target = np.copy(target.data.cpu().numpy())

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[predict== 255] = 0
        target[target == 255] = 0
        # ---- flatten
        # print(target.shape)
        # print(predict.shape)
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict <23] = 1
        b_true[target <23] = 1
        b_pred[predict == 23] = 0
        b_true[target == 23] = 0
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            target = b_true[idx, :]  # GT
            output = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                target = target[nonempty_idx == 1]
                output = output[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(target == 1, output == 1))).size
            fp = np.array(np.where(np.logical_and(target != 1, output == 1))).size
            fn = np.array(np.where(np.logical_and(target == 1, output != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        # predict = np.copy(predict)
        # target = np.copy(target)
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        # predict[target == 255] = 0
        # target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600
        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn
        
        for idx in range(_bs):
            # print(idx)
            labels = target[idx, :]  # GT
            output = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                output = output[
                    np.where(np.logical_and(nonempty_idx == 1, labels != 255))
                ]
                labels = labels[
                    np.where(np.logical_and(nonempty_idx == 1, labels != 255))
                ]
            for j in range(_C):  # for each class
                tp = np.array(np.where(np.logical_and(labels == j, output == j))).size
                fp = np.array(np.where(np.logical_and(labels != j, output == j))).size
                fn = np.array(np.where(np.logical_and(labels == j, output != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum


