import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
                            classification_report, roc_auc_score, roc_curve, auc
import os
import torch
import numpy as np
from itertools import cycle


def loss_plot(train, val, args):
    epochs = range(1, len(train)+1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train, label='Train Loss')
    plt.plot(epochs, val,   label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Train & Validation Loss')
    
    plt.savefig(args.result_dir+'loss_plot.png')
    plt.close()



def accuracy_plot(train, val, args):
    epochs = range(1, len(train)+1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train, label='Train Acc')
    plt.plot(epochs, val,   label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train & Validation Accuracy')
    plt.legend()
    plt.savefig(args.result_dir+'accuracy_plot.png')
    plt.close()


def confusionmatrix(gt, pred, args):

    cfm = confusion_matrix(gt, pred)
    disp = ConfusionMatrixDisplay(cfm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(args.result_dir+'confusion_matrix_plot.png')
    plt.close()

def report(args, gt, pred, class_list):
    if os.path.exists(args.result_dir+args.classification_report_dir):
        os.remove(args.result_dir+args.classification_report_dir)
    report = classification_report(gt, pred, target_names=class_list)
    with open(args.result_dir+args.classification_report_dir, 'w') as f:
        f.write(report)


def roc_n_auc(args, gt, pred, class_list):
    
    # breakpoint()

    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    one_hot_gt = torch.nn.functional.one_hot(gt, num_classes=10)
    one_hot_pred = torch.nn.functional.one_hot(pred, num_classes=10)
    
    classes = np.unique(gt)

    one_hot_gt = one_hot_gt.numpy()
    one_hot_pred = one_hot_pred.numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # breakpoint()

    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(one_hot_gt[:, i], one_hot_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(one_hot_gt.ravel(), one_hot_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])

    # breakpoint()

    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(class_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Cifar10 class')
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(args.result_dir+'roc_auc_curve.png')
    plt.close()


def learning_rate_plot(args, lr_list):
    epochs = range(1, len(lr_list)+1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, lr_list, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.xticks(ticks=range(1, len(lr_list)+1, (len(lr_list)+1)//10)) # not working if epochs < 10
    plt.savefig(args.result_dir+'learning_rate.png')

    plt.close()



if __name__=='__main__':
    import torch
    from sklearn.metrics import *
    from itertools import cycle
    import numpy as np

    # train = [0.8, 0.5, 0.6, 0.6, 0.4]
    # val   = [0.5, 0.4, 0.3, 0.2, 0.4]a

    # loss_plot(train, val)
    # accuracy_plot(train, val)

    gt = torch.randint(0, 10, (1000,))
    pred = torch.randint(0, 10, (1000,))
    
    one_hot_gt = torch.nn.functional.one_hot(gt)
    one_hot_pred = torch.nn.functional.one_hot(pred)
    
    classes = np.unique(gt)

    one_hot_gt = one_hot_gt.numpy()
    one_hot_pred = one_hot_pred.numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(one_hot_gt[:, i], one_hot_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(one_hot_gt.ravel(), one_hot_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])

    breakpoint()

    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Cifar10 class')
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig('./roc_auc_curve.png')

    # breakpoint()

    a = 0