import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
# from collections import Counter


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--work-dir', default='work_dirs', type=str, 
                    help='the dir to save logs and models')
parser.add_argument('--log-dir', '-ld', default='rla_resnet50_', type=str, 
                    help='the sub dir to save logs and models of a model architecture')


def read_acclog(log_dir, log_name):
    # log_dir = '/home/r12user3/Projects/RLA/ImageNet/RLANet/work_dirs/rla_resnet50/'
    # log_name = 'val_acc1.txt'
    with open(os.path.join(log_dir, log_name),"r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
        
    acc_list = []
    epo_list = []
    for i in range(len(lines)):
        epo_i = lines[i].split(' ')[0]
        acc_i = lines[i].split('(')[1]
        acc_i = acc_i.split(',')[0]
        epo_list.append(int(epo_i))
        acc_list.append(float(acc_i))
    return acc_list, epo_list

def read_losslog(log_dir, log_name):
    # log_dir = '/home/r12user3/Projects/RLA/ImageNet/RLANet/work_dirs/rla_resnet50/'
    # log_name = 'loss_plot.txt'
    with open(os.path.join(log_dir, log_name),"r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    
    loss_list = []
    epo_list = []
    for i in range(len(lines)):
        epo_i = lines[i].split(' ')[0]
        loss_i = lines[i].split(' ')[-1]
        epo_list.append(int(epo_i))
        loss_list.append(round(float(loss_i), 3))
    return loss_list, epo_list


def plot_loss(loss, epochs, save_path, plot_name):
    plt.figure()
    plt.plot(epochs, loss, label='training')
    # plt.plot(epochs, val_loss, label='validation')
    plt.title('Training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(min(epochs), max(epochs))
    plt.ylim(int(min(loss)), int(max(loss)))
    plt.yticks(range(int(min(loss)), int(max(loss)), 1))
    plt.legend(loc='upper right')
    plt.savefig(save_path + '{}.png'.format(plot_name))
    plt.show()
        
def plot_acc(acc, val_acc, epochs, val_epochs, 
             save_path, plot_name):
    plt.figure()
    plt.plot(epochs, acc, label='training')
    plt.plot(val_epochs, val_acc, label='validation')
    plt.title('Training and validation acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.xlim(min(epochs), max(epochs))
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.savefig(save_path + '{}.png'.format(plot_name))
    plt.show()



def main():
    global args
    args = parser.parse_args()
    log_dir = "%s/%s/"%(args.work_dir, args.log_dir)
    save_path = log_dir
    
    acc1_list, epo1_list = read_acclog(log_dir, log_name='train_acc1.txt')
    val_acc1_list, val_epo1_list = read_acclog(log_dir, log_name='val_acc1.txt')
    plot_acc(acc1_list, val_acc1_list, epo1_list, val_epo1_list, 
             save_path, plot_name='acc1_plot')
    
    acc5_list, epo5_list = read_acclog(log_dir, log_name='train_acc5.txt')
    val_acc5_list, val_epo5_list = read_acclog(log_dir, log_name='val_acc5.txt')
    plot_acc(acc5_list, val_acc5_list, epo5_list, val_epo5_list, 
             save_path, plot_name='acc5_plot')
    
    loss_list, epo_list = read_losslog(log_dir, log_name='loss_plot.txt')
    plot_loss(loss_list, epo_list, save_path, plot_name='loss_plot')
    
        
if __name__ == '__main__':
    main()