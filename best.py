# select the best acc1, acc5
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--work-dir', default='work_dirs', type=str, 
                    help='the dir to save logs and models')
parser.add_argument('--log-dir', '-ld', default='rla_resnet50', type=str, 
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


def main():
    global args
    args = parser.parse_args()
    log_dir = "%s/%s/"%(args.work_dir, args.log_dir)
    acc1_list, epo1_list = read_acclog(log_dir, log_name='val_acc1.txt')
    best_acc1 = np.max(acc1_list)
    best_idx1 = acc1_list.index(np.max(acc1_list))
    best_epo1 = epo1_list[best_idx1]
    
    acc5_list, epo5_list = read_acclog(log_dir, log_name='val_acc5.txt')
    best_acc5 = np.max(acc5_list)
    best_idx5 = acc5_list.index(np.max(acc5_list))
    best_epo5 = epo5_list[best_idx5]
    
    with open(os.path.join(log_dir, 'best.txt'), "w") as f:
        f.write("Acc@1:" + str(best_acc1) + "\n")
        f.write("Acc@5:" + str(best_acc5))
    
    print ("-" * 80)
    print ("* best Acc@1: {:.3f} at epoch {}".format(best_acc1, best_epo1))
    print ("-" * 80)
    print ("* best Acc@5: {:.3f} at epoch {}".format(best_acc5, best_epo5))
    print ("-" * 80)
    
    
if __name__ == '__main__':
    main()
    
    