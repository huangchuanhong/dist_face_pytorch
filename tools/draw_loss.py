import matplotlib.pyplot as plt

lr_step = 0.1
base_lr = 1

def get_loss(log_file):
    losses_list = []
    losses = []
    pre_lr = base_lr
    count = 0
    with open(log_file) as f:
        for line in f.readlines():
            if not 'loss: ' in line:
                continue
            count += 1
            loss = float(line.strip().split(':')[-1].strip())
            lr = float(line.strip().split('lr:')[1].split(',')[0].strip())
            if abs(pre_lr * lr_step - lr) < 1e-10:
                losses_list.append(losses)
                losses = []
                pre_lr = lr
            losses.append(loss)
        losses_list.append(losses)
    return losses_list, count

def draw_loss(losses_list, count):
    plt.figure()
    plt.xlim(0, count)
    #plt.ylim(0, 7)
    curr = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for idx, losses in enumerate(losses_list):
        x = list(range(curr, curr + len(losses)))
        plt.plot(x, losses, colors[idx])
        #plt.hold(True)
        curr += len(losses)
    #plt.show()
    plt.savefig('loss.jpg')


#file_path = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/25w_regnetp_circleloss_nbase_1top_mt_244_243_248_251/20200526_075418.log'
#file_path = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/97w_regnet_amsoftmax_nbase_1top_mt_244_243_248_251/20200512_001535.log'
file_path = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/25w_regnet_circleloss_nbase_1top_mt_249_244_245_246_wd_0_00001/20200603_030751.log'
losses_list, count = get_loss(file_path)
draw_loss(losses_list, count)

