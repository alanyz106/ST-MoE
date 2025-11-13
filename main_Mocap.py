import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
sys.path.append('..')
from tensorboardX import SummaryWriter
from utils import other_utils as util
from tqdm import tqdm
from option.option_Mocap import Options
from Dataset_tools import Dataset_Mocap as datasets
from model import Experts as model
from model.ST_MoE import ST_MoE as ST_MoE
import torch.nn as nn
import os
from datetime import datetime  # 添加datetime导入
import time

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def main(opt):
    seed = 1234567890
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        select_cuda = opt.cudaid
        device = torch.device(f'cuda:{select_cuda}')
        torch.cuda.set_device(device)
        print(f"The using GPU is device {select_cuda}")
    else:
        device = torch.device('cpu')
        print("No GPU available, using CPU.")
    opt.device = device
    if opt.mode == 'train':
        print('>>> DATA loading >>>')
        dataset = datasets.Datasets(opt, mode='train')
        eval_dataset = datasets.Datasets(opt, mode='test')

        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        eval_data_loader = DataLoader(eval_dataset, batch_size=opt.test_batch, shuffle=True, num_workers=12, pin_memory=True)
    elif opt.mode == 'test':
        print('>>> DATA loading >>>')
        dataset = datasets.Datasets(opt, mode='test')

        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        # data_loader = DataLoader(dataset, batch_size=opt.test_batch, shuffle=True, num_workers=12, pin_memory=False)
        data_loader = DataLoader(dataset, batch_size=opt.test_batch, shuffle=False, num_workers=0, pin_memory=True)

    in_features = opt.in_features
    nb_kpts = int(in_features/3)  # number of keypoints
    
    body_edges = np.array(
        [[0,1], [1,2],[2,3],[0,4],
        [4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
    )

    print('>>> MODEL >>>')
    if opt.model == 'ST_MoE':
        net_pred = ST_MoE(seq_len=opt.seq_len, d_model=opt.d_model, opt=opt, num_kpt=nb_kpts, dataset="Mocap")
        print(net_pred)


    for p in net_pred.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
            
    net_pred.to(device)
    lr_now = opt.lr_now
    start_epoch = 1
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
        
    if opt.mode == 'test':
        if '.pth.tar' in opt.ckpt:
            model_path_len = opt.ckpt
        elif opt.test_epoch is not None:
            model_path_len = '{}/ckpt_epo{}.pth.tar'.format(opt.ckpt, opt.test_epoch)
        else:
            model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)

        print(">>> loading ckpt from '{}'".format(model_path_len))
        # 自动根据 device 选择加载到 CPU 或 GPU
        ckpt = torch.load(model_path_len, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        lr_now = ckpt['lr']

        net_pred.load_state_dict(ckpt['state_dict'],strict=False)
        
        print(">>> ckpt loaded (epoch: {} | err: {} | lr: {})".format(ckpt['epoch'], ckpt['err'], lr_now))
        
         # 打印模型参数名称和形状
        # print("\n=== 检查点模型参数 ===")
        # for name, param in net_pred.named_parameters():
        #     print(f"参数名: {name}, 形状: {param.shape}")
    

    if opt.mode == 'train': #train

        optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)

        util.save_ckpt({'epoch': 0, 'lr': lr_now, 'err': 0, 'state_dict': net_pred.state_dict(), 'optimizer': optimizer.state_dict()}, 0, dataset="Mocap",opt=opt)
        # writer = SummaryWriter(opt.tensorboard)
        mpjpe_flag = 10000
        total_iteration_time = 0.0  # 用于累积每个 epoch 的总迭代时间
        for epo in tqdm(range(start_epoch, opt.epoch + 1)):
            # 获取当前时间并格式化
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ret_train,avg_iteration_time = run_model(nb_kpts, net_pred, opt.batch_size, optimizer, data_loader=data_loader, opt=opt, epo=epo)
            total_iteration_time += avg_iteration_time  # 累加迭代时间
            mpjpe_mean, mpjpe_avg, ape_mean, vim_mean = eval(opt, net_pred, eval_data_loader, nb_kpts, epo)
            # exit(0)
            # writer.add_scalar('scalar/train', ret_train['loss_train'], epo)

            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / 50))

             # 将时间戳加入日志数据和列头
            ret_log = np.array([current_time, epo, lr_now, mpjpe_mean], dtype=object)
            head = np.array(['time', 'epoch', 'lr', 'mpjpe_mean'])
            
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            util.save_csv_log(opt, head, ret_log, is_create=(epo == 1), file_name="train_log")
            if mpjpe_mean < mpjpe_flag:
                isbest = True
                mpjpe_flag = mpjpe_mean
            else:
                isbest = False

            print('time: {}, epo{}, train error: {:.3f}, mpjpe_mean: {:.3f}, best_mean: {:.3f}, mpjpe_avg: {:.3f}, ape_mean: {:.3f}, vim_mean: {:.3f}, lr: {:.6f}'
      .format(current_time, epo, ret_train['loss_train'], mpjpe_mean, mpjpe_flag, mpjpe_avg, ape_mean, vim_mean, lr_now))
            print(opt.ckpt)

            util.save_ckpt({'epoch': epo,
                            'lr': lr_now,
                            'err': ret_train['loss_train'],
                            'state_dict': net_pred.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           epo, dataset="Mocap",opt=opt,Isbest=isbest)
        # writer.close()

    else: #test

        run_model(nb_kpts, net_pred, opt.test_batch, data_loader=data_loader, opt=opt)

def run_model(nb_kpts, net_pred, batch_size, optimizer=None, data_loader=None, opt=None, epo=0):

    n = 0
    if opt.mode == 'train': #train
        net_pred.train()
        loss_train = 0
        total_iteration_time = 0.0  # 用于累积每个 epoch 的总迭代时间
        for batch_idx, (x, y) in enumerate(data_loader): # in_n + kz
            start_time = time.time()  # 记录迭代开始时间
            torch.cuda.empty_cache()
            if np.shape(x)[0] < batch_size:
                continue #when only one sample in this batch
            n += batch_size

            x = x.float().cuda()
            y = y.float().cuda()

            x_c = x.clone().detach()
            y_c = y.clone().detach()

            data_out, mix_loss = net_pred(x_c, y_c)

            data_gt = y_c.transpose(2, 3)
            loss = mix_loss

            optimizer.zero_grad()
            loss.backward()
            loss_train += loss.item() * batch_size
            optimizer.step()
            end_time = time.time()  # 记录迭代结束时间
            iteration_time = end_time - start_time  # 计算单次迭代时间
            total_iteration_time += iteration_time  # 累加迭代时间
            # print(batch_idx)
        # print(n)
        res_dic = {"loss_train" : loss_train / n }
        avg_iteration_time = total_iteration_time / (batch_idx+1)
        print(f"Average iteration time per batch: {avg_iteration_time:.4f} seconds")
        return res_dic,avg_iteration_time

    else: #test
        net_pred.eval()
        mpjpe_joi = np.zeros([opt.seq_len])
        ape_joi = np.zeros([5])
        vim_joi = np.zeros([5])
        all_predictions = []  # 新增：收集所有预测结果
        # n = 0
        for batch_idx, (x, y) in enumerate(data_loader): # raw_in_n + out_n
            if np.shape(x)[0] < batch_size:
                continue #when only one sample in this batch
            n += batch_size

            x = x.float().to(opt.device) #[96, 3, 45, 75],[bs,n,j*3,seqlen]
            y = y.float().to(opt.device) #[96, 3, 45, 75]
            # print(x.shape, y.shape)

            data_out, _ = net_pred(x, y) # torch.Size([96, 3, 75, 45])
            data_gt = y.transpose(2, 3)
            num_per = y.shape[1]
            if opt.isSavePredictionstoNpy:
                # 保存当前batch的预测结果
                batch_pred = data_out.detach().cpu().numpy()
                all_predictions.append(batch_pred)

            data_gt = data_gt.reshape(batch_size, num_per, opt.seq_len, nb_kpts, 3)
            data_out = data_out.reshape(batch_size, num_per, opt.seq_len, nb_kpts, 3)
            tmp_joi = torch.sum(torch.mean(torch.mean(torch.norm(data_gt - data_out, dim=4), dim=3), dim=1), dim=0) 
            # print(tmp_joi)
            mpjpe_joi += tmp_joi.cpu().data.numpy() #[seq_len]

            tmp_ape_joi = APE(data_out[:, :, opt.frame_in:, :, :], data_gt[:, :, opt.frame_in:, :, :], [4, 9, 14, 19, 24])
            ape_joi += tmp_ape_joi#.data.numpy()

            data_vim_gt = data_gt[:, :, opt.frame_in:, :, :].transpose(2, 1)
            data_vim_gt = data_vim_gt.reshape(batch_size, opt.seq_len, -1, 3)
            data_vim_pred = data_out[:, :, opt.frame_in:, :, :].transpose(2, 1)
            data_vim_pred = data_vim_pred.reshape(batch_size, opt.seq_len, -1, 3)
            tmp_vim_joi = batch_VIM(data_vim_gt.cpu().data.numpy(), data_vim_pred.cpu().data.numpy(), [4, 9, 14, 19, 24])
            vim_joi += tmp_vim_joi#.data.numpy()

        if opt.isSavePredictionstoNpy:
            # 合并所有batch的预测结果
            all_predictions = np.concatenate(all_predictions, axis=0)
            # 新增：打印输出形状
            print("预测结果的形状:", all_predictions.shape)  

            # 保存为npy文件
            np.save(os.path.join(opt.ckpt, f'{opt.dataset}_{opt.model}_predictions.npy'), all_predictions)
        
        # mpjpe_joi 集合了所有batch的误差值，要取均值
        mpjpe_joi = mpjpe_joi/n * 1000  # n = testing dataset length
        ape_joi = ape_joi/n * 1000 * batch_size
        vim_joi = vim_joi/n * 100
        # print(ape_joi.shape, vim_joi.shape)
        # select_frame = [4, 9, 14, 19, 24]
        select_frame = [4,14,24]

        print(mpjpe_joi[opt.frame_in:][select_frame])
        print("APE: ", ape_joi)
        print("VIM: ", vim_joi)

        mpjpe_avg = np.mean(mpjpe_joi[opt.frame_in:][select_frame])
        mpjpe_mean = np.mean(mpjpe_joi[opt.frame_in:])
        ape_mean = np.mean(ape_joi[[0, 2, 4]])
        vim_mean = np.mean(vim_joi)

        res_dic = {"mpjpe_joi": mpjpe_joi}


        if opt.save_results:
            import json
            key_exp = opt.exp + '_testepo'+str(opt.test_epoch)
            print('save name exp:', opt.exp)
            print('MPJPE mean: ', mpjpe_mean)
            print('MPJPE AVG: ', mpjpe_avg)
            print('APE_mean: ', ape_mean)
            print('VIM_mean: ', vim_mean)

            ts = "AGV"

            results = {key_exp: {}}
            results[key_exp][ts]={"mpjpe_joi": mpjpe_joi.tolist()}

            with open('{}/results.json'.format(opt.ckpt), 'w') as w:
                json.dump(results, w)

        return res_dic

def eval(opt, net_pred, data_loader, nb_kpts, epo):
    net_pred.eval()
    mpjpe_joi = np.zeros([opt.seq_len])
    ape_joi = np.zeros([5])
    vim_joi = np.zeros([5])
    n = 0

    for batch_idx, (x, y) in enumerate(data_loader): # in_n + kz
        if np.shape(x)[0] < opt.test_batch:
            continue #when only one sample in this batch
        n += opt.test_batch

        # if batch_idx == 0:  # 仅打印第一个 batch
        #     print(f"Epoch {epo}, First Batch - Sample 0 (x):", x[0, 0, 0, 0].item())
        #     print(f"Epoch {epo}, First Batch - Sample 0 (y):", y[0, 0, 0, 0].item())

        x = x.float().cuda()
        y = y.float().cuda()


        data_out, loss = net_pred(x, y)#[:,:,0]  # bz, 2kz, 108
        num_per = y.shape[1]
        data_gt = y.transpose(2, 3)


        # print(data_out.shape, data_gt.shape)
        data_gt = data_gt.reshape(opt.test_batch, num_per, opt.seq_len, nb_kpts, 3)
        data_out = data_out.reshape(opt.test_batch, num_per, opt.seq_len, nb_kpts, 3)
        tmp_joi = torch.sum(torch.mean(torch.mean(torch.norm(data_gt - data_out, dim=4), dim=3), dim=1), dim=0)
        # print(tmp_joi)
        mpjpe_joi += tmp_joi.cpu().data.numpy()

        tmp_ape_joi = APE(data_out[:, :, opt.frame_in:, :, :], data_gt[:, :, opt.frame_in:, :, :], [4, 9, 14, 19, 24])
        ape_joi += tmp_ape_joi#.data.numpy()

        data_vim_gt = data_gt[:, :, opt.frame_in:, :, :].transpose(2, 1)
        data_vim_gt = data_vim_gt.reshape(opt.test_batch, opt.seq_len, -1, 3)
        data_vim_pred = data_out[:, :, opt.frame_in:, :, :].transpose(2, 1)
        data_vim_pred = data_vim_pred.reshape(opt.test_batch, opt.seq_len, -1, 3)
        tmp_vim_joi = batch_VIM(data_vim_gt.cpu().data.numpy(), data_vim_pred.cpu().data.numpy(), [4, 9, 14, 19, 24])
        vim_joi += tmp_vim_joi#.data.numpy()

    # print(n)

    mpjpe_joi = mpjpe_joi/n * 1000  # n = testing dataset length
    ape_joi = ape_joi/n * 1000 * opt.test_batch
    vim_joi = vim_joi/n * 1000
    # print(ape_joi.shape, vim_joi.shape)
    print(mpjpe_joi)
    print("APE: ", ape_joi)
    print("VIM: ", vim_joi)
    # select_frame = [4, 9, 14, 19, 24]
    select_frame = [4, 14, 24]
    mpjpe_mean = np.mean(mpjpe_joi[opt.frame_in:][select_frame])
    mpjpe_avg = np.mean(mpjpe_joi[opt.frame_in:])
    ape_mean = np.mean(ape_joi[[0,2,4]])
    vim_mean = np.mean(vim_joi)

    if opt.save_results:
        import json
        key_exp = 'epoch:'+str(epo)
        # 获取当前时间并格式化
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = {key_exp: {}}
        # 添加时间戳字段
        results[key_exp]["time"] = current_time  # 使用之前定义的current_time变量
        results[key_exp]["mpjpe_joi"] = mpjpe_joi.tolist()
        results[key_exp]["mpjpe_mean"] = mpjpe_mean.tolist()
        results[key_exp]["ape_joi"] = ape_joi.tolist()
        results[key_exp]["ape_mean"] = ape_mean.tolist()
        results[key_exp]["vim_joi"] = vim_joi.tolist()
        results[key_exp]["vim_mean"] = vim_mean.tolist()
        
        with open('{}/eval_results.json'.format(opt.ckpt), 'a') as w:
            json.dump(results, w)
            w.write('\n')

    return mpjpe_mean, mpjpe_avg, ape_mean, vim_mean

def APE(V_pred, V_trgt, frame_idx):

    V_pred = V_pred - V_pred[:, :, :, 0:1, :]
    V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]

    err = np.arange(len(frame_idx), dtype=np.float_)

    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
    return err

def batch_VIM(GT, pred, select_frames):
    '''Calculate the VIM at selected timestamps.

    Args:
        GT: [B, T, J, 3].

    Returns:
        errorPose: [T].
    '''
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, axis=(2, 3))
    errorPose = np.sqrt(errorPose)
    errorPose = errorPose.sum(axis=0)
    # scale = 100
    return errorPose[select_frames]# * scale

if __name__ == '__main__':
    option = Options().parse()
    main(option)


