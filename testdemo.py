# System / Python
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# PyTorch
import torch
from torch.utils.data.dataloader import DataLoader
# Custom
from IXI_dataset import IXIData as Dataset
from net import ParallelNetwork as Network
from mri_tools import rAtA

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to model
parser.add_argument('--num-layers', type=int, default=9, help='number of iterations')
parser.add_argument('--in-channels', type=int, default=1, help='number of model input channels')
parser.add_argument('--out-channels', type=int, default=1, help='number of model output channels')
# batch size, num workers
parser.add_argument('--batch-size', type=int, default=4, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
# parameters related to test data
parser.add_argument('--test-path', type=str, default='/home/hc/data/IXI_T1/test', help='path of test data')
parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask_8.00x_acs24.mat', help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=1.0, help='sampling rate of test data')
# others
parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')


def validate(args):
    torch.cuda.set_device(0)

    test_set = Dataset(args.test_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.test_sample_rate)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = Network(num_layers=args.num_layers, rank=0)
    # load checkpoint
    model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(0))
    model.load_state_dict(checkpoint['model'])
    print('The model is loaded.')
    model = model.up_network.cuda(0)

    print('Now testing {}.'.format(args.exp_name))
    model.eval()
    with torch.no_grad():
        average_psnr, average_ssim, average_psnr_zerof, average_ssim_zerof, average_time, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        t = tqdm(test_loader, desc='testing', total=int(len(test_loader)))
        for iter_num, data_batch in enumerate(t):
            label = data_batch[0].to(0, non_blocking=True)
            mask_under = data_batch[1].to(0, non_blocking=True)
            # fname = data_batch[4]
            # slice_id = data_batch[5]
            under_img = rAtA(label, mask_under)
            # inference
            start_time = time.time()
            output, _ = model(under_img.permute(0, 3, 1, 2).contiguous(), mask_under)
            infer_time = time.time() - start_time
            average_time += infer_time
            output = output.permute(0, 2, 3, 1).contiguous()
            # calculate and print test information
            under_img, output, label = under_img.detach().cpu().numpy(), output.detach().cpu().numpy(), label.float().detach().cpu().numpy()
            total_num += under_img.shape[0]
            batch_psnr, batch_ssim, batch_psnr_zerof, batch_ssim_zerof = 0.0, 0.0, 0.0, 0.0
            for i in range(under_img.shape[0]):
                under_slice, output_slice, label_slice = under_img[i].squeeze(), output[i].squeeze(), label[i].squeeze()
                psnr = peak_signal_noise_ratio(label_slice, output_slice, data_range=label_slice.max())
                psnr_zerof = peak_signal_noise_ratio(label_slice, under_slice, data_range=label_slice.max())
                ssim = structural_similarity(label_slice, output_slice, data_range=label_slice.max())
                ssim_zerof = structural_similarity(label_slice, under_slice, data_range=label_slice.max())
                batch_psnr += psnr
                batch_ssim += ssim
                batch_psnr_zerof += psnr_zerof
                batch_ssim_zerof += ssim_zerof
            average_psnr += batch_psnr
            average_ssim += batch_ssim
            average_psnr_zerof += batch_psnr_zerof
            average_ssim_zerof += batch_ssim_zerof
        average_psnr /= total_num
        average_ssim /= total_num
        average_psnr_zerof /= total_num
        average_ssim_zerof /= total_num
        average_time /= total_num
    print('average_time:{:.5f}s\tzerof_psnr:{:.5f}\tzerof_ssim:{:.5f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(
        average_time, average_psnr_zerof, average_ssim_zerof, average_psnr, average_ssim))


if __name__ == '__main__':
    args_ = parser.parse_args()
    validate(args_)
