import os.path as osp
import glob
import cv2
import numpy as np
import torch
from models.ResDNet import ResDNet
from models.SRResDNet import SRResDNet
from collections import OrderedDict

def crop_forward(model, x, sf, shave=10, min_size=100000, bic=None):
    """
        chop for less memory consumption during test
        """
    n_GPUs = 1
    scale = sf
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]
    
    if bic is not None:
        bic_h_size = h_size*scale
        bic_w_size = w_size*scale
        bic_h = h*scale
        bic_w = w*scale
        
        bic_list = [
            bic[:, :, 0:bic_h_size, 0:bic_w_size],
            bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
            bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
            bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]
        
    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if bic is not None:
                bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)
            
            sr_batch_temp = model(lr_batch)
            
            if isinstance(sr_batch_temp, list):
                sr_batch = sr_batch_temp[-1]
            else:
                sr_batch = sr_batch_temp
                
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            crop_forward(model, x=patch, sf=scale, shave=shave, min_size=min_size) \
            for patch in lr_list
            ]
        
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale
    
    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        

    return output


def main():
    model_path = 'trained_nets_x4/srrescycgan.pth'
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')
    test_img_folder = 'LR/*'
    scale_factor = 4
    use_chop = True

    # loading model
    resdnet = ResDNet(depth=5)
    model = SRResDNet(resdnet, scale=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    test_results = OrderedDict()
    test_results['time'] = []
    idx = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for path_lr in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path_lr))[0]
        print('Img:', idx, base)
        # read images: LR
        img_lr = cv2.imread(path_lr, cv2.IMREAD_COLOR)
        img_LR = torch.from_numpy(np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img_LR.unsqueeze(0)
        img_LR = img_LR.to(device)
        #print('img_LR:', img_LR.shape, img_LR.min(), img_LR.max())

        start.record()
        with torch.no_grad():
            if use_chop:
                output_SR = crop_forward(model, img_LR, sf=scale_factor)
            else:
                output_SR = model(img_LR)
        end.record()
        torch.cuda.synchronize()
        end_time = start.elapsed_time(end)
        test_results['time'].append(end_time)  # milliseconds
        output_sr = output_SR.data.squeeze().float().cpu().clamp_(0, 255).numpy()
        output_sr = np.transpose(output_sr[[2, 1, 0], :, :], (1, 2, 0))
        #print('output:', output.shape, output.min(), output.max())

        print('{:->4d}--> {:>10s}, time: {:.4f} miliseconds.'.format(idx, base, end_time))

        # save images
        cv2.imwrite('sr_results_x4/{:s}.png'.format(base), output_sr)

        del img_LR, img_lr
        del  output_SR, output_sr
        torch.cuda.empty_cache()

    avg_time = sum(test_results['time']) / len(test_results['time']) / 1000.0
    print('Avg. Time:{:.4f} seconds.'.format(avg_time))


if __name__ == '__main__':
    main()
