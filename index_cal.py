import torch
from torch import nn

def mse_c(pred, true, mask):
    if mask is not None:
        error = (pred - true) ** 2
        mse = error.sum() / mask.sum()
    else:
        mse = ((pred - true) ** 2).mean()
    return mse

def mae_c(pred, true, mask):
    if mask is not None:
        error = torch.abs(pred - true)
        mae = error.sum() / mask.sum()
    else:
        mae = torch.abs(pred - true).mean()
    return mae

def std_c(pred, true, mask):
    error = pred - true
    if mask is not None:
        error_mean = error.sum() / mask.sum()
        std = torch.sqrt(((error - error_mean) ** 2 * mask).sum() / mask.sum())
    else:
        std = torch.std(error)
    return std

# def iqr_c(pred, true, mask):
#     if mask is not None:
#         error = pred - true
#         q1 = torch.quantile(error[mask > 0], 0.25)
#         q3 = torch.quantile(error[mask > 0], 0.75)
#     else:
#         q1 = torch.quantile(pred - true, 0.25)
#         q3 = torch.quantile(pred - true, 0.75)
#     iqr = q3 - q1
#     return iqr

def iqr_c(pred, true, mask):
    error = pred - true
    masked_errors_q1 = []
    masked_errors_q3 = []

    if mask is not None:
        for i in range(error.shape[0]):
            valid_errors = error[i][mask[i] > 0]
            if valid_errors.numel() > 0:
                masked_errors_q1.append(torch.quantile(valid_errors, 0.25))
                masked_errors_q3.append(torch.quantile(valid_errors, 0.75))

        if len(masked_errors_q1) > 0:
            q1 = torch.stack(masked_errors_q1)
            q3 = torch.stack(masked_errors_q3)
            iqr = q3 - q1
            return iqr.mean()
        else:
            return torch.tensor(float('nan'), device = error.device)
        
    else:
        if error.numel() > 0:
            q1 = torch.quantile(error, 0.25, dim = 1)
            q3 = torch.quantile(error, 0.75, dim = 1)
            iqr = q3 - q1
            return iqr.mean()
        else:
            return torch.tensor(float('nan'), device = error.device)
        
def r2_c(pred, true, mask):
    error = pred - true
    
    if mask is not None:
        mask_sum = mask.sum(dim = 1, keepdim = True)
        mean_true = true.sum(dim = 1, keepdim = True) / mask_sum.clamp(min = 1e-6)
        ss_res = (error ** 2).sum(dim = 1)
        ss_tot = (((true - mean_true) ** 2) * mask).sum(dim = 1)  # 计算每个样本的总方差
    else:
        mean_true = true.mean(dim = 1, keepdim = True)
        ss_res = (error ** 2).sum(dim = 1)
        ss_tot = ((true - mean_true) ** 2).sum(dim = 1)

    r2_per_sample = 1 - ss_res / ss_tot.clamp(min = 1e-6)
    r2_mean = r2_per_sample.mean()

    return r2_mean



















