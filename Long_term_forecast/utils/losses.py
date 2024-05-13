import torch
import torch.nn as nn


class mae_loss(nn.Module):
    def __init__(self):
        super(mae_loss, self).__init__()

    @staticmethod
    def forward(pred, true, mask_value=None):
        if mask_value is not None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(true - pred))


class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()

    @staticmethod
    def forward(pred, true, mask_value=None):
        if mask_value is not None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.mean((pred - true) ** 2)


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    @staticmethod
    def forward(pred, true, mask_value=None):
        if mask_value is not None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(torch.div((true - pred), true)))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    @staticmethod
    def forward(pred, true, mask_value=None):
        if mask_value is not None:
            mask = torch.gt(true, mask_value)
            pred = torch.masked_select(pred, mask)
            true = torch.masked_select(true, mask)
        return torch.mean(torch.abs(true - pred) / (torch.abs(true) + torch.abs(pred)))
