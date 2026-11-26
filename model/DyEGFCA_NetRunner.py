import torch
import numpy as np
from torchinfo import summary
from datetime import datetime
from lib.metrics import RMSE_MAE_MAPE

class DyEGFCA_NetRunner():
    def __init__(
        self,
        cfg: dict,
        device,
        scaler,
        log=None,
    ):
        super().__init__()
        # self.masks = self._get_mask()
        self.cfg = cfg
        self.device = device
        self.scaler = scaler
        self.log = log

        self.clip_grad = cfg.get("clip_grad")
        self.batches_seen = 0

        self.masks = self._get_mask(cfg)
        # 初始化 LearnableMask

    def _get_mask(self,cfg):
        model_args = cfg.get("model_args", {})
        seq_len = model_args.get("in_steps", 12)
        num_nodes = model_args.get("num_nodes", 207)
        patch_len = model_args.get("patch_len", 3)
        dtype = torch.float32
        a = seq_len // patch_len
        L = num_nodes * a
        N = a
        print(L,"and",N,"and",seq_len,"and",num_nodes,"and",patch_len)
        # 处理除零错误
        if N == 0:
            N = 1
            print(f"Warning: Adjusted N to 1 (seq_len={seq_len}, patch_len={patch_len})")

        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(
                dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0) #[207,3,207]
        print("mask is not none")
        return masks


    def train_one_epoch(self, model, trainset_loader, optimizer, scheduler, criterion, epoch):
        model.train()
        batch_loss_list = []
        task_list, aux_list = [], []
        for x_batch, y_batch in trainset_loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)

            y_true = y[..., [0]]
            y_cov = y[..., 1:]
            output, moe_loss1 = model(x, y_cov, self.masks, self.scaler.transform(y_true), self.batches_seen,
                                      is_training=True)
            y_pred = self.scaler.inverse_transform(output)
            self.batches_seen += 1
            task_loss = criterion(y_pred, y_true)
            aux = 0.05 * moe_loss1
            loss = task_loss + aux
            batch_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            optimizer.step()
        epoch_loss = np.mean(batch_loss_list)
        scheduler.step()
        return epoch_loss

    @torch.no_grad()
    def eval_model(self, model, valset_loader, criterion):
        model.eval()
        batch_loss_list = []
        targets, predictions = [], []
        for x_batch, y_batch in valset_loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)
            y_true = y[..., [0]]
            y_cov = y[..., 1:]
            output ,_ = model(x, y_cov,self.masks,is_training=False)
            y_pred = self.scaler.inverse_transform(output)
            loss = criterion(y_pred, y_true)
            batch_loss_list.append(loss.item())
        val_loss = np.mean(batch_loss_list)
        return val_loss

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()
        y_list = []
        out_list = []
        for x_batch, y_batch in loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)
            y_true = y[..., [0]]
            y_cov = y[..., 1:]
            output,_ = model(x, y_cov,self.masks,is_training=False)
            y_pred = self.scaler.inverse_transform(output)
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            out_list.append(y_pred)
            y_list.append(y_true)
        out = np.vstack(out_list).squeeze()  # (samples, out_steps, num_nodes)
        y = np.vstack(y_list).squeeze()
        return y, out

    def model_summary(self, model, dataloader):
        x_shape = next(iter(dataloader))[0].shape
        y_cov_shape = next(iter(dataloader))[1][..., 1:].shape
        masks_shape = self.masks.shape
        return summary(
            model,
            [x_shape, y_cov_shape,masks_shape],
            verbose=0,  # avoid print twice
            device=self.device,
        )
