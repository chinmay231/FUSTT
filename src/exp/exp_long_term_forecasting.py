from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.mc_dropout import mc_predict, gaussian_pi, picp_mpiw


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        #print("Train data:",train_data)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print("Parameters:",sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                #start = time.time()
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        # === unified result root ===
        root_folder = os.path.join('./resultfile', setting)
        folder_vis  = os.path.join(root_folder, 'visuals')
        folder_np   = os.path.join(root_folder, 'npy')
        folder_unc  = os.path.join(root_folder, 'uncertainty')
        folder_attn = os.path.join(root_folder, 'attention')
        os.makedirs(folder_vis, exist_ok=True)
        os.makedirs(folder_np,  exist_ok=True)
        os.makedirs(folder_unc, exist_ok=True)
        os.makedirs(folder_attn, exist_ok=True)

        # MC-Dropout controls (safe defaults if not in args)
        T = getattr(self.args, 'mc_passes', 0)        # set >0 to enable MCD (e.g., 50)
        alpha = getattr(self.args, 'pi_alpha', 0.05)  # 95% PI

        # accumulators for global PICP/MPIW when MCD is on
        total_inside = 0
        total_width  = 0.0
        total_points = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward (deterministic point forecast)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]              # [B,P,F]
                y_true  = batch_y[:, -self.args.pred_len:, f_dim:]              # [B,P,F]

                # save a quick visual every 20 iters (as before)
                out_np  = outputs.detach().cpu().numpy()
                true_np = y_true.detach().cpu().numpy()
                preds.append(out_np)
                trues.append(true_np)

                if i % 20 == 0:
                    from utils.tools import visual
                    input_np = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], out_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_vis, f'{i}.pdf'))

                # === MC-Dropout uncertainty (optional) ===
                if T and T > 0:
                    mean, std = mc_predict(self.model, (batch_x, batch_x_mark, dec_inp, batch_y_mark), T=T)  # [B,P,F]
                    lo, hi   = gaussian_pi(mean, std, alpha=alpha)                                          # [B,P,F]

                    # accumulate coverage/width
                    inside = ((y_true >= lo) & (y_true <= hi)).float().sum().item()
                    width  = (hi - lo).mean().item()
                    pts    = y_true.numel()
                    total_inside += inside
                    total_width  += width
                    total_points += pts

                    # save first few batches completely for plotting
                    if i < 5:
                        np.savez(os.path.join(folder_unc, f'batch_{i}.npz'),
                                mean=mean.cpu().numpy(),
                                std=std.cpu().numpy(),
                                lo=lo.cpu().numpy(),
                                hi=hi.cpu().numpy(),
                                y=y_true.cpu().numpy())

                # === attention dump (first few batches) ===
                if self.args.output_attention and i < 3:
                    attn_pack = self.model.get_last_attn()
                    if "encoder_attn" in attn_pack:
                        for li, a in enumerate(attn_pack["encoder_attn"]):
                            np.save(os.path.join(folder_attn, f'encoder_layer{li}_batch{i}.npy'),
                                    a.detach().cpu().numpy())

        # stack predictions
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # metrics + saves
        from utils.metrics import metric
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # unified saves
        np.save(os.path.join(folder_np, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_np, 'pred.npy'), preds)
        np.save(os.path.join(folder_np, 'true.npy'), trues)

        # write a concise text summary (you asked to paste into a separate file)
        with open(os.path.join(root_folder, 'summary.txt'), 'w') as f:
            f.write(f"SETTING: {setting}\n")
            f.write(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}\n")

        # finalize MCD coverage if enabled
        if T and T > 0:
            picp = total_inside / total_points
            mpiw = total_width / max(1, len(test_loader))
            print(f"[MCD] (1-alpha)={1-alpha:.2f}  PICP={picp:.3f}  MPIW={mpiw:.6f}")
            with open(os.path.join(root_folder, 'uncertainty_summary.txt'), 'w') as f:
                f.write(f"MC_PASSES: {T}, ALPHA: {alpha}\n")
                f.write(f"PICP: {picp:.6f}\n")
                f.write(f"MPIW: {mpiw:.6f}\n")

        # still return model-compatible end
        return
