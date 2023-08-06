import torch
import numpy as np
import time
from tqdm import tqdm

from nbme.tracking import AverageMeter, timeSince


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

    def training(self, fold, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        start = end = time.time()
        global_step = 0
        for step, (inputs, labels) in enumerate(train_loader):
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            y_preds = self.model(inputs)
            loss = self.criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.update(loss.item(), batch_size)
            if self.config.batch_scheduler:
                self.scheduler.step()
            if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
                print(
                    'Epoch: [{0}][{1}/{2}] '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'Grad: {grad_norm:.4f}  '
                    'LR: {lr:.8f}  '
                    .format(
                        epoch+1, step, len(fold), 
                        remain=timeSince(start, float(step+1)/len(fold)),
                        loss=losses,
                        grad_norm=grad_norm,
                        lr=scheduler.get_lr()[0]
                    )
                )
            if CFG.wandb:
                wandb.log(
                    {f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_lr()[0]}
                )
        return losses.avg

    def validation(self, valid_loader):
        self.model.eval()
        losses = AverageMeter()
        preds = []
        for step, (inputs, labels) in enumerate(valid_loader):
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            labels = labels.to(self.device)
            batch_size = labels.size(0)
            with torch.no_grad():
                y_preds = self.model(inputs)
            loss = self.criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
            losses.update(loss.item(), batch_size)
            preds.append(y_preds.sigmoid().to('cpu').numpy())
            if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
                print(
                    'EVAL: [{0}/{1}] '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    .format(
                        step, len(valid_loader),
                        loss=losses,
                        remain=timeSince(start, float(step+1)/len(valid_loader))
                    )
                )
        predictions = np.concatenate(preds)
        return losses.avg, predictions

    def inference(self, test_loader):
        self.model.eval()
        preds = []
        for inputs in tqdm(test_loader, total=len(test_loader)):
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                y_preds = self.model(inputs)
            preds.append(y_preds.sigmoid().to('cpu').numpy())
        predictions = np.concatenate(preds)
        return predictions
