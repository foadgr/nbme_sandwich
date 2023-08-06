import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from nbme.model import CustomModel
from nbme.dataset import TrainDataset
from nbme.trainer import Trainer
from nbme.evaluation_metric import get_score
from nbme.scoring import get_char_probs, get_results, get_predictions, create_labels_for_scoring
from nbme.logging import get_logger


OUTPUT_DIR = './output/'
LOGGER = get_logger(filename=OUTPUT_DIR+'train')


class TrainingLoop:
    def __init__(self, folds, config, device):
        self.folds = folds
        self.config = config
        self.device = device

    def get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.config.encoder_lr, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.config.encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': self.config.decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def get_scheduler(self, optimizer, num_train_steps):
        if self.config.scheduler == 'linear':
            return get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.config.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif self.config.scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=self.config.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=self.config.num_cycles
            )
            
    def train_fold(self, fold):
        LOGGER.info(f"========== fold: {fold} training ==========")
        print(self.config)
        train_folds = self.folds[self.folds['fold'] != fold].reset_index(drop=True)
        valid_folds = self.folds[self.folds['fold'] == fold].reset_index(drop=True)
        valid_texts = valid_folds['pn_history'].values
        valid_labels = create_labels_for_scoring(valid_folds)

        train_dataset = TrainDataset(self.config, train_folds)
        valid_dataset = TrainDataset(self.config, valid_folds)

        train_loader = DataLoader(train_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=True,
                                num_workers=self.config.num_workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=False,
                                num_workers=self.config.num_workers, pin_memory=True, drop_last=False)

        model = CustomModel(self.config, config_path=None, pretrained=True)
        torch.save(model.config, OUTPUT_DIR + 'config.pth')
        model.to(self.device)

        optimizer_parameters = self.get_optimizer_params(model)
        optimizer = AdamW(optimizer_parameters, lr=self.config.encoder_lr, eps=self.config.eps, betas=self.config.betas)

        num_train_steps = int(len(train_folds) / self.config.batch_size * self.config.epochs)
        scheduler = self.get_scheduler(optimizer, num_train_steps)

        criterion = nn.BCEWithLogitsLoss(reduction="none")
        
        trainer = Trainer(model, criterion, optimizer, scheduler, self.device, self.config)

        best_score = 0.
        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Training
            avg_loss = trainer.training(fold, train_loader, epoch)

            # Validation
            avg_val_loss, predictions = trainer.validation(valid_loader)
            predictions = predictions.reshape((len(valid_folds), self.config.max_len))
            
            # Scoring
            char_probs = get_char_probs(valid_texts, predictions, self.config.tokenizer)
            results = get_results(char_probs, th=0.5)
            preds = get_predictions(results)
            score = get_score(valid_labels, preds)

            elapsed = time.time() - start_time

            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
            if self.config.wandb:
                wandb.log({f"[fold{fold}] epoch": epoch+1, 
                        f"[fold{fold}] avg_train_loss": avg_loss, 
                        f"[fold{fold}] avg_val_loss": avg_val_loss,
                        f"[fold{fold}] score": score})
            
            if best_score < score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': predictions},
                            OUTPUT_DIR+f"{self.config.model.replace('/', '-')}_fold{fold}_best.pth")

        predictions = torch.load(OUTPUT_DIR+f"{self.config.model.replace('/', '-')}_fold{fold}_best.pth", 
                                map_location=torch.device('cpu'))['predictions']
        valid_folds[[i for i in range(self.config.max_len)]] = predictions

        torch.cuda.empty_cache()
        gc.collect()

        return valid_folds
    
    def run(self):
        for fold in range(self.config.n_folds):
            self.train_fold(fold)