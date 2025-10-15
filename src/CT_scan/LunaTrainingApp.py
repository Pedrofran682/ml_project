import argparse
import logging
import datetime
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from .LunaDataset import LunaDataset
from torch.utils.data import DataLoader
from .util import enumerateWithEstimate
from .model import LunaModel

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDS = 2
METRICS_SIZE = 3
log = logging.getLogger(__name__)


class LunaTrainigApp:
    def __init__(self, sys_argv=None):
        if sys_argv is not None:
            sys_argv = sys_argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=2,
                            type=int)
        parser.add_argument('--batch-size',
                            help='Batch size to use for training', default=32, type=int)
        parser.add_argument('--epochs',
                            help='Number of epochs to train for', default=1, type=int)

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()


    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

        return model


    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)


    def initTrainDataLoader(self):
        train_ds = LunaDataset(val_stride=10, isValSet_bool=False)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda)
        return train_dl


    def initValDataLoader(self):
        train_ds = LunaDataset(val_stride=10, isValSet_bool=True)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda)
        return train_dl


    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        train_dl = self.initTrainDataLoader()
        val_dl = self.initValDataLoader()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            trnMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)


    def doTraining(self, epoch_ndx: int, train_dl: DataLoader):
        self.model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        batch_iter = enumerateWithEstimate(train_dl, f"E{epoch_ndx} Training", start_ndx = train_dl.num_workers)
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    train_dl.batch_size,
                    trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()
        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')


    def doValidation(self, epoch_ndx: int, val_dl: DataLoader):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            batch_iter = enumerateWithEstimate(val_dl, f"E{epoch_ndx} Validarion", start_ndx = val_dl.num_workers)

            for batch_ndx, batch_tup in batch_iter:
                loss_var = self.computeBatchLoss(
                        batch_ndx,
                        batch_ndx,
                        val_dl.batch_size,
                        valMetrics_g)
            return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx: int, 
                         batch_tup: tuple[torch.tensor,torch.tensor],
                         batch_size: int,
                         metrics_g: torch.tensor):
        input_t, label_t, _series_list, _center_list  = batch_tup
        input_g = input_t.to(self.device)
        label_g = label_t.to(self.device)
        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDS, start_ndx:end_ndx] = loss_g[:, 1].detach()

        return loss_g.mean()


    def logMetrics(self, epoch_ndx: int, mode_str: str,
                   metrics_t: torch.tensor,
                   classificationThreshhold: float = 0.5):
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshhold
        negPred_mask = metrics_t[METRICS_PRED_ND] <= classificationThreshhold
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posPred_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDS].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDS, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDS, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/pos'] = neg_correct / np.float32(neg_counts) * 100
        metrics_dict['correct/neg'] = pos_correct / np.float32(pos_counts) * 100

        log.info(("E{} {:8} {loss/all:.4f} loss, " + "{correct/all:-5.1f}% correct, "
                  ).format(
                     epoch_ndx,
                     mode_str,
                     **metrics_dict
                 ))
        log.info(("E{} {:8} {loss/neg:.4f} loss, " + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count})"
                  ).format(
                     epoch_ndx,
                     mode_str + '_neg',
                     neg_correct=neg_correct,
                     neg_count=neg_count,
                     **metrics_dict,
                 ))
        log.info(("E{} {:8} {loss/pos:.4f} loss, " + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count})"
                  ).format(
                     epoch_ndx,
                     mode_str + '_neg',
                     pos_correct=pos_correct,
                     pos_count=pos_count,
                     **metrics_dict,
                 ))

























