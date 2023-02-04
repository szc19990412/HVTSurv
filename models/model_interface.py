import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import NLLSurvLoss, CrossEntropySurvLoss, cox_log_rank, _predictions_to_pycox, CoxSurvLoss
from sksurv.metrics import concordance_index_censored
# from pycox.evaluation import EvalSurv
from sklearn.utils import resample


#---->
import torch
import torch.nn as nn
import torch.nn.functional as F

#---->
import pytorch_lightning as pl


class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = NLLSurvLoss(loss.alpha_surv)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0

    #---->Remove v_num from the progress bar
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']
        Y_hat = results_dict['Y_hat']

        #---->calculate loss
        loss = self.loss(hazards=hazards, S=S, Y=label.long(), c=c)

        return {'loss': loss} 

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']
        Y_hat = results_dict['Y_hat']

        loss = self.loss(hazards=hazards, S=S, Y=label.long(), c=c)
        risk = -torch.sum(S, dim=1).cpu().item()

        return {'loss' : loss.item(), 'risk' : risk, 'censorship' : c.item(), 'event_time' : event_time.item()}


    def validation_epoch_end(self, val_step_outputs):
        all_val_loss = np.stack([x['loss'] for x in val_step_outputs])
        all_risk_scores = np.stack([x['risk'] for x in val_step_outputs])
        all_censorships = np.stack([x['censorship'] for x in val_step_outputs])
        all_event_times = np.stack([x['event_time'] for x in val_step_outputs])
        
        #---->Calculation of metrics
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        pvalue_pred = cox_log_rank(all_risk_scores, (1-all_censorships), all_event_times)
        self.log('val_loss', np.mean(all_val_loss), prog_bar=True, on_epoch=True, logger=True)
        self.log('c_index', c_index, prog_bar=True, on_epoch=True, logger=True)
        self.log('p_value', pvalue_pred, prog_bar=True, on_epoch=True, logger=True)
    


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)

        return [optimizer]

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']
        Y_hat = results_dict['Y_hat']

        risk = -torch.sum(S, dim=1).cpu().item()

        return {'risk' : risk, 'censorship' : c.item(), 'event_time' : event_time.item(), 'S' : S.detach().cpu().numpy()}

    def test_epoch_end(self, output_results):
        all_risk_scores = np.stack([x['risk'] for x in output_results])
        all_censorships = np.stack([x['censorship'] for x in output_results])
        all_event_times = np.stack([x['event_time'] for x in output_results])
        all_properties = np.stack([x['S'] for x in output_results])
        
        #---->Calculation of metrics
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        pvalue_pred = cox_log_rank(all_risk_scores, (1-all_censorships), all_event_times)
        print(f'c_index={c_index}, p_value={pvalue_pred}')

        #---->bootstrap
        n = 1000
        skipped = 0
        boot_c_index = []
        from tqdm import tqdm
        for i in tqdm(range(n)):
            boot_ids = resample(np.arange(len(all_risk_scores)), replace=True)
            risk_scores = all_risk_scores[boot_ids]
            censorships = all_censorships[boot_ids]
            event_times = all_event_times[boot_ids]
            properties = all_properties[boot_ids]
            # When running samples with small number of patients (e.g. some
            # individual cancer types) sometimes there are no admissible pairs
            # to compute the C-index (or other metrics).
            # In those cases continue and print a warning at the end
            try:
                c_index_buff = concordance_index_censored((1-censorships).astype(bool), event_times, risk_scores, tied_tol=1e-08)[0]
                #---->save
                boot_c_index.append(c_index_buff)

            except ZeroDivisionError as error:                                  
                err = error                                                     
                skipped += 1                                                    
                continue  
        if skipped > 0:
            warnings.warn(
                f'Skipped {skipped} bootstraps ({err}).')

        #---->Calculate the gap between the bootstraps and the actual metric
        c_index_differences = sorted([x - c_index for x in boot_c_index])
        c_index_percent = np.percentile(c_index_differences, [2.5, 97.5])
        c_index_low, c_index_high = tuple(round(c_index + x, 4)
                              for x in [c_index_percent[0], c_index_percent[1]])

        #---->Save all metrics as csv
        dict = {'c_index':c_index, 'c_index_high':c_index_high, 'c_index_low':c_index_low,
                'p_value':pvalue_pred}
        result = pd.DataFrame(list(dict.items()))
        result.to_csv(self.log_path / 'result.csv')

        #---->Save the three indicators of all_risk_scores, all_censorships, and all_event_times, and ask for all folds
        np.savez(self.log_path / 'all_risk_scores.npz', all_risk_scores)
        np.savez(self.log_path / 'all_censorships.npz', all_censorships)
        np.savez(self.log_path / 'all_event_times.npz', all_event_times)

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)