import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from data_modules.classification_datamodule import ClassificationDataModule
from metrics.confusion_matrix import ConfusionMatrix
from registry import Registry
from utils.config_validation import DEFAULT_CLASSIFICATION_LOSS, DEFAULT_OPTIMIZER, Config



class ClassificationTask(pl.LightningModule):

    def __init__(
            self,
            model: dict,
            datamodule: ClassificationDataModule,
            loss: dict = None,
            metrics: list = None,
            optimizer: dict = None,
            scheduler: dict = None,
            visualize_first_batch: bool = False,
            res_dir=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.classes = datamodule.classes
        self.num_classes = len(self.classes)

        if loss is None:
            warnings.warn('Loss is not set in the config. Creating default loss.')
            loss = DEFAULT_CLASSIFICATION_LOSS
        loss = Config(**loss)
        class_weights = datamodule.class_weights if loss.params.get('is_weighted', False) else None
        loss.params.pop('is_weighted', None)
        self.loss = Registry.LOSSES[loss.name](weight=class_weights, **loss.params)

        if optimizer is None:
            warnings.warn('Optimizer is not set in the config. Creating default optimizer.')
            optimizer = DEFAULT_OPTIMIZER
        self.optimizer_config = Config(**optimizer)
        self.scheduler_config = Config(**scheduler) if scheduler else None

        model_config = Config(**model)
        self.model = Registry.MODELS[model_config.name](
            num_classes=self.num_classes, **model_config.params
        ).get_model()

        self.confusion_matrix = ConfusionMatrix(self.num_classes)
        self.metrics = {}
        if metrics is not None:
            for metric in metrics:
                metric = Config(**metric)
                self.metrics[metric.name] = Registry.METRICS[metric.name](**metric.params)

        self.visualize_train = visualize_first_batch
        self.visualize_val = visualize_first_batch
        self.visualize_test = visualize_first_batch

        self.res_dir = res_dir

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, true = batch
        if self.visualize_train:
            visualize_batch(img)
            self.visualize_train = False
        output = self(img)
        loss = self.loss(output, true)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', loss, logger=False, prog_bar=True)
        self.logger.log_metrics({'train_loss': loss}, step=self.current_epoch)

    def on_validation_epoch_start(self):
        self.confusion_matrix.reset()

    def validation_step(self, batch, batch_idx):
        img, target = batch
        if self.visualize_val:
            visualize_batch(img)
            self.visualize_val = False
        output = self(img)
        loss = self.loss(output, target)
        prediction = np.argmax(output.cpu().numpy(), axis=1)
        self.confusion_matrix.update(prediction, target.cpu().numpy())
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', loss, logger=False, prog_bar=True)
        self.logger.log_metrics({'val_loss': loss}, step=self.current_epoch)
        for metric_name, metric in self.metrics.items():
            key = '_'.join(['val', metric_name.lower()])
            value = metric.get_value(self.confusion_matrix.get_value())
            self.log(key, value, logger=False, prog_bar=True)
            self.logger.log_metrics({key: value}, step=self.current_epoch)

    def on_test_epoch_start(self):
        self.confusion_matrix.reset()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        img, target = batch
        if self.visualize_test:
            visualize_batch(img)
            self.visualize_test = False
        prediction = np.argmax(self(img).cpu().numpy(), axis=1)
        self.confusion_matrix.update(prediction, target.cpu().numpy())

    def test_epoch_end(self, outputs) -> None:
        title_string = []
        for metric_name, metric in self.metrics.items():
            key = '_'.join(['test', metric_name.lower()])
            value = metric.get_value(self.confusion_matrix.get_value())
            title_string.append('='.join([str(key), f'{value:.2f}']))
            self.log(key, value, logger=False, prog_bar=True)
            self.logger.log_metrics({key: value}, step=self.current_epoch)
        title_string = ', '.join(title_string)
        save_path = self.res_dir / 'confusion_matrix.png' if self.res_dir else None
        plot_confusion_matrix(
            self.confusion_matrix.get_value(),
            title=title_string,
            categories=self.classes,
            save_path=save_path,
            sort=False,
            show=True
        )

    def configure_optimizers(self):
        config = {}
        opt = Registry.OPTIMIZERS[self.optimizer_config.name](
            self.model.parameters(), **self.optimizer_config.params
        )
        config['optimizer'] = opt
        if self.scheduler_config:
            sch = Registry.SCHEDULERS[self.scheduler_config.name](
                opt, **self.scheduler_config.params
            )
            config['lr_scheduler'] = {
                'scheduler': sch,
                'monitor': 'val_loss'
            }
        return config
