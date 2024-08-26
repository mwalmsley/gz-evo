
import torch
import torchmetrics
import pytorch_lightning as pl
import timm

class GenericBaseline(pl.LightningModule):
    """
    All Zoobot models use the lightningmodule API and so share this structure
    super generic, just to outline the structure. nothing specific to dirichlet, gz, etc
    only assumes an encoder and a head
    """

    def __init__(
        self,
        architecture_name: str = 'convnext_nano',
        timm_kwargs = {},
        channels: float = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        head_kwargs = {},
        ):

        super().__init__()
        
        self.head_kwargs = head_kwargs
        self.timm_kwargs = timm_kwargs
        self.save_hyperparameters()  # saves all args by default
    
        self.encoder = timm.create_model(architecture_name, in_chans=channels, num_classes=0, **timm_kwargs)
        self.head = self.create_head()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.setup_metrics()



    def setup_metrics(self, nan_strategy='error'):  # may sometimes want to ignore nan even in main metrics

        self.loss_metrics = torch.nn.ModuleDict({
            'train/supervised_loss': torchmetrics.MeanMetric(nan_strategy=nan_strategy),
            'validation/supervised_loss': torchmetrics.MeanMetric(nan_strategy=nan_strategy),
            'test/supervised_loss': torchmetrics.MeanMetric(nan_strategy=nan_strategy),
        })

        self.setup_other_metrics()
        
        # TODO could add per-campaign metrics automatically?

    def forward(self, x):
        assert x.shape[1] < 4  # torchlike BCHW
        x = self.encoder(x)
        return self.head(x)
    
    def make_step(self, batch, step_name):
        x, labels = batch['image'], batch['label']  # batch is dict with many keys
        predictions = self(x)
        loss = self.calculate_loss_and_update_loss_metrics(predictions, labels, step_name)      
        outputs = {'loss': loss, 'prediction': predictions, 'label': labels}
        self.update_other_metrics(outputs, step_name=step_name)
        return outputs

    def configure_optimizers(self):
        # reliable simple option for baselines, you may want to subclass this
         return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )  

    def training_step(self, batch, batch_idx):
        return self.make_step(batch, step_name='train')

    def validation_step(self, batch, batch_idx):
        return self.make_step(batch, step_name='validation')
    
    def test_step(self, batch, batch_idx):
        return self.make_step(batch, step_name='test')

    def on_train_epoch_end(self) -> None:
        # called *after* on_validation_epoch_end, confusingly
        # do NOT log_all_metrics here. 
        # logging a metric resets it, and on_validation_epoch_end just logged and reset everything, so you will only log nans
        self.log_all_metrics(split='train')

    def on_validation_epoch_end(self) -> None:
        self.log_all_metrics(split='validation')

    def on_test_epoch_end(self) -> None:
        # logging.info('start test epoch end')
        self.log_all_metrics(split='test')
        # logging.info('end test epoch end')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # I can't work out how to get webdataset to return a single item im, not a tuple (im,).
        # this is fine for training but annoying for predict
        # help welcome. meanwhile, this works around it
        if isinstance(batch, list) and len(batch) == 1:
            return self(batch[0])
        # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#inference
        # this calls forward, while avoiding the need for e.g. model.eval(), torch.no_grad()
        # x, y = batch  # would be usual format, but here, batch does not include labels
        return self(batch)
    
    # subclassed below for the various tasks, or extend yourself

    def calculate_loss_and_update_loss_metrics(self, predictions, labels, step_name):
        raise NotImplementedError('Must be subclassed')
    
    def setup_other_metrics(self):
        raise NotImplementedError('Must be subclassed')
    
    def update_other_metrics(self, outputs, step_name):
        raise NotImplementedError('Must be subclassed')
    
    def log_all_metrics(self, split):
        raise NotImplementedError('Must be subclassed')
    
    def create_head(self):
        raise NotImplementedError('Must be subclassed')


class ClassificationBaseline(GenericBaseline):
            
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # use multi-class cross entropy as loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_classes = kwargs['head_kwargs']['num_classes']

    def setup_other_metrics(self):
        self.accuracy_metrics = torch.nn.ModuleDict({
            'train/supervised_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=self.head_kwargs['num_classes']),
            'validation/supervised_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=self.head_kwargs['num_classes']),
            'test/supervised_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=self.head_kwargs['num_classes'])
        })

        
    def calculate_loss_and_update_loss_metrics(self, predictions, labels, step_name):
        loss = self.loss_fn(predictions, labels)  # expects predictions and labels to be cross-entropy ready e.g. one-hot labels
        self.loss_metrics[f'{step_name}/supervised_loss'](loss)
        return loss
    
    def update_other_metrics(self, outputs, step_name):
        self.accuracy_metrics[f'{step_name}/supervised_accuracy'](outputs['prediction'], outputs['label'])

    def log_all_metrics(self, split):
        assert split is not None
        for metric_collection in (self.loss_metrics, self.accuracy_metrics):
            # prog_bar = metric_collection == self.loss_metrics
            prog_bar = True
            for name, metric in metric_collection.items():
                if split in name:
                    # logging.info(name)
                    self.log(name, metric, on_epoch=True, on_step=False, prog_bar=prog_bar, logger=True)


    def create_head(self):
        return torch.nn.Sequential(
            # TODO global pooling layer?
            torch.nn.Dropout(self.head_kwargs['dropout_rate']),
            torch.nn.Linear(self.encoder.num_features, self.head_kwargs['num_classes']), 
            torch.nn.Softmax(dim=-1)
        )
        