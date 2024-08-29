import logging

import torch
import torchmetrics
import pytorch_lightning as pl
import timm

# from zoobot.shared import schemas

class GenericBaseline(pl.LightningModule):
    """
    All Zoobot models use the lightningmodule API and so share this structure
    super generic, just to outline the structure. nothing specific to dirichlet, gz, etc
    only assumes an encoder and a head
    """

    def __init__(
        self,
        label_cols=['label'],
        architecture_name: str = 'convnext_nano',
        timm_kwargs = {},
        channels: float = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        head_kwargs = {},
        ):

        super().__init__()
        
        self.label_cols = label_cols

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
        x = batch['image']

        # for classification this is simply
        # e.g. {'label': batch['label']} i.e {'label': tensor}
        # for regression this is e.g.
        # {'smooth-or-featured_smooth_fraction': tensor, ...}
        labels = {key: batch[key] for key in self.label_cols}  
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
        loss = self.loss_fn(predictions, labels['label'])  # expects predictions and labels to be cross-entropy ready e.g. one-hot labels
        self.loss_metrics[f'{step_name}/supervised_loss'](loss)
        return loss
    
    def update_other_metrics(self, outputs, step_name):
        self.accuracy_metrics[f'{step_name}/supervised_accuracy'](outputs['prediction'], outputs['label']['label'])

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
        


"""
question_answer_pairs example
{
    'smooth-or-featured-gz2': ['_smooth', '_featured-or-disk', '_artifact'],
    'disk-edge-on-gz2': ['_yes', '_no'],
    'has-spiral-arms-gz2': ['_yes', '_no']
    ...
}
"""


class RegressionBaseline(GenericBaseline):
            
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # use multi-class cross entropy as loss function
        self.question_answer_pairs = kwargs['head_kwargs']['question_answer_pairs']
        self.loss_fn = CustomWeightedMSELoss(self.question_answer_pairs)
        self.answer_fraction_keys = [q + a + '_fraction' for q, a_list in self.question_answer_pairs.items() for a in a_list]
        # logging.info(f'answer keys: {self.answer_fraction_keys}')
        # dependencies = kwargs['head_kwargs']['dependencies']
        # schema = schemas.Schema(question_answer_pairs, dependencies)




    def setup_other_metrics(self):
        # without weighting, supervised loss (above) is the sum of the mean squared errors across all answers

        # also have manually-calculated MSE across all answers?

        regression_metrics = {
            'train/supervised_total_unweighted_mse': torchmetrics.MeanMetric(nan_strategy='ignore'),  # ignore nans via MeanMetric
            'validation/supervised_total_unweighted_mse': torchmetrics.MeanMetric(nan_strategy='ignore'),
            'test/supervised_total_unweighted_mse': torchmetrics.MeanMetric(nan_strategy='ignore')
        }
        question_answer_pairs = self.head_kwargs['question_answer_pairs']
        answer_fraction_keys = [q + a + '_fraction' for q, a_list in question_answer_pairs.items() for a in a_list]
        for answer_col in answer_fraction_keys:
            regression_metrics[f'train/supervised_unweighted_mse_{answer_col}'] = torchmetrics.MeanMetric(nan_strategy='ignore')
            regression_metrics[f'validation/supervised_unweighted_mse_{answer_col}'] = torchmetrics.MeanMetric(nan_strategy='ignore')
            regression_metrics[f'test/supervised_unweighted_mse_{answer_col}'] = torchmetrics.MeanMetric(nan_strategy='ignore')

        self.regression_metrics = torch.nn.ModuleDict(regression_metrics)

        
    def calculate_loss_and_update_loss_metrics(self, predictions, labels, step_name):
        loss = self.loss_fn(predictions, labels)  # expects predictions and labels to be cross-entropy ready e.g. one-hot labels
        self.loss_metrics[f'{step_name}/supervised_loss'](loss)
        return loss
    
    def update_other_metrics(self, outputs, step_name):
        predictions = outputs['prediction']
        targets = torch.stack([outputs['label'][answer_col] for answer_col in self.answer_fraction_keys], dim=1)
        self.regression_metrics[f'{step_name}/supervised_total_unweighted_mse'](
            # mean squared error, summed across answers, then averaged by MeanMetric
            torch.sum(torch.abs(predictions - targets) ** 2, dim=1)  
        )

        for answer_col in self.answer_fraction_keys:
            answer_index = self.answer_fraction_keys.index(answer_col)
            self.regression_metrics[f'{step_name}/supervised_unweighted_mse_{answer_col}'](
                # mean squared error, for one answer, averaged by MeanMetric
                torch.abs(predictions[:, answer_index] - outputs['label'][answer_col]) ** 2
            )


    def log_all_metrics(self, split):
        assert split is not None
        for metric_collection in (self.loss_metrics, self.regression_metrics):
            prog_bar = metric_collection == self.loss_metrics  # don't log all
            for name, metric in metric_collection.items():
                if split in name:
                    # logging.info(name)
                    self.log(name, metric, on_epoch=True, on_step=False, prog_bar=prog_bar, logger=True)


    def create_head(self):
        num_answers = len([a for answer_list in self.head_kwargs['question_answer_pairs'].values() for a in answer_list])
        return torch.nn.Sequential(
            # TODO global pooling layer?
            torch.nn.Dropout(self.head_kwargs['dropout_rate']),
            torch.nn.Linear(self.encoder.num_features, num_answers), 
            torch.nn.Sigmoid()
        )
        

class CustomWeightedMSELoss(torch.nn.Module):
    def __init__(self, question_answer_pairs):
        super().__init__()
        self.question_answer_pairs = question_answer_pairs
        self.question_totals_keys = [question + '_total-votes' for question in self.question_answer_pairs.keys()]
        self.answer_fraction_keys = [q + a + '_fraction' for q, a_list in self.question_answer_pairs.items() for a in a_list]
        logging.info(f'question total keys: {self.question_totals_keys}')
        logging.info(f'answer keys: {self.answer_fraction_keys}')


    def forward(self, inputs, targets):
        # inputs is B x N, where N is the number of answer keys (fractions)
        # targets is dictlike with keys of answer_keys and question_totals_keys, each with values of shape (B)

        total_loss = 0
        
        for question, answers in self.question_answer_pairs.items():
            # question_total_key = question + '_total-votes'
            # question_total = targets[question_total_key]
            for answer in answers:
                answer_key = question + answer + '_fraction'
                # TODO maybe predict dict not tensor?
                answer_index_in_preds = self.answer_fraction_keys.index(answer_key)
                answer_predicted_fraction = inputs[:, answer_index_in_preds]
                answer_true_fraction = targets[answer_key]
                answer_loss = torch.nn.functional.mse_loss(answer_predicted_fraction, answer_true_fraction, reduction='none')
                

                # apply weighting
                # answer_loss = answer_loss * torch.sqrt(question_total)  # upweight the more people answer

                # masked_answer_loss = torch.where(question_total > 0, answer_loss, torch.tensor(0.0))  # only apply loss if labelled
                # total_loss += masked_answer_loss

                # before reduction, loss is (always) summed across answers to give shape (B)
                total_loss += answer_loss

        return torch.mean(total_loss)  # and then with reduction, mean across batch. Never do mean across answers.

        