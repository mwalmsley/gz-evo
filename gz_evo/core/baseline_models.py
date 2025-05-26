import logging

import torch
import torchmetrics
import pytorch_lightning as pl
import timm
import pandas as pd

from gz_evo.core import baseline_datamodules

class GenericBaseline(pl.LightningModule):
    """
    Generic supervised model, based on Zoobot. Intended to be subclassed.
    """

    def __init__(
        self,
        label_cols=['label'],
        # encoder args
        architecture_name: str = 'convnext_nano',
        timm_kwargs = {},
        channels: float = 3,
        # training/finetuning args
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        n_blocks: int = -1,
        lr_decay: float = 0.9,
        from_scratch: bool = False, # override the above
        # args for the head
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
        self.n_blocks = n_blocks
        self.lr_decay = lr_decay
        self.from_scratch = from_scratch

        self.setup_metrics()


    def setup_metrics(self, nan_strategy='error'):  # may sometimes want to ignore nan even in main metrics

        self.loss_metrics = torch.nn.ModuleDict({
            'train/supervised_loss': torchmetrics.MeanMetric(nan_strategy=nan_strategy),
            'validation/supervised_loss': torchmetrics.MeanMetric(nan_strategy=nan_strategy),
            'test/supervised_loss': torchmetrics.MeanMetric(nan_strategy=nan_strategy),
        })

        self.setup_other_metrics()

    # simple option for baselines, designed for training from scratch
    # def configure_optimizers(self):
    #     # reliable simple option for baselines, you may want to subclass this
    #      return torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay
    #     )  

    # temporarily copied from Zoobot - plausible that finetuning for core datasets will be important
    def configure_optimizers(self):  
        """
        This controls which parameters get optimized

        self.head is always optimized, with no learning rate decay
        when self.n_blocks == 0, only self.head is optimized (i.e. frozen* encoder)
        
        for self.encoder, we enumerate the blocks (groups of layers) to potentially finetune
        and then pick the top self.n_blocks to finetune
        
        weight_decay is applied to both the head and (if relevant) the encoder
        learning rate decay is applied to the encoder only: lr x (lr_decay^block_n), ignoring the head (block 0)

        What counts as a "block" is a bit fuzzy, but I generally use the self.encoder.stages from timm. I also count the stem as a block.

        batch norm layers may optionally still have updated statistics using always_train_batchnorm
        """

        lr = self.learning_rate
        params = [{"params": self.head.parameters(), "lr": lr}]

        logging.info(f'Encoder architecture to finetune: {type(self.encoder)}')

        if self.from_scratch:
            logging.warning('self.from_scratch is True, training everything and ignoring all settings')
            params += [{"params": self.encoder.parameters(), "lr": lr}]
            return torch.optim.AdamW(params, weight_decay=self.weight_decay)

        if isinstance(self.encoder, timm.models.EfficientNet): # includes v2
            # TODO for now, these count as separate layers, not ideal
            early_tuneable_layers = [self.encoder.conv_stem, self.encoder.bn1]
            encoder_blocks = list(self.encoder.blocks)
            tuneable_blocks = early_tuneable_layers + encoder_blocks
        elif isinstance(self.encoder, timm.models.ResNet):
            # all timm resnets seem to have this structure
            tuneable_blocks = [
                # similarly
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.layer1,
                self.encoder.layer2,
                self.encoder.layer3,
                self.encoder.layer4
            ]
        elif isinstance(self.encoder, timm.models.MaxxVit):
            tuneable_blocks = [self.encoder.stem] + [stage for stage in self.encoder.stages]
        elif isinstance(self.encoder, timm.models.ConvNeXt):  # stem + 4 blocks, for all sizes
            # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py#L264
            tuneable_blocks = [self.encoder.stem] + [stage for stage in self.encoder.stages]
        elif isinstance(self.encoder, timm.models.VisionTransformer):
            tuneable_blocks = [self.encoder.patch_embed] + [stage for stage in self.encoder.blocks]

        else:
            raise ValueError(f'Encoder architecture not automatically recognised: {type(self.encoder)}')
        
        # interpret -1 as all blocks
        if self.n_blocks == -1:
            logging.info('n_blocks is -1, finetuning all blocks')
            self.n_blocks = len(tuneable_blocks)

        assert self.n_blocks <= len(
            tuneable_blocks
        ), f"Network only has {len(tuneable_blocks)} tuneable blocks, {self.n_blocks} specified for finetuning"

        
        # take n blocks, ordered highest layer to lowest layer
        tuneable_blocks.reverse()
        logging.info('possible blocks to tune: {}'.format(len(tuneable_blocks)))
        # will finetune all params in first N
        logging.info('blocks that will be tuned: {}'.format(self.n_blocks))
        blocks_to_tune = tuneable_blocks[:self.n_blocks]
        # optionally, can finetune batchnorm params in remaining layers
        remaining_blocks = tuneable_blocks[self.n_blocks:]
        logging.info('Remaining blocks: {}'.format(len(remaining_blocks)))
        assert not any([block in remaining_blocks for block in blocks_to_tune]), 'Some blocks are in both tuneable and remaining'

        # Append parameters of layers for finetuning along with decayed learning rate
        for i, block in enumerate(blocks_to_tune):  # _ is the block name e.g. '3'
            params.append({
                    "params": block.parameters(),
                    "lr": lr * (self.lr_decay**i)
                })

        logging.info('param groups: {}'.format(len(params)))

        opt = torch.optim.AdamW(params, weight_decay=self.weight_decay)  # lr included in params dict
        logging.info('Optimizer ready')

        return opt
        
    def forward(self, x):
        assert x.shape[1] < 4  # torchlike BCHW
        x = self.encoder(x)
        return self.head(x)
    
    def make_step(self, batch, step_name):
        x = batch['image']

        labels = {key: batch[key] for key in self.label_cols}  # here is (not needed) label_col filter
        # e.g. {'smooth_yes: 12, 'smooth_no': 8, 'smooth_fraction': 0.6, ...}
        predictions = self(x)
        loss = self.calculate_loss_and_update_loss_metrics(predictions, labels, step_name)      
        outputs = {'loss': loss, 'prediction': predictions, 'label': labels}
        self.update_other_metrics(outputs, step_name=step_name)
        return outputs

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
        self.log_all_metrics(split='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError('predict_step must be subclassed')
    
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



    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds =  self(batch['image'])
        # TODO semi-lazy hardcoding for the moment
        header = baseline_datamodules.LABEL_ORDER
        df = pd.DataFrame(preds.cpu().numpy(), columns=header)
        df['id_str'] = batch['id_str']  # str, no need to cast
        return df


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
        self.answer_keys = [q + a for q, a_list in self.question_answer_pairs.items() for a in a_list]
        self.answer_fraction_keys = [q + a + '_fraction' for q, a_list in self.question_answer_pairs.items() for a in a_list]


    def setup_other_metrics(self):
        # without weighting, supervised loss (above) is the sum of the mean squared errors across all answers
        # also have manually-calculated MSE across all answers?

        # minor duplication: we run this during super().__init__, before own init, 
        # so can't use self.answer_fraction_keys or even self.question_answer_pairs yet, only self.label_cols
        answer_fraction_keys = [col + '_fraction' for col in self.label_cols if col.endswith('_fraction')]

        regression_metrics = {
            'train/supervised_total_unweighted_mse': torchmetrics.MeanMetric(nan_strategy='ignore'),  # ignore nans via MeanMetric
            'validation/supervised_total_unweighted_mse': torchmetrics.MeanMetric(nan_strategy='ignore'),
            'test/supervised_total_unweighted_mse': torchmetrics.MeanMetric(nan_strategy='ignore')
        }
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


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds =  self(batch['image'])
        header = self.answer_keys
        df = pd.DataFrame(preds.cpu().numpy(), columns=header)
        df['id_str'] = batch['id_str']  # str, no need to cast
        return df


    def create_head(self):
        num_features = self.encoder.num_features
        question_answer_pairs = self.head_kwargs['question_answer_pairs']
        # return torch.nn.Sequential(
        #     # TODO global pooling layer?
        #     torch.nn.Dropout(self.head_kwargs['dropout_rate']),
        #     torch.nn.Linear(self.encoder.num_features, num_answers), 
        #     torch.nn.Sigmoid()
        # )
        return SoftmaxHeadPerAnswer(num_features, question_answer_pairs, dropout_rate=self.head_kwargs['dropout_rate'])

class SoftmaxHeadPerAnswer(torch.nn.Module):

    def __init__(self, num_features: int, question_answer_pairs: dict, dropout_rate: float = 0.5):  # dropout newly added
        super().__init__()

        self.question_answer_pairs = question_answer_pairs
        # Q heads, each one making softmax predictions for the answers to a question
        self.heads = torch.nn.ModuleDict({
            question: SoftmaxHead(num_features, len(answers), dropout_rate) for question, answers in question_answer_pairs.items()
        })

    def forward(self, x):
        preds = [head(x) for head in self.heads.values()]
        return torch.cat(preds, dim=1)
    

def SoftmaxHead(num_features, num_answers, dropout_rate):
    return torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),  # newly added
            torch.nn.Linear(num_features, num_answers), 
            torch.nn.Softmax(dim=1)  # this is our estimated probability for each answer (for multinomial loss)
        )

class CustomWeightedMSELoss(torch.nn.Module):
    def __init__(self, question_answer_pairs):
        super().__init__()
        self.question_answer_pairs = question_answer_pairs
        # looks similar to the RegressionBaseline init, but this is a different self
        self.answer_keys = [q + a for q, a_list in self.question_answer_pairs.items() for a in a_list]
        logging.info(f'answer keys: {self.answer_keys}')


    def forward(self, inputs, targets):
        # inputs, prediction vector, is B x N, where N is the number of answer keys (fractions). Might change to dictlike.
        # targets, labels from datamodule, is dictlike with keys of answer_keys, each with values of shape (B)

        loss = 0

        logging.info(targets.keys())  # should be answer_keys (and answer_fraction_keys, but ignored here)
                
        for question, answers in self.question_answer_pairs.items():

            q_answer_keys = [question + answer for answer in answers]

            counts = torch.stack([targets[key] for key in q_answer_keys], dim=1).int()

            # work out the answer indices for the prediction vector
            answer_indices = [self.answer_keys.index(key) for key in q_answer_keys]
            predicted_probs = inputs[:, answer_indices]

            # negative log likelihood of observed counts using multinomial p predicted by model
            # this is a simplified version of the Dirichlet-Multinomial loss from Zoobot etc, where the model predicts concentrations
            # here, the model predicts the probabilities of each answer without a dirichlet distribution, like W+2022

            # total counts is implicit from counts, nice one torch :)
            # question_loss = -1 * torch.distributions.multinomial.Multinomial(probs=predicted_probs).log_prob(counts)
            question_loss = -1 * get_multinomial_log_prob(predicted_probs, counts)  # DIY version, I forget why...
            loss += question_loss
                
        # treating all answers equally, take a nanmean of everything
        return torch.nanmean(loss)  # and then with reduction, mean across batch. Never do mean across answers.



def get_multinomial_log_prob(probs, counts):
        logits = torch.distributions.utils.probs_to_logits(probs)
        log_factorial_n = torch.lgamma(counts.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(counts + 1).sum(-1)
        logits[(counts == 0) & (logits == -torch.inf)] = 0
        log_powers = (logits * counts).sum(-1)
        return log_factorial_n - log_factorial_xs + log_powers
