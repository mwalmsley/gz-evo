import logging

import pytorch_lightning as pl

from gz_evo.core import baseline_training, baseline_models 
from gz_evo.core.classification.classification_baseline import set_up_task_data


def evaluate():

    # debug_dir = '/home/walml/repos/gz-evo/results/baselines/classification/'
    # beluga_dir = '/project/def-bovy/walml/repos/gz-evo/results/baselines/classification/'
    results_dir = '/share/nas2/walml/repos/gz-evo/results/baselines/classification/'

    for dataset_name, architecture_name, checkpoint_dir in [
        # ('gz_evo', 'tf_efficientnetv2_s',  results_dir + 'tf_efficientnetv2_s_534895718'),
        # ('gz_evo', 'maxvit_base',  results_dir + 'maxvit_base_534895718'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_534895718'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_finetune_494155588'),
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_534895718'),  # not great, might need redo. Redone and still not great.
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_finetune_494155588'),  # failed
        
        # ('gz_evo', 'maxvit_tiny',  results_dir + 'maxvit_tiny_rw_224_534895718'),
        # ('gz_evo', 'resnet50',  results_dir + 'resnet50_534895718'),
        # ('gz_evo', 'convnextv2_base.fcmae_ft_in22k_in1k',  results_dir + 'convnextv2_base.fcmae_ft_in22k_in1k_534895718'),
        
        ('gz_evo', 'convnext_large', results_dir + 'convnext_large_534895718'),
        ('gz_evo', 'maxvit_large', results_dir + 'maxvit_large_534895718'),
        ('gz_evo', 'maxvit_small', results_dir + 'maxvit_small_3966912'),
        ('gz_evo', 'tf_efficientnetv2_l', results_dir + 'tf_efficientnetv2_l_534895718'),
        ('gz_evo', 'tf_efficientnetv2_m', results_dir + 'tf_efficientnetv2_m_534895718'),

    ]:
        logging.info(f"Evaluating {dataset_name} {architecture_name} {checkpoint_dir}")
        cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir='foobar') # save_dir is not used

        # overrride batch size
        # cfg.batch_size = cfg.batch_size // 2  
        # not sure why but predictions don't fit with full batch size,
        # even though neither training nor predictions are distributed
        # it seems to be that the model doesn't fit in memory when second gpu is in use, even though it is not used
        try:
            baseline_training.evaluate_single_model(
                checkpoint_dir, cfg, model_lightning_class=baseline_models.ClassificationBaseline, task_data_func=set_up_task_data
            )
        except Exception as e:
            logging.error(f"Failed to evaluate {dataset_name} {architecture_name} {checkpoint_dir}")
            logging.error(e)

    logging.info('Test predictions complete for all models. Exiting.')
    
    """
    rsync -avz walml@beluga.alliancecan.ca:"/project/def-bovy/walml/repos/gz-evo/results/baselines/classification" --exclude="*.ckpt" results/baselines
    rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' --exclude="*.ckpt" walml@galahad.ast.man.ac.uk:"/share/nas2/walml/repos/gz-evo/results/baselines/classification" --exclude="*.ckpt" results/baselines
    """



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting classification baseline")

    seed = 42
    pl.seed_everything(seed)

    evaluate()
