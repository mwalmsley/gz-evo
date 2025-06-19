import logging

from gz_evo.core import baseline_training, baseline_models
from gz_evo.core.multinomial.train import set_up_task_data


def evaluate():

    # dataset_name = 'gz2'
    # architecture_name = 'convnext_atto'
    # checkpoint_dir = '/home/walml/repos/gz-evo/results/baselines/regression/convnext_atto_534895718'
    # evaluate_single_model(checkpoint_dir, architecture_name, dataset_name)

    # debug_dir = '/home/walml/repos/gz-evo/results/baselines/regression/'
    # beluga_dir = '/project/def-bovy/walml/repos/gz-evo/results/baselines/regression/'
    results_dir = '/share/nas2/walml/repos/gz-evo/results/baselines/regression/'

    for dataset_name, architecture_name, checkpoint_dir in [
        # ('gz_evo', 'resnet50',  results_dir + 'resnet50_534895718_1746547649'),  # technically still training
        # ('gz_evo', 'convnext_nano',  results_dir + 'convnext_nano_534895718_1746542691'),
        # ('gz_evo', 'convnext_base',  results_dir + 'convnext_base_534895718_1746547550'),
        # ('gz_evo', 'tf_efficientnetv2_s',  results_dir + 'tf_efficientnetv2_s_534895718_1746547782')
        # ('gz_evo', 'maxvit_tiny',  results_dir + 'maxvit_tiny_534895718_1746547757'),
        ('gz_evo', 'convnext_large', results_dir + 'convnext_large_534895718_1746548055'),
        ('gz_evo', 'maxvit_base', results_dir + 'maxvit_base_534895718_1746561752'),
        # ('gz_evo', 'maxvit_large', results_dir + 'maxvit_large_534895718_1746561915'),
        ('gz_evo', 'tf_efficientnetv2_l', results_dir + 'tf_efficientnetv2_l_534895718_1746653208'),
        ('gz_evo', 'tf_efficientnetv2_m', results_dir + 'tf_efficientnetv2_m_534895718_1746653116'),
    ]:

        logging.info(f"Evaluating {dataset_name} {architecture_name} {checkpoint_dir}")
        cfg = baseline_training.get_config(architecture_name, dataset_name, save_dir='foobar')

        baseline_training.evaluate_single_model(
            checkpoint_dir, cfg, model_lightning_class=baseline_models.RegressionBaseline, task_data_func=set_up_task_data
            )

    logging.info('Test predictions complete for all models. Exiting.')
    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    evaluate()
    
    """
    rsync -avz walml@beluga.alliancecan.ca:"/project/def-bovy/walml/repos/gz-evo/results/baselines/regression" --exclude="*.ckpt" results/baselines
    """