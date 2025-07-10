from foundation.models.base_hybrid import BaseHybridLearner
from huggingface_hub import hf_hub_download

if __name__ == "__main__":

    repo_id = 'mwalmsley/wip-hybrid-encoder-n0jvg4dc'
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="last.ckpt", repo_type="model")
    model = BaseHybridLearner.load_from_checkpoint(ckpt_path)
    print(model)