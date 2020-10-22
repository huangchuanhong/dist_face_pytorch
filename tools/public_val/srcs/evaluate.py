import torch
import numpy as np
from .utils import l2_norm, hflip_batch
from .verifacation import evaluate

def trans_to_our_normalization(batch):
    batch *= 2
    batch += 0.5
    batch -= torch.tensor([0.482352, 0.45490, 0.40392]).view((-1, 3, 1, 1))
    batch /= torch.tensor([0.392157, 0.392157, 0.392157]).view((-1, 3, 1, 1))
    return batch

def val(conf, model, carray, issame, nrof_folds=5, tta=False, our_normalization=False):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), conf.model.top_model.feature_dim])
    with torch.no_grad():
        while idx + conf.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + conf.batch_size])
            if our_normalization:
                batch = trans_to_our_normalization(batch)
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to('cuda')) + model(fliped.to('cuda'))
                embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch.cpu())
            else:
                output = model(batch.to('cuda')).cpu()
                embeddings[idx:idx + conf.batch_size] = l2_norm(output)
            idx += conf.batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if our_normalization:
                batch = trans_to_our_normalization(batch)
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to('cuda')) + model(fliped.to('cuda'))
                embeddings[idx:] = l2_norm(emb_batch.cpu())
            else: 
                output = self.model.base_model(batch.to('cuda')).cpu()
                embeddings[idx:] = l2_norm(output)
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        return accuracy.mean(), best_thresholds.mean(), tpr, fpr
