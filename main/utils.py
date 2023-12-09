import os
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedGroupKFold
from apex.normalization import FusedLayerNorm, FusedRMSNorm
import numpy as np
import pandas as pd


def mutate_map(df: pd.DataFrame, fname: str):
    font_size = 6
    id1 = 269545321
    id2 = 269724007
    shape = (391, 457)
    pred_DMS = df[id1 : id2 + 1]["reactivity_DMS_MaP"].to_numpy().reshape(*shape)
    pred_2A3 = df[id1 : id2 + 1]["reactivity_2A3_MaP"].to_numpy().reshape(*shape)
    fig = plt.figure()
    plt.subplot(121)
    plt.title(f"reactivity_DMS_MaP", fontsize=font_size)
    plt.imshow(pred_DMS, vmin=0, vmax=1, cmap="gray_r")
    plt.subplot(122)
    plt.title(f"reactivity_2A3_MaP", fontsize=font_size)
    plt.imshow(pred_2A3, vmin=0, vmax=1, cmap="gray_r")
    plt.tight_layout()
    plt.savefig(fname, dpi=500)
    plt.clf()
    plt.close()


def sort_weight_decay_params(model, dbg=False):
    # https://github.com/karpathy/minGPT
    decay, no_decay = set(), set()
    decay_whitelist = (
        nn.Linear,
        nn.MultiheadAttention,
    )
    decay_blacklist = (
        nn.Embedding,
        nn.LayerNorm,
        FusedLayerNorm,
        FusedRMSNorm,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith(("bias", "freqs", "scale", "mem")):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, decay_whitelist):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, decay_blacklist):
                no_decay.add(fpn)
    pd = {pn: p for pn, p in model.named_parameters()}
    if dbg:
        print(decay & no_decay)
        print(pd.keys() - (decay | no_decay))

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full parameter name
            # BatchNorm 계열 파라미터는 감쇠를 적용하지 않음
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                no_decay.add(fpn)
            # Embedding 계열 파라미터는 감쇠를 적용하지 않을 수 있음
            elif isinstance(m, nn.Embedding):
                no_decay.add(fpn)
            # 기존 로직을 유지함
            elif pn.endswith(("bias", "LayerNorm.weight")):
                no_decay.add(fpn)
            # RNN 계열 파라미터 처리
            elif isinstance(m, (nn.RNNBase, nn.LSTM, nn.GRU)):
                if "weight" in pn:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            # 1D & 2D Convolution 계열 파라미터 처리
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                if "weight" in pn:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

    
    assert len(decay & no_decay) == 0
    assert len(pd.keys() - (decay | no_decay)) == 0
    decay, no_decay = sorted(list(decay)), sorted(list(no_decay))
    decay = [pd[pn] for pn in sorted(list(decay))]
    no_decay = [pd[pn] for pn in no_decay]
    return decay, no_decay


def grid_search(hp: dict, hp_skips: list) -> list:
    def search(hp: dict) -> list:
        kl = [k for k, v in hp.items() if type(v) == list]
        if not kl:
            args = Namespace()
            for k, v in hp.items():
                setattr(args, k, v)
            return [args]
        out = []
        for item in hp[kl[0]]:
            hp_ = hp.copy()
            hp_[kl[0]] = item
            out += search(hp_)
        return out

    def skip(hp: Namespace, hp_skips: list) -> bool:
        if not hp_skips:
            return False
        for hp_skip in hp_skips:
            for k, v in hp_skip.items():
                v = [v] if not isinstance(v, list) else v
                if not getattr(hp, k) in v:
                    match = False
                    break
                match = True
            if match:
                return True
        return False

    return [_ for _ in search(hp) if not skip(_, hp_skips)]


class ExCB(Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            raise exception


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None) -> None:
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)

    @property
    def log_dir(self) -> str:
        version = (
            self.version
            if isinstance(self.version, str)
            else f"version_{self.version:02}"
        )
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir


def kfold(df: pd.DataFrame, n_folds: int, seed: int, cache_dir: str = "cache"):
    fname = f"{cache_dir}/{n_folds}_{seed}.parquet"
    try:
        return pd.read_parquet(fname).values.tolist()
    except:
        pass
    folds = StratifiedGroupKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    folds = list(folds.split(df, df.seq.apply(len), df.seq_id))
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame(folds).to_parquet(fname)
    return folds

def get_nan_arr(L):
    nan_arr = np.zeros(L)
    nan_arr[:] = float('nan')
    return nan_arr


def merge_train_infer(df_train, df_infer, df_pseudo):
    df_infer['SN_2A3'] = 0.99
    df_infer['SN_DMS'] = 0.99
    df_infer.rename(columns = {'sequence_id': 'seq_id', 'sequence': 'seq'}, inplace = True)
    len_arr = np.array((df_infer.seq).apply(len))
    react_DMS = []
    react_2A3 = []    
    
    for i in range(0, len(df_infer)):
        id_min = df_infer.loc[i, 'id_min']
        id_max = df_infer.loc[i, 'id_max']
        
        seq_len_nan_DMS = get_nan_arr(len_arr[i])
        seq_len_nan_2A3 = get_nan_arr(len_arr[i])

        reactivity_DMS_MaP_values  = df_pseudo.loc[id_min:id_max, 'reactivity_DMS_MaP'].tolist()
        reactivity_2A3_MaP_values  = df_pseudo.loc[id_min:id_max, 'reactivity_2A3_MaP'].tolist()
        
        seq_len_nan_DMS[:len(reactivity_DMS_MaP_values)] = reactivity_DMS_MaP_values
        seq_len_nan_2A3[:len(reactivity_2A3_MaP_values)] = reactivity_2A3_MaP_values
        
        react_DMS.append(seq_len_nan_DMS)
        react_2A3.append(seq_len_nan_2A3)
                                    
    df_infer['react_DMS'] = react_DMS
    df_infer['react_2A3'] = react_2A3
    
    df_train = pd.concat((df_train, df_infer))
    return df_train

