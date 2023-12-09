#!/usr/bin/env python3
import gc
import logging
import os
import random
import sys
import time
import warnings

import lightning.pytorch as pl
import pandas as pd
import rich
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from rich import print

sys.path.append("../main")
from bottle import RNA_Lightning
from data import RNA_DM
from utils import ExCB, TBLogger, grid_search, kfold, mutate_map

if __name__ == "__main__":
    debug = 0
    n_workers = 8
    pred = False
    ckpt = True
    seed = 420    
    
    # early_stop = 3 # origin 10, pseudo 3
    early_stop = 7 # origin 10, pseudo 3    
    n_trials = 0    
    n_folds = 5
    
    # run_folds = [0]        
    # run_folds = [0, 1, 2, 3, 4]
    run_folds = [0, 1, 2, 3, 4]
        
    log_dir = "e34"
    pred_dir = "subs"
    metric = "loss/V"

    # if infer, set pseudo_labeling, pretraining: False.
    pseudo_labeling = False # pseudo labeling
    pretraining = True
    
    hp_conf = {
        "n_epochs": 200,
        # "lr": 2e-3, # origin 2e-3 (first)        
        # "lr": 2e-4, # origin 2e-3, using pseudo (bs 100):2e-4
        "lr": 2e-5, # origin 2e-3, pseudo: 2e-4, pseudo 후에: 2e-5 (는 2 epoch에서 val 튐) (third)
                
        "lr_warmup": 0.015, # origin 0.015        
        "wt_decay": 0.05, # origin 1e-1
        "grad_clip": 5,
        # bs 설정팁: 만약, single bs 500이라면, multi-gpu가 2개라면, 500 / 2 = 250으로 설정해야함.
                
        "batch_size": 250, # origin 250 (2 gpu: 250 * 2 = 500), pseudo: 100            
        "n_grad_accum": 1, # origin 1, pseudo: 2
        
        "n_mem": 0,        
        "sn_min": 0.6,
        # "sn_min": 0.2,        
        "aux_loop": [None, "eterna"][0],  # "contra"],
        "aux_struct": [None, "eterna"][0],  # "contra"],
        "aux_scale": 0.1,        
        "aux_smooth": 0.1, # set 0.5 : cv 0.12932 -> cv 0.12917
        "aug_reverse": True,
        "emb_grad_frac": 1,
        "norm_layout": "dual",
        "pos_bias_heads": 6,
        "pos_bias_params": (32, 128),
        "pos_rope": False,
        "pos_sine": False,
        "norm_rms": True,
        "norm_lax": False,
        "qkv_bias": False,
        "ffn_bias": False,
        "ffn_multi": 4,
        "n_layers": 12, # origin 12
        "n_heads": 6,
        "d_heads": 48,
        "p_dropout": 0.05,
        "att_fn": ["sdpa", "xmea"][1],
        "n_folds": n_folds,
        "seed": seed,
        "note": "",
        "n_layers_lstm":1,
        'kernel_size_gc': 7,
        "use_lenmatched_batch": True,           
    }
    hp_skips = [
        # {"aux_loop": 'eterna', "aux_struct": None},
        # {"aux_loop": None, "aux_struct": 'eterna'},
    ]
    df_train = "../data/train_data_processed_ALL.parquet"
    df_infer = "../data/test_sequences_processed_ALL.parquet"
    df_pseudo = "../data/submission_for_pseudo_v5_LB013884.parquet" # LB 0.13885
    
    # # 모델 체크포인트 경로들을 리스트로 지정합니다.
    # ckpt_paths  = [ # by_roger_v2 models (train -> train + pseudo -> ...
    #     "pretraining_models/fold0/epoch=35-step=18612.ckpt",
    #     "pretraining_models/fold1/epoch=24-step=12950.ckpt",
    #     "pretraining_models/fold2/epoch=38-step=20163.ckpt",
    #     "pretraining_models/fold3/epoch=40-step=21197.ckpt",
    #     "pretraining_models/fold4/epoch=32-step=17094.ckpt",
    # ]
    
    ckpt_paths  = [ # by_roger_v2 models (pseudo pretraining models)
        "pretraining_pseudo_models_v2/version_20/epoch=32-step=58872.ckpt",
        "pretraining_pseudo_models_v2/version_21/epoch=34-step=62440.ckpt",
        "pretraining_pseudo_models_v2/version_22/epoch=19-step=35680.ckpt",
        "pretraining_pseudo_models_v2/version_23/epoch=12-step=23192.ckpt",
        "pretraining_pseudo_models_v2/version_24/epoch=16-step=30328.ckpt",
    ]
    
        
    try:
        with rich.get_console().status("Reticulating Splines"):
            if debug:
                run_folds, ckpt = [0], False
            else:
                warnings.filterwarnings("ignore")
                for n in logging.root.manager.loggerDict:
                    logging.getLogger(n).setLevel(logging.WARN)
            torch.set_float32_matmul_precision("medium")
            torch.manual_seed(seed)
            random.seed(seed)
            os.makedirs(log_dir, exist_ok=True)
            df_infer = pd.read_parquet(df_infer) if (pred|pseudo_labeling) else None
            df_train = pd.read_parquet(df_train)
            df_pseudo = pd.read_parquet(df_pseudo)  
            
            pred_cols = ["reactivity_DMS_MaP", "reactivity_2A3_MaP"]
            folds = kfold(df_train, n_folds, seed)
            folds_for_pseudo = kfold(df_infer, n_folds, seed)
            
            trials = grid_search(hp_conf, hp_skips)
            n_trials = len(trials) if not n_trials else n_trials
            n_trials = len(trials) if len(trials) < n_trials else n_trials
        print(f"Log: {log_dir} | EStop: {early_stop} | Ckpt: {ckpt} | Pred: {pred}")
        for i, hp in enumerate(trials[:n_trials]):
            for j, f in enumerate(run_folds):
                print(f"Trial {i + 1}/{n_trials} Fold {j + 1}/{len(run_folds)} ({f})")
                hp.fold = f
                tbl = TBLogger(os.getcwd(), log_dir, default_hp_metric=False)
                cb = [RichProgressBar(), ExCB()]                
                cb += [ModelCheckpoint(tbl.log_dir, None, metric)] if ckpt else []                
                cb += [EarlyStopping(metric, 0, early_stop)] if early_stop else []
                                
                dm = RNA_DM(hp, n_workers, df_infer, df_train, folds[f], pseudo_labeling, df_pseudo, folds_for_pseudo[f]) # add pseudo
                
                if pretraining:
                    print('Pretraining...')
                    model = RNA_Lightning.load_from_checkpoint(ckpt_paths[f], hp=hp, strict=False)
                else:
                    print('No pretraining...')
                    model = RNA_Lightning(hp)

                
                trainer = pl.Trainer(
                    precision="16-mixed",
                    accelerator="gpu",
                    benchmark=True,
                    max_epochs=hp.n_epochs,
                    accumulate_grad_batches=hp.n_grad_accum,
                    gradient_clip_val=hp.grad_clip,
                    fast_dev_run=debug,
                    num_sanity_val_steps=0,
                    enable_model_summary=False,
                    logger=tbl,
                    callbacks=cb,
                    devices = [0, 1],                    
                )
                gc.collect()
                try:
                    trainer.fit(model, datamodule=dm)
                except KeyboardInterrupt:
                    print("Fit Interrupted")
                    if i + 1 < n_trials:
                        with rich.get_console().status("Quit?") as s:
                            for k in range(3):
                                s.update(f"Quit? {3-k}")
                                time.sleep(1)
                    continue
                if pred:
                    try:
                        cp = None if debug else "best"
                        preds = trainer.predict(model, datamodule=dm, ckpt_path=cp)
                    except KeyboardInterrupt:
                        print("Prediction Interrupted")
                        continue
                    with rich.get_console().status("Processing Submission"):
                        preds = torch.concat(preds).float()
                        preds = pd.DataFrame(preds, columns=pred_cols)
                        preds.insert(0, "id", preds.index)
                        if not debug:
                            os.makedirs(pred_dir, exist_ok=True)
                            fn = f"{pred_dir}/{log_dir}v{tbl.version:02}"
                            preds.to_parquet(f"{fn}.parquet", index=False)
                            mutate_map(preds, f"{fn}.png")
                        del preds
    except KeyboardInterrupt:
        print("Goodbye")
        sys.exit()
