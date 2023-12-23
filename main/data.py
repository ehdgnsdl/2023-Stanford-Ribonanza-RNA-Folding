import itertools
import os
import random
from argparse import Namespace

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import merge_train_infer

class RNA_DS(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sn_min: int | float = 0,
        aux_loop: str | None = None,
        aux_struct: str | None = None,
        aug_reverse: bool = False,
    ):
        if sn_min > 0:
            if isinstance(sn_min, int):
                df = df.loc[(df.SN_DMS >= sn_min) & (df.SN_2A3 >= sn_min)]
            else:
                df = df.loc[df.SN_DMS + df.SN_2A3 >= sn_min * 2]
            df = df.reset_index(drop=True)
        
        # print(f"sn_min:{sn_min} len:{len(df)}")
        self.df = df
        self.aux_loop = aux_loop
        self.aux_struct = aux_struct
        self.aug_reverse = aug_reverse
        self.infer = sn_min < 0
        self.len_max = (df.sequence if self.infer else df.seq).apply(len).max()
        self.kmap_seq = {x: i for i, x in enumerate("_AUGC")}
        self.kmap_loop = {x: i for i, x in enumerate("_SMIBHEX")}
        self.kmap_struct = {x: i for i, x in enumerate("_.()")}
        
        self.token_dicts = {
            "seq": {x: i for i, x in enumerate("_AUGC")}, # 5
            "eterna_struct": {x: i for i, x in enumerate('_.()')}, # 4
            "eterna_loop": {x: i for i, x in enumerate("_SMIBHEX")}, # 8
            "contra_struct": {x: i for i, x in enumerate('_.()')}, # 4 
            "contra_loop": {x: i for i, x in enumerate("_SMIBHEX")}, # 8
        }
        self.len_arr = np.array((df.sequence if self.infer else df.seq).apply(len))
        
    

    def __len__(self):
        return len(self.df)

    def pad_seq(self, seq: torch.Tensor):
        return F.pad(seq, (0, self.len_max - len(seq)))

    # 첫 번째 차원(시퀀스 길이)에만 패딩을 추가합니다.
    def pad_inputs(self, seq: torch.Tensor):        
        # 0을 기본 패딩 값으로 사용합니다.
        padding = (0, 0, 0, self.len_max - seq.size(0))  # (왼쪽, 오른쪽, 위, 아래)
        return F.pad(seq, padding, "constant", 0)


    ############## preprocess ###########################    
    def preprocess_feature_col(self, series, col):        
        dic = self.token_dicts[col]
        dic_len = len(dic)
        seq_length = len(series[col])
        ident = np.identity(dic_len)
        # convert to one hot
        one_hot_encoded = np.array([ident[dic[x]] for x in series[col]])
        # shape: seq_length x dic_length
        return one_hot_encoded

    def preprocess_inputs(self, series, cols):
        one_hot_encoded_features = [self.preprocess_feature_col(series, col) for col in cols]
        # Concatenate along the last dimension
        concatenated_features = np.concatenate(one_hot_encoded_features, axis=1)        
        return concatenated_features
    ###################################################
    @staticmethod
    def get_structure_adj_row_contra(row, WITH_INFER):
        ## get adjacent matrix from a single row of structure sequence

        # Extract information from the row
        structure = row['contra_struct']
        sequence = row['sequence'] if WITH_INFER else row['seq']   
        seq_length = len(row['sequence']) if WITH_INFER else len(row['seq'])
        
        # Initialize the adjacent matrices for each type of base pair
        a_structures = {
            ("A", "U"): np.zeros([seq_length, seq_length]),
            ("C", "G"): np.zeros([seq_length, seq_length]),
            ("U", "G"): np.zeros([seq_length, seq_length]),
            ("U", "A"): np.zeros([seq_length, seq_length]),
            ("G", "C"): np.zeros([seq_length, seq_length]),
            ("G", "U"): np.zeros([seq_length, seq_length]),
        }
        
        # Initialize the stack for keeping track of the structure
        cue = []
        
        # Parse the structure and sequence to fill in the adjacent matrices
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        # Combine all individual base pair matrices into one
        a_structure = np.stack([a for a in a_structures.values()], axis=2)
        a_structure = np.sum(a_structure, axis=2, keepdims=True)
        
        return a_structure

    ###################################################
    @staticmethod
    def get_structure_adj_row_eterna(row, WITH_INFER):
        ## get adjacent matrix from a single row of structure sequence

        # Extract information from the row
        structure = row['eterna_struct']
        sequence = row['sequence'] if WITH_INFER else row['seq']   
        seq_length = len(row['sequence']) if WITH_INFER else len(row['seq'])
        
        # Initialize the adjacent matrices for each type of base pair
        a_structures = {
            ("A", "U"): np.zeros([seq_length, seq_length]),
            ("C", "G"): np.zeros([seq_length, seq_length]),
            ("U", "G"): np.zeros([seq_length, seq_length]),
            ("U", "A"): np.zeros([seq_length, seq_length]),
            ("G", "C"): np.zeros([seq_length, seq_length]),
            ("G", "U"): np.zeros([seq_length, seq_length]),
        }
        
        # Initialize the stack for keeping track of the structure
        cue = []
        
        # Parse the structure and sequence to fill in the adjacent matrices
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        # Combine all individual base pair matrices into one
        a_structure = np.stack([a for a in a_structures.values()], axis=2)
        a_structure = np.sum(a_structure, axis=2, keepdims=True)
        
        return a_structure

    # distance adj
    @staticmethod
    def get_distance_matrix(As):
        ## adjacent matrix based on distance on the sequence
        ## D[i, j] = 1 / (abs(i - j) + 1) ** pow, pow = 1, 2, 4
        
        idx = np.arange(As.shape[1])
        D = np.abs(idx[:, None] - idx[None, :]) + 1
        D = 1/D
        
        Dss = []
        for i in [1, 2, 4]: 
            Dss.append(D ** i)
        Ds = np.stack(Dss, axis=2)
                
        return Ds

    @staticmethod
    def pad_to_target_shape(array, target_dim1, target_dim2, pad_value=0):
        # array의 현재 모양을 가져옵니다.
        current_shape = array.shape
        
        # 첫 번째와 두 번째 차원에 필요한 패딩 양을 계산합니다.
        pad_width = [
            (0, max(target_dim1 - current_shape[0], 0)),  # 첫 번째 차원
            (0, max(target_dim2 - current_shape[1], 0))   # 두 번째 차원
        ]
        
        # 나머지 차원에 대해서는 패딩을 추가하지 않습니다.
        pad_width += [(0, 0) for _ in current_shape[2:]]
    
        # 패딩을 적용합니다.
        return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)

    
    def _pad(self, x: torch.Tensor):
        z = [0] * (1 + (x.ndim - 1) * 2)
        v = float("nan") if x.dtype == torch.float else 0
        return F.pad(x, z + [self.len_max - len(x)], value=v)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]            
        seq = r.sequence if self.infer else r.seq
        seq_len = len(seq)    
        seq = torch.IntTensor([self.kmap_seq[_] for _ in seq])
        mask = torch.zeros(self.len_max, dtype=torch.bool)
        mask[: len(seq)] = True
        ##############################################################

        eterna_loop = r.eterna_loop
        eterna_loop = torch.IntTensor([self.kmap_loop[_] for _ in eterna_loop])
    
        
        eterna_free_energy = r.eterna_free_energy
        contra_free_energy = r.contra_free_energy        
        eterna_free_energy = torch.FloatTensor([eterna_free_energy])
        contra_free_energy = torch.FloatTensor([contra_free_energy])
        ###################################################################
        seq_id = r.sequence_id if self.infer else r.seq_id
        if self.infer:
            As_eterna = np.load(f'../bpps/{seq_id}.npy') # eterna bpp
            # As_contra = np.load(f'../bpps_contrafold/{seq_id}.npy') # eterna bpp
            # As_rfold = np.load(f'../bpp_rfold/{seq_id}.npy')[0] # contra bpp
        
        else:
            As_eterna = np.load(f'../../bpps/{seq_id}.npy')
            # As_contra = np.load(f'../../bpps_contrafold/{seq_id}.npy')
            # As_rfold = np.load(f'../../bpp_rfold/{seq_id}.npy')[0]            

        # As = np.load(f'../../bpps/{seq_id}.npy') # As: (177, 177)
        Ss_contra = self.get_structure_adj_row_contra(r, WITH_INFER=self.infer) # Ss: (177, 177, 1)
        Ss_eterna = self.get_structure_adj_row_eterna(r, WITH_INFER=self.infer) # Ss: (177, 177, 1)        
        Ds_eterna = self.get_distance_matrix(As_eterna) # Ds: (177, 177, 3)
        

    
        ## concat adjecents
        # As = np.concatenate([As_eterna[:,:,None], Ss_eterna, Ds_eterna], axis=2).astype(np.float32)   
        # As = np.concatenate([As_eterna[:,:,None]], axis=2).astype(np.float32)  # only bpps

        As = np.concatenate([As_eterna[:,:,None]], axis=2).astype(np.float32)  # only bpps
        
        # As = self.pad_to_target_shape(As, self.len_max, self.len_max) if As.shape[0] < self.len_max else As[:seq_len, :seq_len, :]
        As = self.pad_to_target_shape(As, self.len_max, self.len_max)
        As = torch.FloatTensor(As) # tensorformat
        ###################################################################
            
        if self.infer:
            return {"seq": self._pad(seq), "As": As, "mask": mask, "eterna_free_energy": eterna_free_energy, "contra_free_energy": contra_free_energy, "eterna_loop": self._pad(eterna_loop)}
            # return {"seq": self._pad(seq), "mask": mask}
        aux = {"loop": None, "struct": None}
        if self.aux_loop:
            loop = r.eterna_loop if self.aux_loop == "eterna" else r.contra_loop
            aux["loop"] = torch.LongTensor([self.kmap_loop[_] for _ in loop])
        if self.aux_struct:
            struct = r.eterna_struct if self.aux_struct == "eterna" else r.contra_struct
            aux["struct"] = torch.LongTensor([self.kmap_struct[_] for _ in struct])
        react = torch.Tensor(np.array([r.react_DMS, r.react_2A3]))
        react = react.transpose(0, 1).clip(0, 1)[: len(seq)]
        if self.aug_reverse and random.random() < 0.1: # 0.5 -> 0.1
            seq, react = seq.flip(0), react.flip(0)
            aux = {k: v.flip(0) for k, v in aux.items() if v != None}
        aux = {k: self._pad(v) for k, v in aux.items() if v != None}
        out = {"seq": self._pad(seq), "As": As, "mask": mask, "eterna_free_energy": eterna_free_energy, "contra_free_energy": contra_free_energy, "eterna_loop": self._pad(eterna_loop), "react": self._pad(react)}
        # out = {"seq": self._pad(seq), "mask": mask, "react": self._pad(react)}
        return out | aux

def collate_fn(samples):
    list_keys = list(samples[0].keys())
    output = {}
    truncated_output = {}
    for k in list_keys:        
        output[k] = torch.stack([sample[k] for sample in samples], 0)    
    max_len = output["mask"].sum(-1).max()    
    for k in list_keys:
        if(len(output[k].shape)==4): # Matrix
            truncated_output[k] = output[k][:,:max_len,:max_len]
        else:                
            truncated_output[k] = output[k][:,:max_len]    
    return truncated_output


class SingleGPULenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        yielded_batches = []
        buckets = [[]] * 100
        yielded = 0
        for idx in self.sampler:
            L = self.sampler.data_source.len_arr[idx]
            L = max(1,L // 16) 
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                #yield batch
                yielded_batches.append(batch)
                yielded += 1
                buckets[L] = []                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                #yield batch
                yielded_batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            #yield batch
            yielded_batches.append(batch)
        batch_order = np.arange(0, len(yielded_batches), dtype = int)
        np.random.shuffle(batch_order)
        for idx in range(0, len(yielded_batches)):
            yield yielded_batches[batch_order[idx]]
            

class MultiGPULenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        yielded_batches = []
        buckets = [[]] * 100
        yielded = 0
        for idx in self.sampler:
            L = self.sampler.dataset.len_arr[idx]
            L = max(1,L // 16) 
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                #yield batch
                yielded_batches.append(batch)
                yielded += 1
                buckets[L] = []                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                #yield batch
                yielded_batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            #yield batch
            yielded_batches.append(batch)
        batch_order = np.arange(0, len(yielded_batches), dtype = int)
        np.random.shuffle(batch_order)
        for idx in range(0, len(yielded_batches)):
            yield yielded_batches[batch_order[idx]]


class RNA_DM(pl.LightningDataModule):
    def __init__(
        self,
        hp: Namespace,
        n_workers: int = 0,
        df_infer: pd.DataFrame | None = None,
        df_train: pd.DataFrame | None = None,        
        fold_idxs: list | None = None,
        
        pseudo_labeling: bool=False,
        df_pseudo: pd.DataFrame | None = None,
        fold_idxs_for_pseudo: list | None = None,
    ):
        super().__init__()
        self.df_train = None
        self.df_val = None      
        self.df_pseudo = None
        self.df_infer = df_infer
        
        if fold_idxs and df_train is not None:
            self.df_train = df_train.iloc[fold_idxs[0]]
            self.df_val = df_train.iloc[fold_idxs[1]]            

        
        if pseudo_labeling: # using pseudo labeling
            before_train_shape = self.df_train.shape
            print(before_train_shape)
            self.df_infer_train = df_infer.iloc[fold_idxs_for_pseudo[0]].reset_index(drop=True)
            self.df_infer_val = df_infer.iloc[fold_idxs_for_pseudo[1]].reset_index(drop=True)
            self.df_pseudo = df_pseudo            
            
            self.df_train = merge_train_infer(self.df_train, self.df_infer_train, self.df_pseudo)
            self.df_val = merge_train_infer(self.df_val, self.df_infer_val, self.df_pseudo)
            after_train_shape = self.df_train.shape
            print(after_train_shape)
        
            if before_train_shape[0] == after_train_shape[0]:
                print(f'Not Using Pseudo label...')
            else:
                print(f'Using Pseudo label...')
        
                    
        self.sn_min = getattr(hp, "sn_min", 0)
        self.aux = getattr(hp, "aux_loop", None), getattr(hp, "aux_struct", None)
        self.aug = getattr(hp, "aug_reverse", False)
        self.kwargs = {
            # "batch_size": hp.batch_size,
            "num_workers": n_workers,
            "pin_memory": bool(n_workers),
        }
        self.is_multi_gpu = torch.cuda.device_count() > 1
        self.use_lenmatched_batch = hp.use_lenmatched_batch
        self.batch_size = hp.batch_size

    def train_dataloader(self):
        assert self.df_train is not None
        ds = RNA_DS(self.df_train, self.sn_min, *self.aux, self.aug)
        if self.use_lenmatched_batch:
            sampler = torch.utils.data.RandomSampler(ds)
            if self.is_multi_gpu:
                sampler = MultiGPULenMatchBatchSampler(sampler, batch_size=self.batch_size, drop_last=False)              
            else:
                sampler = SingleGPULenMatchBatchSampler(sampler, batch_size=self.batch_size, drop_last=False)
            return DataLoader(ds, batch_sampler = sampler, collate_fn = collate_fn,  **self.kwargs)
        else:
            kwargs = self.kwargs | {"shuffle": True}
            return torch.utils.data.DataLoader(ds, batch_size = self.batch_size, **kwargs)
        
    def val_dataloader(self):
        assert self.df_val is not None
        ds = RNA_DS(self.df_val, 1, *self.aux)
        dl = DataLoader(ds, batch_size=self.batch_size, collate_fn = collate_fn, **self.kwargs)    
        return dl

    def predict_dataloader(self):
        assert self.df_infer is not None    
        ds = RNA_DS(self.df_infer, -1, *self.aux)
        dl = DataLoader(ds, collate_fn = collate_fn, batch_size = self.batch_size, **self.kwargs)        
        return dl
