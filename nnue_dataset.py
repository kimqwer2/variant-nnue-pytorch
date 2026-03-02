import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('psqt_indices', ctypes.POINTER(ctypes.c_int)),
        ('layer_stack_indices', ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        white_indices = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_indices = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        psqt_indices = torch.from_numpy(np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        layer_stack_indices = torch.from_numpy(np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        return us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices

SparseBatchPtr = ctypes.POINTER(SparseBatch)


class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filenames,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        device='cpu',
        horizontal_mirroring=False):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        if isinstance(filenames, (str, bytes)):
            filenames = [filenames]
        if not filenames:
            raise ValueError('At least one training filename is required.')

        self.filenames_raw = [f.decode('utf-8') if isinstance(f, bytes) else f for f in filenames]
        self.filenames = [name.encode('utf-8') for name in self.filenames_raw]
        self.filenames_arr = (ctypes.c_char_p * len(self.filenames))(*self.filenames)
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device
        self.horizontal_mirroring = horizontal_mirroring

        self.stream = self.create_stream(
            self.feature_set,
            self.num_workers,
            self.filenames_arr,
            len(self.filenames),
            self.batch_size,
            cyclic,
            filtered,
            random_fen_skipping,
            horizontal_mirroring
        )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors(self.device)
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        if hasattr(self, "stream"):
            self.destroy_stream(self.stream)

create_sparse_batch_stream = dll.create_sparse_batch_stream
create_sparse_batch_stream.restype = ctypes.c_void_p
create_sparse_batch_stream.argtypes = [
    ctypes.c_char_p,  # feature_set
    ctypes.c_int,     # num_workers
    ctypes.POINTER(ctypes.c_char_p),  # filenames
    ctypes.c_int,     # num_files
    ctypes.c_int,     # batch_size
    ctypes.c_bool,    # cyclic
    ctypes.c_bool,    # filtered
    ctypes.c_int,     # random_fen_skipping
    ctypes.c_bool     # horizontal_mirroring
]
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
destroy_sparse_batch = dll.destroy_sparse_batch


class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filenames, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', horizontal_mirroring=False):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filenames,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device,
            horizontal_mirroring)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filenames, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', horizontal_mirroring=False):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    if isinstance(filenames, (str, bytes)):
      filenames = [filenames]
    self.filenames = filenames
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.device = device
    self.horizontal_mirroring = horizontal_mirroring

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filenames, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, device=self.device, horizontal_mirroring=self.horizontal_mirroring)

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches):
    super(FixedNumBatchesDataset, self).__init__()
    self.dataset = dataset
    self.iter = iter(self.dataset)
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return next(self.iter)
