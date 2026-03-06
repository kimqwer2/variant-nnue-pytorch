import argparse
import model as M
import nnue_dataset
import pytorch_lightning as pl
import features
import os
import shutil
import torch
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

def make_data_loaders(train_filenames, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, cyclic, horizontal_mirroring, main_device, epoch_size, val_size):
  features_name = feature_set.name
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filenames, batch_size, num_workers=num_workers,
                                                   cyclic=cyclic, filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device,
                                                   horizontal_mirroring=horizontal_mirroring)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device, horizontal_mirroring=False,
                                                   cyclic=False)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def main():
  parser = argparse.ArgumentParser(description="Trains the network.")
  parser.add_argument("train", nargs='+', help="Training data files (.bin), one or more.")
  parser.add_argument("val", help="Validation data (.bin)")
  parser = pl.Trainer.add_argparse_args(parser)
  parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
  parser.add_argument("--draw-weight", default=1.0, type=float, dest='draw_weight', help="Loss weight multiplier for positions with draw outcome target (0.5).")
  parser.add_argument("--lr", default=1.5e-3, type=float, dest='lr', help="Base learning rate for optimizer.")
  parser.add_argument("--cyclic-data", action='store_true', dest='cyclic_data', help="Enable cyclic/infinite data loading for training streams.")
  parser.add_argument("--horizontal-mirroring", action='store_true', dest='horizontal_mirroring', help="Enable random horizontal mirroring augmentation for training batches.")
  parser.add_argument("--save-best-model", action='store_true', dest='save_best_model', help="Also save best validation-loss checkpoint as best_model.pt.")
  parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for bin.")
  parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
  parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
  parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping_deprecated', help="If enabled positions that are bad training targets will be skipped during loading. Default: True, kept for backwards compatibility. This option is ignored")
  parser.add_argument("--no-smart-fen-skipping", action='store_true', dest='no_smart_fen_skipping', help="If used then no smart fen skipping will be done. By default smart fen skipping is done.")
  parser.add_argument("--random-fen-skipping", default=3, type=int, dest='random_fen_skipping', help="skip fens randomly on average random_fen_skipping before using one.")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initializes training using the weights from the given .pt model")
  parser.add_argument("--epoch-size", type=int, default=20000000, dest='epoch_size', help="Number of positions per epoch.")
  parser.add_argument("--validation-size", type=int, default=1000000, dest='validation_size', help="Number of positions per validation step.")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  if not 0.0 <= args.lambda_ <= 1.0:
    raise Exception(f'--lambda must be in [0.0, 1.0], got {args.lambda_}.')
  if args.draw_weight < 0.0:
    raise Exception(f'--draw-weight must be non-negative, got {args.draw_weight}.')
  if args.lr <= 0.0:
    raise Exception(f'--lr must be positive, got {args.lr}.')

  for train_path in args.train:
    if not os.path.exists(train_path):
      raise Exception('{0} does not exist'.format(train_path))
  if not os.path.exists(args.val):
    raise Exception('{0} does not exist'.format(args.val))

  feature_set = features.get_feature_set_from_name(args.features)

  if args.resume_from_model is None:
    nnue = M.NNUE(
      feature_set=feature_set,
      lambda_=args.lambda_,
      draw_weight=args.draw_weight,
      lr=args.lr,
    )
    nnue.cuda()
  else:
    # Load with weights_only=False to avoid safe_globals complexity
    # This is safe since we trust the checkpoint source
    nnue = torch.load(args.resume_from_model, weights_only=False)
    nnue.set_feature_set(feature_set)
    nnue.lambda_ = args.lambda_
    nnue.draw_weight = args.draw_weight
    nnue.lr = args.lr
    nnue.cuda()

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))

  print("Training with {} validating with {}".format(', '.join(args.train), args.val))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = 16384
  print('Using batch size {}'.format(batch_size))

  print('Smart fen skipping: {}'.format(not args.no_smart_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  callbacks = [pl.callbacks.ModelCheckpoint(save_last=True, every_n_epochs=1, save_top_k=-1)]
  best_model_callback = None
  if args.save_best_model:
    best_model_callback = pl.callbacks.ModelCheckpoint(
      monitor='val_loss',
      mode='min',
      save_top_k=1,
      filename='best_model',
      dirpath=logdir,
      every_n_epochs=1,
      save_weights_only=True,
    )
    callbacks.append(best_model_callback)
  trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=tb_logger)

  main_device = trainer.strategy.root_device if trainer.strategy.root_device.index is None else 'cuda:' + str(trainer.strategy.root_device.index)

  print('Using c++ data loader')
  train, val = make_data_loaders(args.train, args.val, feature_set, args.num_workers, batch_size, not args.no_smart_fen_skipping,
                                 args.random_fen_skipping, args.cyclic_data, args.horizontal_mirroring,
                                 main_device, args.epoch_size, args.validation_size)

  trainer.fit(nnue, train, val)

  if args.save_best_model and best_model_callback is not None and best_model_callback.best_model_path:
    shutil.copyfile(best_model_callback.best_model_path, os.path.join(logdir, 'best_model.pt'))

if __name__ == '__main__':
  main()
