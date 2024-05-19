import collections
import json
import time
import copy
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.utils.data
import os
import random
from torchvision.datasets import ImageFolder
from domainbed.datasets import get_dataset, split_dataset, set_transfroms
from domainbed import algorithms
from domainbed.evaluator import Evaluator, accuracy_from_loader
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module
from domainbed import hparams_registry
from sconf import Config
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score


def main():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", nargs="*", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="data/Dataset")
    parser.add_argument("--test_dir", type=str, default="data/Dataset/zhong")
    parser.add_argument("--dataset", type=str, default="WBC_DG")
    parser.add_argument("--algorithm", type=str, default="Mixup")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=2, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=8001, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=400,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    args, left_argv = parser.parse_known_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic
    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)
    print(args, hparams)
    device = "cpu"
    if args.device:
        device = args.device
        hparams["device"] = device
    elif torch.cuda.is_available():
        device = "cuda"
    
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset([], args, hparams, algorithm_class)
    n_envs = len(dataset)
    iterator = misc.SplitIterator([])
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=int)
    batch_sizes = batch_sizes.tolist()
    classes = in_splits[0][0].underlying_dataset.classes

    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    
    train_feat_loaders = [FastDataLoader(
        dataset=env,
        batch_size=batch_size,
        num_workers=dataset.N_WORKERS)
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ] if 'BoDA' in args.algorithm else None

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(out_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    
    eval_weights = [None for _, weights in (out_splits)]
    eval_loader_names = ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))
    feat_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]

    #######################################################
    # test set
    #######################################################

    _, test_set= split_dataset(
            ImageFolder(args.test_dir),
            0,
        )
    set_transfroms(test_set, "test", hparams, algorithm_class)
    test_loader = FastDataLoader(
        dataset=test_set,
        batch_size=hparams["test_batchsize"],
        num_workers=16
    )

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset),
        hparams,
    )
    
    algorithm.to(device)
    if args.n_gpu > 1:
        algorithm.featurizer = torch.nn.DataParallel(algorithm.featurizer)


    if 'BoDA' in args.algorithm:
        train_labels = dict()
        for i, (env, _) in enumerate(in_splits):
            labels = np.array(env.underlying_dataset.targets)
            in_labels = labels[env.keys].tolist()
            train_labels["env{}_in".format(i)] = in_labels
        algorithm.init_env_labels(train_labels, device)

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm, device=device)
        swad_cls = getattr(swad_module, hparams["swad"])
        swad = swad_cls(None, **hparams.swad_kwargs)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])


    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches.items()
        }
        train_features = {}
        if 'BoDA' in args.algorithm and (step > 0 and step % hparams["feat_update_freq"] == 0):
            curr_tr_feats, curr_tr_labels = collections.defaultdict(list), collections.defaultdict(list)
            for name, loader in sorted(zip(feat_loader_names, train_feat_loaders), key=lambda x: x[0]):
                algorithm.eval()
                with torch.no_grad():
                    for batch in loader:
                        x = batch["x"].to(device)
                        y = batch["y"].to(device)
                        feats = algorithm.return_feats(x)
                        curr_tr_feats[name].extend(feats.data)
                        curr_tr_labels[name].extend(y.data)
            train_features = {'feats': curr_tr_feats, 'labels': curr_tr_labels}
        inputs = {**batches, "step": step, "max_step": n_steps, "env_feats":train_features}
        
        algorithm.train()
        algorithm.update(**inputs)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)


        if step % checkpoint_freq == 0 and swad:
            
            out_acc = 0.0
            out_head_acc = 0.0
            out_tail_acc = 0.0
            out_loss = 0.0
            algorithm.eval()
            for env_i, (name, loader_kwargs, weights) in enumerate(eval_meta):
                loader = FastDataLoader(**loader_kwargs)
                acc, head_acc, tail_acc, loss = accuracy_from_loader(algorithm, loader, None, device=device)
                out_acc += acc / n_envs
                out_head_acc += head_acc / n_envs
                out_tail_acc += tail_acc / n_envs
                out_loss += loss / n_envs
            print("step:%d"%(step))
            print(out_acc, out_head_acc, out_tail_acc, out_loss)
            if 'SADA' in args.algorithm and step < hparams["self_supervised_step"]:
                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset
                continue
            swad.update_and_evaluate(swad_algorithm, out_acc, out_loss, None)
            swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset


    if swad:
        swad_algorithm = swad.get_final_model(device)
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps, algorithm, device)
        algorithm = swad_algorithm


    print(accuracy_from_loader(algorithm, test_loader, None,device=device))
    y_true = []
    y_pred = []
    for i, batch in enumerate(test_loader):
        y_true += batch["y"].tolist()
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        with torch.no_grad():
            y_pred += torch.argmax(algorithm.predict(x), dim=1).detach().cpu().tolist()
    print("algorithm:%s, seed:%s"%(args.algorithm, args.seed))
    print("f1_micro:%f, f1_macro:%f"%(f1_score(y_true, y_pred, average="micro"), f1_score(y_true, y_pred, average="macro")))
            

                
        
if __name__=="__main__":
    main()
