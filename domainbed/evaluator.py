import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader
from sklearn.metrics import f1_score


# def accuracy_from_loader(algorithm, loader, weights, head_class, tail_class, debug=False, device="cpu"):
#     correct = 0
#     head_correct = 0
#     tail_correct = 0
#     total = 0
#     head_total = 0
#     tail_total = 0
#     losssum = 0.0
#     weights_offset = 0

#     algorithm.eval()

#     for i, batch in enumerate(loader):
#         x = batch["x"].to(device)
#         y = batch["y"].to(device)

#         with torch.no_grad():
#             logits = algorithm.predict(x)
#             loss = F.cross_entropy(logits, y).item()

#         B = len(x)
#         losssum += loss * B

#         if weights is None:
#             batch_weights = torch.ones(len(x))
#         else:
#             batch_weights = weights[weights_offset : weights_offset + len(x)]
#             weights_offset += len(x)
#         batch_weights = batch_weights.to(device)
#         head = torch.eq(y.view(-1, 1), head_class.view(-1, 1).T).sum(1)
#         tail = torch.eq(y.view(-1, 1), tail_class.view(-1, 1).T).sum(1)
#         if logits.size(1) == 1:
#             correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
#         else:
#             y_pred = logits.argmax(1)
#             correct += (y_pred.eq(y).float() * batch_weights).sum().item()
#             head_correct += (y_pred.eq(y).float() * batch_weights * head).sum().item()
#             tail_correct += (y_pred.eq(y).float() * batch_weights * tail).sum().item()
#         total += batch_weights.sum().item()
#         head_total += (batch_weights* head).sum().item()
#         tail_total += (batch_weights* tail).sum().item()

#         if debug:
#             break

#     algorithm.train()

#     acc = correct / total
#     head_acc = head_correct / head_total
#     tail_acc = tail_correct / tail_total
#     loss = losssum / total
#     return acc, head_acc, tail_acc, loss


# def accuracy(algorithm, loader_kwargs, weights, head_class, tail_class, device="cpu",**kwargs):
#     if isinstance(loader_kwargs, dict):
#         loader = FastDataLoader(**loader_kwargs)
#     elif isinstance(loader_kwargs, FastDataLoader):
#         loader = loader_kwargs
#     else:
#         raise ValueError(loader_kwargs)
#     return accuracy_from_loader(algorithm, loader, weights, head_class, tail_class, device=device, **kwargs)


# class Evaluator:
#     def __init__(
#         self, test_envs, eval_meta, n_envs, logger, head_class, tail_class, evalmode="fast", debug=False, target_env=None, device="cpu"
#     ):
#         all_envs = list(range(n_envs))
#         train_envs = sorted(set(all_envs) - set(test_envs))
#         self.test_envs = test_envs
#         self.train_envs = train_envs
#         self.eval_meta = eval_meta
#         self.n_envs = n_envs
#         self.logger = logger
#         self.head_class = head_class.to(device)
#         self.tail_class = tail_class.to(device)
#         self.evalmode = evalmode
#         self.debug = debug
#         self.device = device

#         if target_env is not None:
#             self.set_target_env(target_env)

#     def set_target_env(self, target_env):
#         """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
#         self.test_envs = [target_env]

#     def evaluate(self, algorithm, ret_losses=False):
#         n_train_envs = len(self.train_envs)
#         n_test_envs = len(self.test_envs)
#         assert n_test_envs == 1
#         summaries = collections.defaultdict(float)
#         # for key order
#         summaries["test_in"] = 0.0
#         summaries["test_in_head"] = 0.0
#         summaries["test_in_tail"] = 0.0
#         summaries["test_in"] = 0.0
#         summaries["test_out"] = 0.0
#         summaries["train_in"] = 0.0
#         summaries["train_out"] = 0.0
#         accuracies = {}
#         losses = {}

#         # order: in_splits + out_splits.
#         for name, loader_kwargs, weights in self.eval_meta:
#             # env\d_[in|out]
#             env_name, inout = name.split("_")
#             env_num = int(env_name[3:])

#             skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
#             if skip_eval:
#                 continue

#             is_test = env_num in self.test_envs
#             acc, head_acc, tail_acc, loss = accuracy(algorithm, loader_kwargs, weights, self.head_class, self.tail_class, debug=self.debug, device=self.device)
#             accuracies[name] = acc
#             losses[name] = loss

#             if env_num in self.train_envs:
#                 summaries["train_" + inout] += acc / n_train_envs
#                 if inout == "out":
#                     summaries["tr_" + inout + "loss"] += loss / n_train_envs
#             elif is_test:
#                 summaries["test_" + inout] += acc / n_test_envs
#                 if inout == "in":
#                     summaries["test_in_head"] += head_acc / n_test_envs
#                     summaries["test_in_tail"] += tail_acc / n_test_envs
        
#         if ret_losses:
#             return accuracies, summaries, losses
#         else:
#             return accuracies, summaries


def accuracy_from_loader(algorithm, loader, weights, debug=False, device="cpu"):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0
    y_true = []
    y_pred = []
    algorithm.eval()

    for i, batch in enumerate(loader):
        y_true += batch["y"].tolist()
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()
            y_pred += torch.argmax(logits, dim=1).cpu().tolist()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            pred = logits.argmax(1)
            correct += (pred.eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()
    acc = correct / total
    loss = losssum / total
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return acc, f1_micro, f1_macro, loss


def accuracy(algorithm, loader_kwargs, weights, device="cpu",**kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, device=device, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None, device="cpu"
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.device = device

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_f1_micro"] = 0.0
        summaries["test_f1_macro"] = 0.0
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, f1_micro, f1_macro, loss = accuracy(algorithm, loader_kwargs, weights, debug=self.debug, device=self.device)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs
                if inout == "in":
                    summaries["test_f1_micro"] += f1_micro / n_test_envs
                    summaries["test_f1_macro"] += f1_macro / n_test_envs
        
        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
