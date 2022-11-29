import os
import dataclasses
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Tuple
from sklearn import metrics
from scipy.special import expit, softmax

try:
    from torch.cuda.amp import autocast  # pylint: disable=import-error,no-name-in-module
    from torch.cuda.amp import GradScaler  # pylint: disable=import-error,no-name-in-module
except ModuleNotFoundError:
    pass

@dataclasses.dataclass
class TrainerOptions:
    net: torch.nn.Module = None
    dataloader: torch.utils.data.DataLoader = None
    num_classes: int = 0
    optimizer: torch.optim.Optimizer = None
    criterion: torch.nn.CrossEntropyLoss = None
    save_dir: Path = None
    checkpoint_dir: Path = None
    freeze: list = dataclasses.field(default_factory=list)
    accumulate_over_n_batches: int = 1
    distributed: bool = False
    gpu_rank: int = 0
    n_gpus: int = 1
    test_time_bn: bool = False
    dtype: torch.dtype = torch.float32
    mixedprecision: bool = False
    multilabel: bool = False
    regression: bool = False

class Trainer():
    images_seen: int = 0
    accumulated_batches: int = 0
    accumulated_loss: float = 0.0
    accumulated_accuracy: float = 0.0

    all_logits = []
    all_labels = []

    def __init__(self, options: TrainerOptions):
        self.net = options.net
        self.dataloader = options.dataloader
        self.num_classes = options.num_classes
        self.optimizer = options.optimizer
        self.criterion = options.criterion
        self.save_dir = options.save_dir
        self.checkpoint_dir = options.checkpoint_dir
        self.freeze = options.freeze
        self.accumulate_over_n_batches = options.accumulate_over_n_batches
        self.distributed = options.distributed
        self.gpu_rank = options.gpu_rank
        self.n_gpus = options.n_gpus
        self.test_time_bn = options.test_time_bn
        self.dtype = options.dtype
        if self.distributed: self.config_distributed(self.n_gpus, self.gpu_rank)
        self.mixedprecision = options.mixedprecision
        if self.mixedprecision:
            self.grad_scaler = GradScaler(init_scale=8192, growth_interval=4)
        self.multilabel = options.multilabel
        self.regression = options.regression
        self.reset_epoch_stats()

    def config_distributed(self, n_gpus, gpu_rank=None):
        self.sync_networks_distributed_if_needed()
        self.n_gpus = torch.cuda.device_count() if n_gpus is None else n_gpus
        assert gpu_rank is not None
        self.gpu_rank = gpu_rank

    def sync_networks_distributed_if_needed(self, check=True):
        if self.distributed: self.sync_network_distributed(self.net, check)

    def sync_network_distributed(self, net, check=True):
        for _, param in net.named_parameters():
            dist.broadcast(param.data, 0)

        for mod in net.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                dist.broadcast(mod.running_mean, 0)
                dist.broadcast(mod.running_var, 0)

    def prepare_network_for_training(self):
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        self.net.train()
        for mod in self.freeze:
            mod.eval()

    def prepare_network_for_tuning(self):
        torch.set_grad_enabled(False)
        self.net.eval()
        self.prepare_batchnorm_for_tuning(self.net)

    def prepare_batchnorm_for_tuning(self, net):
        for mod in net.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                if self.test_time_bn: mod.train()
                else: mod.eval()

    def reset_epoch_stats(self):
        self.accumulated_loss = 0
        self.accumulated_batch_metrics = {'acc': 0.0}
        self.batches_seen = 0
        self.images_seen = 0
        self.accumulated_batches = 0

        self.all_probs = np.empty((0,self.num_classes))
        self.all_logits = []
        self.all_labels = []

    def save_batch_stats(self, loss, batch_metrics, logits, labels):
        self.accumulated_loss += float(loss) * len(labels)
        for name, val in batch_metrics.items():
            self.accumulated_batch_metrics[name] += val * len(labels)
        self.batches_seen += 1
        self.images_seen += len(labels)

        if self.multilabel:
            probs = expit(logits)
        else:
            probs = softmax(logits, axis=1)
        self.all_probs = np.append(self.all_probs , probs, axis=0)

        self.all_logits.append(logits)
        self.all_labels.append(labels.copy())  # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189 | fix RuntimeError: received 0 items of ancdata

    def stack_epoch_logits(self):
        self.all_logits, self.all_labels = self.epoch_logits_and_labels(gather=True)

    def correct_loss_for_multigpu(self):
        self.images_seen = len(self.all_logits)
        self.accumulated_loss = self.get_epoch_loss() * len(self.all_logits)
        self.accumulated_epoch_metrics = self.get_epoch_metrics()

    def epoch_logits_and_labels(self, gather=False):
        logits, labels = [], []
        if len(self.all_logits) > 0:
            logits = np.vstack(self.all_logits)
            labels = np.vstack(self.all_labels)

            if self.distributed and gather:
                logits = list(self.gather(logits))
                labels = list(self.gather(labels))

            logits = torch.from_numpy(np.array(logits))
            labels = torch.from_numpy(np.array(labels).astype(self.all_labels[0].dtype))

            # reshape to correct shapes
            if len(self.all_labels[0].shape) == 1:
                labels = labels.flatten()
            # labels = labels.view(-1, self.all_labels[0].shape[0])
            if len(self.all_logits[0].shape) == 1:
                logits = logits.flatten()
            # logits = logits.view(-1, self.all_logits[0].shape[0])
            return logits.float(), labels
        else:
            return torch.FloatTensor(), torch.LongTensor()

    def gather(self, results):
        results = torch.tensor(results, dtype=torch.float32).cuda()
        tensor_list = [results.new_empty(results.shape) for _ in range(self.n_gpus)]
        dist.all_gather(tensor_list, results)
        cpu_list = [tensor.cpu().numpy() for tensor in tensor_list]
        return np.concatenate(cpu_list, axis=0)

    def get_batch_loss(self):
        if self.images_seen == 0:
            return -1
        return self.accumulated_loss / self.images_seen

    def get_epoch_loss(self):
        accumulated_loss = 0.0
        for logit, label in zip(self.all_logits, self.all_labels):
            accumulated_loss += float(self.criterion(logit[None], label[None]))
        return accumulated_loss / len(self.all_logits)

    def get_epoch_metrics(self):
        if self.images_seen == 0:
            return -1
        epoch_metric_dict = {}
        for name, val in self.accumulated_batch_metrics.items():
            epoch_metric_dict[name] = val / self.images_seen
        if self.num_classes == 2:
            epoch_metric_dict['auc'] = metrics.roc_auc_score(self.all_labels, self.all_probs[:,1])
        else:
            epoch_metric_dict['auc'] = metrics.roc_auc_score(self.all_labels, self.all_probs, multi_class='ovr')
        return epoch_metric_dict

    def train_epoch(self, batch_callback) -> Tuple[np.array, np.array]:
        self.sync_networks_distributed_if_needed()
        self.prepare_network_for_training()
        self.reset_epoch_stats()
        self.train_full_dataloader(batch_callback)
        self.stack_epoch_logits()
        if self.distributed:
            self.correct_loss_for_multigpu()
        return self.all_logits, self.all_labels

    def tune_epoch(self, batch_callback):
        self.prepare_network_for_tuning()
        self.reset_epoch_stats()
        self.tune_full_dataloader(batch_callback)
        self.stack_epoch_logits()
        if self.distributed:
            self.correct_loss_for_multigpu()
        return self.all_logits, self.all_labels

    def train_full_dataloader(self, batch_callback):
        for x, y in self.dataloader:
            loss, batch_metrics, logits = self.train_on_batch(x, y)
            self.save_batch_stats(loss, batch_metrics, logits, y.cpu().numpy())
            batch_callback(self, self.batches_seen, loss)

    def tune_full_dataloader(self, batch_callback):
        for x, y in self.dataloader:
            loss, batch_metrics, logits = self.forward_batch(x, y)
            self.save_batch_stats(loss, batch_metrics, logits, y.cpu().numpy())
            batch_callback(self, self.batches_seen, loss)

    def forward_batch(self, x, y):
        if self.mixedprecision:
            with autocast():
                output, loss = self.forward_batch_with_loss(x, y)
        else:
            output, loss = self.forward_batch_with_loss(x, y)
        output = output.detach().cpu()
        metrics = self.get_batch_metrics(output, y.cpu())
        # NOTE: removed a `del output` here, could cause memory issues
        return loss, metrics, output.numpy()

    def forward_batch_with_loss(self, x, y):
        output = self.net.forward(x.cuda())
        label = y.cuda()
        loss = self.criterion(output, label)
        return output, loss

    def train_on_batch(self, x, y):
        loss, batch_metrics, logits = self.forward_batch(x, y)
        full_loss = float(loss)
        loss = loss / self.accumulate_over_n_batches / self.n_gpus
        if self.mixedprecision:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        self.accumulated_batches += 1
        self.step_optimizer_if_needed()
        return full_loss, batch_metrics, logits

    def step_optimizer_if_needed(self):
        if self.accumulated_batches == self.accumulate_over_n_batches:
            self.distribute_gradients_if_needed()
            if self.mixedprecision:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                # prohibit scales larger than 65536, training crashes,
                # maybe due to gradient accumulation?
                if self.grad_scaler.get_scale() > 65536.0:
                    self.grad_scaler.update(new_scale=65536.0)
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_batches = 0

    def distribute_gradients_if_needed(self):
        if self.distributed:
            for _, param in self.net.named_parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    def save_checkpoint(self, name, epoch, additional={}):
        state = {
            'checkpoint': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        state.update(additional)
        print('Saving', name + '_' + str(epoch) + '_network')
        try:
            torch.save(state, self.checkpoint_dir / Path(name + '_' + str(epoch) + '_network'))
            torch.save(state, self.checkpoint_dir / Path(name + '_last'))
        except Exception as e:
            print('WARNING: Network not stored', e)

    def checkpoint_available_for_name(self, name, epoch=-1):
        if epoch > -1:
            print(self.checkpoint_dir / Path(name + '_' + str(epoch) + '_network'))
            print(os.path.isfile(self.checkpoint_dir / Path(name + '_' + str(epoch) + '_network')))
            return os.path.isfile(self.checkpoint_dir / Path(name + '_' + str(epoch) + '_network'))
        else:
            return os.path.isfile(self.checkpoint_dir / Path(name + '_last'))

    def load_network_checkpoint(self, name):
        state = torch.load(self.checkpoint_dir / Path(name))
        self.load_state_dict(state)

    def load_checkpoint(self, name, epoch=-1):
        if epoch > -1:
            state = torch.load(self.checkpoint_dir / Path(name + '_' + str(epoch) + '_network'),
                               map_location=lambda storage, loc: storage)
        else:
            state = torch.load(self.checkpoint_dir / Path(name + '_last'),
                               map_location=lambda storage, loc: storage)
        return state

    def load_state_dict(self, state):
        try: self.optimizer.load_state_dict(state['optimizer'])
        except KeyError: print('WARNING: Optimizer not restored')
        self.net.load_state_dict(state['state_dict'])

    def load_checkpoint_if_available(self, name, epoch=-1):
        if self.checkpoint_available_for_name(name, epoch):
            state = self.load_checkpoint(name, epoch)
            self.load_state_dict(state)
            return True, state
        return False, None

    def get_batch_metrics(self, logits, labels):
        if self.regression:
            self.batch_metrics = {'acc': 0.0}
            return self.batch_metrics
        if self.multilabel:
            probs = torch.sigmoid(logits.float())
            preds = np.round(probs) # basically 0.5 thresholding
        else:
            preds = torch.topk(logits.float(), 1, dim=1)[1]
        acc = metrics.accuracy_score(labels, preds)
        self.batch_metrics = {'acc': acc}
        return self.batch_metrics
