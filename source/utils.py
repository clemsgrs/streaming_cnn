import os
import sys
import time
import math
import wandb
import torch
import subprocess
import torchvision
import numpy as np

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from source.tissue_dataset import TissueDataset
from source.torch_utils.samplers import OrderedDistributedSampler, DistributedWeightedRandomSampler
from source.torch_utils.streaming_trainer import StreamingCheckpointedTrainer, StreamingTrainerOptions


def initialize_wandb(project, entity, exp_name, dir='./wandb', config={}, tags=None, key=''):
    dir = Path(dir)
    command = f'wandb login {key}'
    subprocess.call(command, shell=True)
    if tags == None:
        tags=[]
    run = wandb.init(project=project, entity=entity, name=exp_name, dir=dir, config=config, tags=tags)
    return run


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def progress_bar(current, total, msg=None):

    global last_time, begin_time

    stty_pipe = os.popen('stty size', 'r')
    stty_output = stty_pipe.read()
    if len(stty_output) > 0:
        _, term_width = stty_output.split()
        term_width = int(term_width)
    else:
        # Set a default in case we couldn't read the term width
        term_width = 100

    TOTAL_BAR_LENGTH = 35.
    last_time = time.time()
    begin_time = last_time

    current -= 1
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for _ in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for _ in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for _ in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for _ in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Experiment():
    tuner: StreamingCheckpointedTrainer
    tune_dataset: torch.utils.data.Dataset
    tune_loader: torch.utils.data.DataLoader

    trainer: StreamingCheckpointedTrainer
    train_dataset: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    train_sampler: torch.utils.data.DistributedSampler

    settings: DictConfig

    distributed: bool
    world_size: int

    verbose: bool
    resumed: bool = False

    optimizer = None
    loss: torch.nn.Module
    epoch: int
    freeze_layers: list = []
    start_at_epoch: int = 0

    net: torch.nn.Module
    stream_net: torch.nn.Sequential

    def __init__(self, settings: DictConfig, running_distributed, world_size):
        """Initialize an experiment, this is an utility class to get training
        quickly with this framework.

        This class is responsible for setting up the model, optimizer, loss,
        dataloaders and the trainer classes.

        Args:
            settings (DictConfig): dataclass containing all the options
                for this experiment, parsed from cli arguments.
            distributed: whether we are gonna train on multiple gpus
            world_size: how many gpus are in the system
        """
        self.settings = settings
        self.rank_0 = (self.settings.local_rank == 0)
        self.world_size = world_size
        self.distributed = running_distributed

        # When training with mixed precision and only finetuning last layers, we
        # do not have to backpropagate the streaming layers
        if self.settings.mixedprecision and not self.settings.train_all_layers:
            self.settings.train_streaming_layers = False

        torch.cuda.set_device(int(self.settings.local_rank))
        torch.backends.cudnn.benchmark = True  # type:ignore

    def run_experiment(self):
        self.configure_experiment()
        if self.settings.only_tune: self.tune_epoch(0)
        else: self.train_and_tune_epochs()

    def configure_experiment(self):
        if self.distributed: self._test_distributed()
        self._configure_batch_size_per_gpu(self.world_size)
        self._configure_dataloaders()
        self._configure_model()
        self._configure_optimizer()
        self._configure_loss()
        self._configure_trainers()
        self._resume_if_needed()
        self._sync_distributed_if_needed()
        self._enable_mixed_precision_if_needed()
        self._log_details()
        if self.settings.variable_input_shapes: self._configure_tile_delta()

    def _test_distributed(self):
        if self.rank_0:
            print('Test distributed')
        results = torch.FloatTensor([0])  # type:ignore
        results = results.cuda()
        tensor_list = [results.new_empty(results.shape) for _ in range(self.world_size)]
        torch.distributed.all_gather(tensor_list, results)
        if self.rank_0:
            print('Succeeded distributed communication')

    def _configure_batch_size_per_gpu(self, world_size):
        """
        This functions calculates how we will devide the batch over multiple
        GPUs, and how many image gradients we are gonna accumulate before doing
        an optimizer step.
        """
        if self.settings.accumulate_batch == -1:
            self.settings.accumulate_batch = int(self.settings.batch_size / world_size)
            self.settings.batch_size = 1
        elif not self.settings.gather_batch_on_one_gpu:
            self.settings.batch_size = int(self.settings.batch_size / world_size)
            self.settings.accumulate_batch = self.settings.accumulate_batch
        else:
            self.settings.batch_size = int(self.settings.batch_size / world_size)

        if self.rank_0:
            print(f'Per GPU batch-size: {self.settings.batch_size}, ' +
                  f'accumulate over batch: {self.settings.accumulate_batch}')

        assert self.settings.batch_size > 0
        assert self.settings.accumulate_batch > 0

    def train_and_tune_epochs(self):
        epochs_to_train = np.arange(self.start_at_epoch, self.settings.epochs)
        for e in epochs_to_train:
            self.train_epoch(e, self.trainer)
            if self.settings.tune and e % self.settings.tune_interval == 0:
                self.tune_epoch(e)

    def train_epoch(self, e, trainer):
        self.epoch = e
        wandb.log({'epoch': e})
        logits, gt = trainer.train_epoch(self._train_batch_callback)
        if self.rank_0:
            self.log_train_metrics()
        if self.distributed:
            self.train_sampler.set_epoch(int(e + 10))
        if self.settings.mixedprecision and e == 0:
            self.trainer.grad_scaler.set_growth_interval(20)

    def log_train_metrics(self):
        metrics = self.trainer.get_epoch_metrics()
        loss = self.trainer.get_epoch_loss()
        print(f'Train loop, accuracy = {metrics["acc"]}, auc = {metrics["auc"]}, loss = {loss}')
        wandb.define_metric('train/loss', step_metric='epoch')
        wandb.log({'train/loss': loss})
        for name, val in metrics.items():
            wandb.define_metric(f'train/{name}', step_metric='epoch')
            wandb.log({f'train/{name}': val})
        for _, param_group in enumerate(self.optimizer.param_groups):
            lr = param_group['lr']
        wandb.log({'train/lr': lr})

    def log_tune_metrics(self):
        metrics = self.tuner.get_epoch_metrics()
        loss = self.tuner.get_epoch_loss()
        print(f'Tune loop, accuracy = {metrics["acc"]}, auc = {metrics["auc"]}, loss = {loss}')
        wandb.define_metric('tune/loss', step_metric='epoch')
        wandb.log({'tune/loss': loss})
        for name, val in metrics.items():
            wandb.define_metric(f'tune/{name}', step_metric='epoch')
            wandb.log({f'tune/{name}': val})

    def tune_epoch(self, e):
        logits, gt = self.tuner.tune_epoch(self._tune_batch_callback)

        if self.rank_0:
            if self.settings.only_tune: str_e = self.settings.resume_epoch
            else: str_e = str(e)
            self.save_logits(logits, gt, str_e)
            self.log_tune_metrics()
            self.save_if_needed(e)

    def save_logits(self, logits, gt, str_e):
        path = Path(self.settings.save_dir)
        try:
            np.save(str(path / Path(f'{self.settings.exp_name}_tune_logits_{str_e}')), logits)
            np.save(str(path / Path(f'{self.settings.exp_name}_tune_gt_{str_e}')), gt)
        except Exception as exc:
            print(f'Logits could not be written to disk: {exc}')

    def save_if_needed(self, e):
        if self.settings.save and not self.settings.only_tune:
            self.trainer.save_checkpoint(self.settings.exp_name, e)

    def _configure_dataloaders(self):
        self.train_dataset = self._get_dataset(training=True, csv_file=self.settings.train_csv)
        self.train_loader, self.train_sampler = self._get_dataloader(self.train_dataset, shuffle=True)
        self.tune_dataset = self._get_dataset(training=False, csv_file=self.settings.tune_csv)
        self.tune_loader, _ = self._get_dataloader(self.tune_dataset, shuffle=False)

    def _get_dataloader(self, dataset: torch.utils.data.Dataset, shuffle=True):
        batch_size, num_workers = 1, self.settings.num_workers
        sampler = None

        if self.settings.weighted_sampler:
            if shuffle:
                sampler = self.weighted_sampler(dataset)
                shuffle = False

        if self.distributed:
            if shuffle:
                shuffle = False
                sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                          num_replicas=torch.cuda.device_count(),
                                                                          rank=self.settings.local_rank)
            else:
                if self.distributed:
                    sampler = OrderedDistributedSampler(dataset, num_replicas=torch.cuda.device_count(),
                                                        rank=self.settings.local_rank, batch_size=1)

        # TODO: maybe disable automatic batching?
        # https://pytorch.org/docs/stable/data.html
        # pin memory True saves GPU memory?
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            pin_memory=False)

        return loader, sampler

    def weighted_sampler(self, dataset):
        labels = np.array([int(label) for _, label in dataset.images])
        total_pos = np.sum(labels)
        total_neg = len(labels) - total_pos
        weights = np.array(labels, dtype=np.float32)
        weights[labels==0] = total_pos / total_neg
        weights[labels==1] = 1
        if self.distributed:
            sampler = DistributedWeightedRandomSampler(weights, num_samples=len(self.train_dataset), replacement=True)
        else:
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(self.train_dataset), replacement=True)
        return sampler

    def _get_dataset(self, training, csv_file):
        limit = self.settings.train_set_size
        if not training:
            # limit = -1
            limit = self.settings.tune_set_size
        variable_input_shapes = self.settings.tune_whole_input if not training else self.settings.variable_input_shapes
        return TissueDataset(img_size=self.settings.image_size,
                             img_dir=self.settings.data_dir,
                             cache_dir=self.settings.copy_dir,
                             filetype=self.settings.filetype,
                             csv_fname=csv_file,
                             training=training,
                             limit_size=limit,
                             variable_input_shapes=variable_input_shapes,
                             tile_size=self.settings.tile_size,
                             multiply_len=self.settings.epoch_multiply if training else 1,
                             num_classes=self.settings.num_classes,
                             regression=self.settings.regression,
                             convert_to_vips=self.settings.convert_to_vips)

    def _train_batch_callback(self, tuner, batches_tuned, loss):
        if self.rank_0 and self.settings.progressbar:
            batch_metrics = tuner.batch_metrics
            epoch_loss = tuner.get_batch_loss()
            batch_acc = batch_metrics['acc']
            progress_bar(batches_tuned, math.ceil(len(self.train_dataset) / float(self.world_size)),
                         '%s loss: %.3f, acc: %.3f, b loss: %.3f' %
                         ("Train", epoch_loss, batch_acc, loss))

    def _tune_batch_callback(self, tuner, batches_tuned, loss):
        if self.rank_0 and self.settings.progressbar:
            batch_metrics = tuner.batch_metrics
            epoch_loss = tuner.get_batch_loss()
            batch_acc = batch_metrics['acc']
            progress_bar(batches_tuned, math.ceil(len(self.tune_dataset) / float(self.world_size)),
                         '%s loss: %.3f, acc: %.3f, b loss: %.3f' %
                         ("Tune", epoch_loss, batch_acc, loss))

    def _configure_optimizer(self):
        params = self._get_trainable_params()
        self.optimizer = torch.optim.SGD(params, lr=self.settings.lr, momentum=0.9)

    def _get_trainable_params(self):
        if self.settings.train_all_layers:
            params = list(self.stream_net.parameters()) + list(self.net.parameters())
        else:
            print('WARNING: optimizer only training last params of network!')
            if self.settings.mixedprecision:
                params = list(self.net.parameters())
                for param in self.stream_net.parameters(): param.requires_grad = False
            else:
                params = list(self.stream_net[-1].parameters()) + list(self.net.parameters())
                for param in self.stream_net[:-1].parameters(): param.requires_grad = False
        return params

    def _configure_trainers(self):
        options = StreamingTrainerOptions()
        options.dataloader = self.train_loader
        options.num_classes = self.settings.num_classes
        options.net = self.net
        options.optimizer = self.optimizer
        options.criterion = self.loss  # type:ignore
        options.save_dir = Path(self.settings.save_dir)
        options.checkpoint_dir = Path(self.settings.checkpoint_dir)
        options.checkpointed_net = self.stream_net
        options.batch_size = self.settings.batch_size
        options.accumulate_over_n_batches = self.settings.accumulate_batch
        options.n_gpus = self.world_size
        options.gpu_rank = int(self.settings.local_rank)
        options.distributed = self.distributed
        options.freeze = self.freeze_layers
        options.tile_shape = (1, 3, self.settings.tile_size, self.settings.tile_size)
        options.dtype = torch.uint8  # not needed, but saves memory
        options.train_streaming_layers = self.settings.train_streaming_layers
        options.mixedprecision = self.settings.mixedprecision
        options.normalize_on_gpu = self.settings.normalize_on_gpu
        options.multilabel = self.settings.multilabel
        options.regression = self.settings.regression
        options.gather_batch_on_one_gpu = self.settings.gather_batch_on_one_gpu
        self.trainer = StreamingCheckpointedTrainer(options)

        self.tuner = StreamingCheckpointedTrainer(options, sCNN=self.trainer.sCNN)
        self.tuner.dataloader = self.tune_loader
        self.tuner.accumulate_over_n_batches = 1

        # StreamingCheckpointedTrainer changes modules, reset optimizer!
        self._reset_optimizer()

    def _configure_tile_delta(self):
        if isinstance(self.trainer, StreamingCheckpointedTrainer):
            delta = self.settings.tile_size - (self.trainer.sCNN.tile_gradient_lost.left
                                               + self.trainer.sCNN.tile_gradient_lost.right)
            delta = delta // self.trainer.sCNN.output_stride[-1]
            delta *= self.trainer.sCNN.output_stride[-1]
            # if delta < 3000:
            #     delta = (3000 // delta + 1) * delta
            print('DELTA', delta.item())
            self.train_dataset.tile_delta = delta.item()
            self.tune_dataset.tile_delta = delta.item()

    def _configure_loss(self):
        weight = None
        if self.settings.multilabel:
            self.loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        elif self.settings.regression:
            self.loss = torch.nn.SmoothL1Loss()
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def _configure_model(self):
        if self.settings.mobilenet:
            self._configure_mobilenet()
        elif self.settings.resnet:
            net = self._configure_resnet()
            self.stream_net, self.net = self._split_model(net)
        self._freeze_bn_layers()

    def _configure_mobilenet(self):
        net = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=self.settings.pretrained)
        net.features[1].conv[0][0].stride = (2, 2)  # TODO: or maybe add averagepool after first conv
        net.classifier = torch.nn.Linear(1280, 1)
        torch.nn.init.normal_(net.classifier.weight, 0, 0.01)
        torch.nn.init.zeros_(net.classifier.bias)
        self.stream_net = torch.nn.Sequential(*net.features[0:13]).cuda()
        net.features = net.features[13:]
        self.net = net.cuda()

    def _configure_resnet(self):
        net = torchvision.models.resnet34(pretrained=self.settings.pretrained)
        net.fc = torch.nn.Linear(512, self.settings.num_classes)
        torch.nn.init.xavier_normal_(net.fc.weight)
        net.fc.bias.data.fill_(0)  # type:ignore
        net.avgpool = torch.nn.AdaptiveMaxPool2d(1)
        net.cuda()
        return net

    def _freeze_bn_layers(self):
        self.freeze_layers = [l for l in self.stream_net.modules() if isinstance(l, torch.nn.BatchNorm2d)]
        self.freeze_layers += [l for l in self.net.modules() if isinstance(l, torch.nn.BatchNorm2d)]

    def _sync_distributed_if_needed(self):
        if self.distributed:
            self.trainer.sync_networks_distributed_if_needed()
            self.tuner.sync_networks_distributed_if_needed()

    def _resume_if_needed(self):
        state = None
        resumed = False

        if self.settings.resuming and self.settings.resume_epoch == -1:
            resumed, state = self._try_resuming_last_checkpoint(resumed)

        if not resumed and self.settings.resume_epoch > -1:
            name = self.settings.resume_name if self.settings.resume_name else self.settings.exp_name

            if self.settings.weight_averaging:
                window = 5
                resumed, state = self.resume_with_averaging(name,
                                                            self.settings.resume_epoch - math.floor(window / 2.),
                                                            self.settings.resume_epoch + math.floor(window / 2.))
            else:
                resumed, state = self._resume_with_epoch(name)

            print('Did not reset optimizer, maybe using lr from checkpoint')
            assert self.trainer.net == self.tuner.net  # type:ignore
            assert resumed

        self.resumed = resumed
        self._calculate_starting_epoch(resumed, state)
        del state

    def _resume_with_epoch(self, name):
        resumed, state = self.trainer.load_checkpoint_if_available(name, self.settings.resume_epoch)
        resumed, state = self.tuner.load_checkpoint_if_available(name, self.settings.resume_epoch)
        return resumed, state

    def resume_with_averaging(self, resume_name, begin_epoch, after_epoch, window=5):
        param_dict = {}

        # sum all parameters
        # checkpoint_range = np.arange(epoch - math.floor(window / 2.), epoch + math.ceil(window / 2.))
        checkpoint_range = np.arange(begin_epoch, after_epoch)
        for i in checkpoint_range:
            try:
                current_param_dict = dict(self.trainer.load_checkpoint(resume_name, i))
            except Exception as e:
                print(f'Did not find {i}', e)
                return False, None

            if not param_dict:
                param_dict = current_param_dict
            else:
                for key in ['state_dict_net', 'state_dict_checkpointed']:
                    for name in current_param_dict[key]:
                        param_dict[key][name].data.add_(current_param_dict[key][name].data)

        for key in ['state_dict_net', 'state_dict_checkpointed']:
            for name in param_dict[key]:
                param_dict[key][name].data.div_(float(len(checkpoint_range)))

        self.trainer.net.load_state_dict(param_dict['state_dict_net'])
        self.trainer.checkpointed_net.load_state_dict(param_dict['state_dict_checkpointed'])

        return True, param_dict

    def _try_resuming_last_checkpoint(self, resumed):
        name = self.settings.exp_name
        resumed, state = self.trainer.load_checkpoint_if_available(name)
        resumed, state = self.tuner.load_checkpoint_if_available(name)

        if resumed and self.rank_0:
            print("WARNING: look out, learning rate from resumed optimizer is used!")
        return resumed, state

    def _reset_optimizer(self):
        self._configure_optimizer()
        self.trainer.optimizer = self.optimizer
        self.tuner.optimizer = self.optimizer

    def _calculate_starting_epoch(self, resumed, state):
        start_at_epoch = 0
        if resumed:
            start_at_epoch = state['checkpoint'] + 1
            if self.rank_0: print(f'Resuming from epoch {start_at_epoch}')

        self.start_at_epoch = start_at_epoch

    def _split_model(self, net):
        if not self.settings.mixedprecision:
            stream_net = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                             net.layer1, net.layer2, net.layer3,
                                             net.layer4[0])
            net.layer4 = net.layer4[1:]
        else:
            stream_net = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                             net.layer1, net.layer2, net.layer3)

        net.layer1 = torch.nn.Sequential()
        net.layer2 = torch.nn.Sequential()
        net.layer3 = torch.nn.Sequential()
        net.conv1 = torch.nn.Sequential()
        net.bn1 = torch.nn.Sequential()
        net.relu = torch.nn.Sequential()
        net.maxpool = torch.nn.Sequential()

        return stream_net, net

    def _enable_mixed_precision_if_needed(self):
        if self.settings.mixedprecision:
            if isinstance(self.trainer, StreamingCheckpointedTrainer):
                self.trainer.sCNN.dtype = torch.half
                self.trainer.mixedprecision = True
            if isinstance(self.tuner, StreamingCheckpointedTrainer):
                self.tuner.sCNN.dtype = torch.half
                self.tuner.mixedprecision = True

    def _log_details(self):
        if self.rank_0:
            print()
            print("PyTorch version", torch.__version__)
            print("Running distributed:", self.distributed)
            print("CUDA memory allocated:", torch.cuda.memory_allocated())
            print("Number of parameters (stream):", count_parameters(self.stream_net))
            print("Number of parameters (final):", count_parameters(self.net))
            print("Len train_loader", len(self.train_loader), '*', self.world_size)
            print("Len tune_loader", len(self.tune_loader), '*', self.world_size)
            print()
            print(OmegaConf.to_yaml(self.settings))
            print()