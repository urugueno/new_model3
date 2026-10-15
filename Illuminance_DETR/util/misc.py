import time
from collections import defaultdict, deque
import datetime
from typing import Optional

import torch
from torch import Tensor, nn

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if not self.deque:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if not self.deque:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        pass

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
            
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()

def collate_fn(batch, args, split):
    illuminance, coords, targets = zip(*batch)
    
    illuminance_batch = torch.stack(illuminance, 0)
    coords_batch = torch.stack(coords, 0)
    
    B, N, W = illuminance_batch.shape

    if split == 'train':            
        dropout_rate = getattr(args, 'sensor_dropout_rate', 0.0)
        if dropout_rate > 0:
            keep_mask = (torch.rand(B, N, device=illuminance_batch.device) > dropout_rate)
            keep_mask[:, 0] = True 
            sensor_padding_mask = ~keep_mask
        else:
            sensor_padding_mask = torch.zeros((B, N), dtype=torch.bool, device=illuminance_batch.device)
    else:
        sensor_padding_mask = torch.zeros((B, N), dtype=torch.bool, device=illuminance_batch.device)
    
    env_vector_batch = None
    if args.environment != 'none':
        if 'env_vector' in targets[0]:
            env_vector_batch = torch.stack([t['env_vector'] for t in targets])
        else:
            env_vector_batch = torch.zeros(B, N, device=illuminance_batch.device)

    samples = {
        'tensors': illuminance_batch,
        'coords': coords_batch,
        'sensor_padding_mask': sensor_padding_mask
    }

    if args.environment in ['EE', 'EQ']:
        samples['env_vector'] = env_vector_batch

    return samples, targets

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

@torch.no_grad()
def recompute_batchnorm_stats(model, data_loader, device):
    """AdaBN: recompute BatchNorm running mean/var from `data_loader` (labels ignored),
    leaving all weights and every other layer's mode untouched."""
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    if not bn_modules:
        return False

    for bn in bn_modules:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()

    for samples, _ in data_loader:
        if isinstance(samples, dict):
            samples = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        else:
            samples = samples.to(device)
        model(samples)

    for bn in bn_modules:
        bn.eval()

    return True

class OnlineNormalizingLoader:
    """Wraps a data_loader and normalizes each batch's samples['tensors'] using
    only the statistics of previously seen batches (not the batch itself), combined
    via an exponential moving average. The first batch falls back to mean_init/std_init
    (e.g. train statistics); the estimate is set to the first batch's own statistics
    right after that batch is yielded, then EMA-updated from the second batch onward."""

    def __init__(self, data_loader, mean_init, std_init, momentum=0.1):
        self.data_loader = data_loader
        self.mean_init = mean_init
        self.std_init = std_init
        self.momentum = momentum

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        mean = self.mean_init.clone()
        var = self.std_init.clone() ** 2
        initialized = False

        for samples, targets in self.data_loader:
            raw = samples['tensors']
            std_eps = torch.sqrt(var) + 1e-6
            normalized = (raw - mean.view(1, -1, 1)) / std_eps.view(1, -1, 1)
            new_samples = dict(samples)
            new_samples['tensors'] = normalized
            yield new_samples, targets

            batch_mean = raw.mean(dim=(0, 2))
            batch_var = raw.var(dim=(0, 2), unbiased=False)
            if not initialized:
                mean, var = batch_mean, batch_var
                initialized = True
            else:
                mean = (1 - self.momentum) * mean + self.momentum * batch_mean
                var = (1 - self.momentum) * var + self.momentum * batch_var