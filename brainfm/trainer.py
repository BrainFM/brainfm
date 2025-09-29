import os
import torch
import time
import datetime
import math
import sys
import logging
from collections import defaultdict, deque
from tqdm import tqdm
from brainfm.utils import Config
from contextlib import nullcontext

# For mixed precision
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

def _to_device(batch, device: torch.device):
    """Move tensors in a (possibly nested) batch dict to device safely."""
    def move(x):
        return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
    return {k: move(v) for k, v in batch.items()}

class SmoothedValue(object):
    """Track values and provide smoothed estimates"""
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
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
         # Return the last value for current loss display
        if len(self.deque):
             # Convert to tensor if not already, handle potential tensor input
             last_val = self.deque[-1]
             if isinstance(last_val, torch.Tensor):
                 return last_val.item()
             return last_val
        return 0 # Or handle appropriately if deque is empty


    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger if logger else logging.getLogger("MetricLogger")

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not isinstance(v, (float, int)):
                 # Ensure we are tracking numerical values
                 if isinstance(v, (list, tuple)) and len(v) == 1:
                     v = v[0] # Handle single element lists/tuples
                 else:
                     # Attempt conversion, raise warning if not possible
                     try:
                         v = float(v)
                     except (ValueError, TypeError):
                         self.logger.warning(f"Warning: Non-numeric value provided for key '{k}': {v}. Skipping update.")
                         continue

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

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # Length may be unknown for some iterables
        total = len(iterable) if hasattr(iterable, "__len__") else None
        if total is not None:
            space_fmt = ':' + str(len(str(total))) + 'd'
            log_msg = [
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ]
        else:
            # No total length known
            log_msg = [
                header,
                '[{0}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ]
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj # Return the object (e.g., batch) for processing
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or (total is not None and i == total - 1):
                if total is not None and i < total:
                    eta_seconds = iter_time.global_avg * (total - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = 'N/A'
                mem_used_gb = torch.cuda.max_memory_allocated() / (MB * 1024) if torch.cuda.is_available() else 0

                if total is not None:
                    log_output = log_msg.format(
                        i, total, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                else:
                    log_output = log_msg.format(
                        i, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))

                # Add LR and Memory usage
                lr = self.meters.get('lr', None)
                lr_str = f"lr: {lr.value:.6f}" if lr else "lr: N/A"
                mem_str = f"max_mem: {mem_used_gb:.1f}GB"

                self.logger.info(f"{log_output}{self.delimiter}{lr_str}{self.delimiter}{mem_str}")

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / (total if total else 1)))


def save_checkpoint(model, optimizer, epoch, path, scaler=None, extra_info=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    if extra_info is not None and isinstance(extra_info, dict):
        checkpoint.update(extra_info)

    torch.save(checkpoint, path)


def train_one_epoch(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    scaler: torch.cuda.amp.GradScaler = None, # For mixed precision
                    logger: logging.Logger = None,
                    log_freq: int = 50, # Log every N batches
                    clip_grad_norm: float = None # Max norm for gradient clipping
                    ):

    model.train()
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    # Check if AMP is available and scaler is provided
    use_amp = AMP_AVAILABLE and scaler is not None

    # Use tqdm for progress bar if available
    try:
        data_iterator = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
    except Exception:
        data_iterator = dataloader  # Fallback if tqdm not installed
        (logger.warning if logger else print)("tqdm not found. Progress bar disabled.")


    for batch_idx, batch in enumerate(metric_logger.log_every(data_iterator, log_freq, header)):
        # Move batch to device (safe: only tensors)
        batch = _to_device(batch, device)

        # --- Forward pass ---
        amp_enabled = (use_amp and device.type == 'cuda')
        optimizer.zero_grad()
        if 'images' in batch:
            images = batch['images']
            modality_embs = batch['modality_embs']
            modality_mask = batch.get('modality_mask', None)
            if amp_enabled:
                with torch.amp.autocast(device_type='cuda'):
                    loss = model.forward_from_volumes(images, modality_embs, modality_mask=modality_mask)
            else:
                loss = model.forward_from_volumes(images, modality_embs, modality_mask=modality_mask)
        else:
            # Legacy path: call model(**batch) in AMP context if enabled
            if amp_enabled:
                with torch.amp.autocast(device_type='cuda'):
                    loss = model(**batch)
            else:
                loss = model(**batch)

        # Check for NaN/Inf loss
        if not torch.isfinite(loss).item():
            msg = f"Non-finite loss at epoch {epoch}, batch {batch_idx}"
            if logger: logger.error(msg)
            else: print(msg)
            # Dump quick stats
            imgs = batch.get("images")
            if imgs is not None:
                print(f"images stats: min={imgs.min().item():.3g}, max={imgs.max().item():.3g}")
            raise RuntimeError(msg)

        # --- Backward Pass & Optimization ---
        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()


        # --- Logging Metrics ---
        #torch.cuda.synchronize() # Wait for GPU ops to finish for accurate timing if needed
        metric_logger.update(loss=loss.item())
        # Get current learning rate (handle param groups if necessary)
        current_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=current_lr)

        # Close tqdm progress bar if used
        if hasattr(data_iterator, "set_postfix"):
            data_iterator.set_postfix({"Loss": metric_logger.loss.avg, "LR": current_lr})

    # Gather stats from all processes if using distributed training (not implemented here)
    #metric_logger.synchronize_between_processes() # Placeholder if needed for DDP
    if logger:
        logger.info(f"Averaged stats for Epoch {epoch}: {metric_logger}")

    return metric_logger.loss.global_avg # Return the average loss for the epoch


def train(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        config: Config,
        device: torch.device,
        logger: logging.Logger = None):
    
    # For pytorch 2.2+ and mixed precision
    # scaler = torch.amp.GradScaler(device_type='cuda') if AMP_AVAILABLE else None
    # For older pytorch versions
    scaler = GradScaler() if AMP_AVAILABLE and torch.cuda.is_available() else None
    num_epochs = config.train.epochs

    if logger: logger.info("Starting training...")
    else: print("Starting training...")

    # --- Training Loop ---
    start_time = time.time()
    for epoch in range(num_epochs):
        avg_train_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            scaler=scaler,
            logger=logger,
            log_freq=config.train.log_freq,
            clip_grad_norm=config.train.clip_grad_norm
        )

        # Step the scheduler *after* the epoch if it's epoch-based
        if lr_scheduler is not None:
            lr_scheduler.step()

        # --- Save Checkpoint ---
        save_freq = config.train.save_freq
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            ckpt_dir = config.paths.ckpt_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=checkpoint_path,
                scaler=scaler,
                extra_info={"avg_train_loss": avg_train_loss}
            )
            if logger: logger.info(f"Checkpoint saved at {checkpoint_path}")


    # --- Log Total Training Time ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if logger: logger.info(f'Total training time: {total_time_str}')
    else: print(f'Total training time: {total_time_str}')
        


if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __init__(self, num_samples=10, shape=(128,128,128), patch_size=(16,16,16)):
            self.num_samples = num_samples
            self.shape = shape
            self.patch_size = patch_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate a dummy input tensor with shape (channels=1, D, H, W)
            input_tensor = torch.randn(1, *self.shape)
            # Generate a dummy target tensor for example (scalar)
            target = torch.tensor(0.0)
            return {"x": input_tensor, "target": target}

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Simple conv layer to reduce spatial dims
            self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(1, 1)

        def forward(self, x, target=None):
            x = self.conv(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            out = self.fc(x)
            # Compute dummy loss if target provided
            if target is not None:
                loss = (out.squeeze() - target).pow(2).mean()
                return loss
            else:
                # Just return output if no target
                return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DummyDataset(num_samples=20, shape=(128,128,128), patch_size=(16,16,16))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = DummyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run one epoch of training on dummy data
    avg_loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0,
        device=device,
        scaler=None,
        logger=None,
        log_freq=5,
        clip_grad_norm=None
    )
    print(f"Dummy epoch average loss: {avg_loss:.6f}")
