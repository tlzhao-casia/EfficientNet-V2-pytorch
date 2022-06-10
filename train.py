import os
import argparse
import copy
import math
import time

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from tqdm import tqdm
import numpy as np

import effnetv2
from config import Config
from meter import AccMeter, AvgMeter

def create_model(rank, config):
	model_cfg = config.model
	kwargs = dict()
	if hasattr(model_cfg, 'kwargs'): kwargs.update(model_cfg.kwargs)
	model = eval(f'effnetv2.{model_cfg.name}(**kwargs)')

	if hasattr(model_cfg, 'checkpoint'):
		if not (config.distributed and rank != 0):
			print(f'==> Loading parameters from {model_cfg.checkpoint}')
		ck = torch.load(model_cfg.checkpoint, map_location = 'cpu')
		if 'model' in ck: ck = ck['model']
		ck = {(k[7:] if k.startswith('module.') else k):v for k, v in ck.items()}
		model.load_state_dict(ck)

	if config.distributed:
		model.to(rank)
		model = DDP(model, [rank])
	model.to(rank)
	
	return model

def create_dataloader(rank, config):
	data_cfg = config.data
	
	if data_cfg.name == 'ImageNet':
		train_transforms = torchvision.transforms.Compose(
			[
				torchvision.transforms.RandomResizedCrop(224),
				torchvision.transforms.RandomHorizontalFlip(),
				torchvision.transforms.RandAugment(),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(
					[0.485, 0.456, 0.406],
					[0.229, 0.224, 0.225]
				)
			]
		)
		val_transforms = torchvision.transforms.Compose(
			[
				torchvision.transforms.Resize(256),
				torchvision.transforms.CenterCrop(224),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(
					[0.485, 0.456, 0.406],
					[0.229, 0.224, 0.225]				
				)
			]
		)
		train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_cfg.root, 'train'), train_transforms)
		val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_cfg.root, 'val'), val_transforms)
	else:
		raise ValueError(f'Unrecognized dataset {data_cfg.name}')
	
	train_sampler = None
	val_sampler = None
	if config.distributed:
		train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle = True)
		val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle = False)
	
	train_loader = torch.utils.data.DataLoader(train_dataset, config.train_batch.start, sampler = train_sampler,
			shuffle = train_sampler is None, num_workers = data_cfg.workers_per_gpu, pin_memory = True)
	val_loader = torch.utils.data.DataLoader(val_dataset, config.val_batch.start, sampler = val_sampler, num_workers = data_cfg.workers_per_gpu, pin_memory = True)

	return train_loader, val_loader

def adjust_lr(optimizer, epoch, config):
	if epoch < config.warmup_epochs:
		lr = (config.lr - config.base_lr) / (config.warmup_epochs) * epoch + config.base_lr
	else:
		theta = math.pi * (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
		lr = config.lr * 0.5 * (1 + math.cos(theta))
	for pg in optimizer.param_groups:
		pg['lr'] = lr 
	return lr

def mixup(data, target, mixup_alpha, num_classes):
	batch_size = len(target)
	soft_target = torch.zeros(batch_size, num_classes, dtype = data.dtype, device = data.device)
	soft_target.scatter_(dim = 1, index = target.reshape(batch_size, 1), value = 1)
	
	if mixup_alpha == 0:
		return data, soft_target

	indices = torch.randperm(batch_size)
	target1 = soft_target
	target2 = soft_target[indices]
	data1 = data
	data2 = data[indices]

	lamb = np.random.beta(mixup_alpha, mixup_alpha)

	data = lamb * data1 + (1 - lamb) * data2
	soft_target = lamb * target1 + (1 - lamb) * target2

	return data, soft_target
	
def criterion(out, target):
	logits = torch.log_softmax(out, dim = 1)
	return - (logits * target).sum(dim = 1).mean()

def gather_tensor(tensor, rank, distributed):
	if not distributed: return tensor

	if rank != 0:
		dist.gather(tensor)
	else:
		tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
		dist.gather(tensor, tensor_list)
	dist.barrier()

	if rank == 0:
		return torch.cat(tensor_list, dim = 0)
	return None

@torch.no_grad()
def evaluation(model, val_loader, rank, config, topks = [1, 5]):
	model.eval()
	acc_meter = AccMeter(topks)
	loss_meter = AvgMeter()

	if not (config.distributed and rank != 0):
		val_loader = tqdm(val_loader)

	for data, target in val_loader:
		data = data.to(rank)
		target = target.to(rank)
		out = model(data)

		# Gather output tensor
		out = gather_tensor(out, rank, config.distributed)
		target = gather_tensor(target, rank, config.distributed)

	
		# Synchronization here. I donot know whether it is necessary, but
		# add it here for safety.	
		if config.distributed: dist.barrier()

		if not (config.distributed and rank != 0):
			loss = torch.nn.CrossEntropyLoss()
			l = loss(out, target)
			loss_meter.update(l.item())
			acc_meter.update(out, target)
	return acc_meter, loss_meter.avg
				

def main(rank, world_size, config):

	if config.distributed:
		dist.init_process_group('gloo', rank = rank, world_size = world_size)
		config = copy.deepcopy(config)
	
	# Create the model
	model = create_model(rank, config)

	# Create dataloader
	train_loader, val_loader = create_dataloader(rank, config)

	mixup_alpha = 0
	def update_regularization(epoch, config, force_update = False):
		def get_value(stage, cfg):
			start = cfg.start
			end = cfg.end
			nstages = len(config.stages)
			value = float(end - start) / float(nstages - 1) * stage + start
			dtype = eval(cfg.type) if isinstance(cfg.type, str) else cfg.type
			if hasattr(cfg, 'divisor'):
				value = _make_divisible(value, cfg.divisor)
			return dtype(value)
		stage = 0
		for e in config.stages[1:]:
			if epoch >= e: stage += 1
		if epoch in config.stages or force_update:
			# Model-related regularizations
			## Dropout
			dropout = get_value(stage, config.dropout)
			for name, m in model.named_modules():
				if isinstance(m, torch.nn.Dropout):
					if not (config.distributed and rank != 0):
						print(f'==> Updating dropout ratio of {name} to {dropout}')
					m.p = dropout
			
			# Data-related regularizations
			## Image size
			image_size = get_value(stage, config.image_size)
			if not (config.distributed and rank != 0):
				print(f'==> Updating image size to {image_size}')
			train_transforms = train_loader.dataset.transform.transforms 
			train_transforms[0].size = (image_size, image_size)
			val_transforms = val_loader.dataset.transform.transforms
			val_transforms[0].size = (int(image_size * 1.15), int(image_size * 1.15))
			val_transforms[1].size = (image_size, image_size)
			
			## Augment magnitude
			aug_magnitude = get_value(stage, config.augment_magnitude)
			if not (config.distributed and rank != 0):
				print(f'==> Updating augment magnitude to {aug_magnitude}')
			train_transforms[2].magnitude = aug_magnitude

			## TODO: Add batch size adaptation
			if not (config.distributed and rank != 0):
				print(f'==> Batch size adaptation not supported now.')

			## Mixup-alpha
			mixup_alpha = get_value(stage, config.mixup_alpha)
			if not (config.distributed and rank != 0):
				print(f'==> Set mixup alpha value to {mixup_alpha:.6f}')

	# Create optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr = config.optim.lr.lr, weight_decay = config.optim.weight_decay, momentum = config.optim.momentum)

	# TODO: Add resume training codes
	
	# Start training
	start_epoch = 0
	best_acc = 0
	if not os.path.exists(config.save):
		os.makedirs(config.save)

	for epoch in range(start_epoch, config.optim.epochs):
		model.train()
		# Adjust the learning rate
		lr = adjust_lr(optimizer, epoch, config.optim.lr)
		# Adjust regularization
		update_regularization(epoch, config)
		# Set dataloader epoch
		if config.distributed:
			train_loader.sampler.set_epoch(epoch)
		if not (config.distributed and rank != 0):
			print(f'Epoch: {epoch + 1} | {config.optim.epochs}. LR = {lr:.6f}')
		bar = train_loader
		if not (config.distributed and rank != 0):
			bar = tqdm(train_loader)
		loss_meter = AvgMeter()
		data_meter = AvgMeter()
		fb_meter = AvgMeter()
		acc_meter = AccMeter()
		epoch_start = time.time()
		end = time.time()
		for i, (data, target) in enumerate(bar):
			start = time.time()
			data_meter.update(start - end)
			data = data.to(rank)
			target = target.to(rank)
			# Apply the mixup augmentation
			num_classes = model.num_classes if not config.distributed else model.module.num_classes 
			data, soft_target = mixup(data, target, mixup_alpha, num_classes)
			out = model(data)
			# Compute the loss
			loss = criterion(out, soft_target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_meter.update(loss.item())
			acc_meter.update(out, target)
			end = time.time()
			fb_meter.update(end - start)
			if (i + 1) % 100 == 0 and not (config.distributed and rank != 0):
				print(f'Iteration {i + 1} | {train_loader.__len__()}. '
					  f'Data time: {(data_meter.avg * 1000):.3f} ms. '
					  f'Forward-backward time: {(fb_meter.avg * 1000):.3f} ms. '
					  f'{acc_meter.accs_str()}'
					  f'Loss: {loss_meter.avg:.6f}')
		val_acc_meter, val_loss = evaluation(model, val_loader, rank, config, topks = [1, 5])
		if not (config.distributed and rank != 0):
			# TODO: Save here
			state = {
				'model': model.module.state_dict() if config.distributed else model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch + 1,
				'acc': val_acc_meter.accs,
				'topk': val_acc_meter.topks
			}
			torch.save(state, os.path.join(config.save, 'model.pth'))
			if val_acc_meter.accs[0] > best_acc:
				best_acc = val_acc_meter.accs[0]
				torch.save(state, os.path.join(config.save, 'model.best.pth'))
			
			epoch_end = time.time()
			print(f'Runtime: {(epoch_end - epoch_start) / 60:.2f} min. '
				  f'Train {acc_meter.accs_str()}. Loss: {loss_meter.avg:.6f}. '
				  f'Val {val_acc_meter.accs_str()}. Loss: {val_loss:.6f}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type = str, default = None,
		help = 'The configuration file for trainig.')
	args = parser.parse_args()

	config = Config.from_file(args.config)
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = str(config.port)

	if torch.cuda.is_available():
		config.setdefault('device', [0])
		config.distributed = isinstance(config.device, list) and len(config.device) > 1
	else:
		config.device = torch.cpu()
		config.distributed = False

	if not config.distributed:
		main(0, 1, config)
	else:
		world_size = len(config.device)
		mp.spawn(
			main,
			args = (world_size, config),
			nprocs = world_size,
			join = True
		)
