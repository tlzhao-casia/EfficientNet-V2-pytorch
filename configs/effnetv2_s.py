# The configuration file for efficientnetv2-s training
save = './logs/efficientnetv2_s'

device = [0, 1, 2, 3, 4, 5, 6, 7]
port = '56789'

# Configure the model
model = dict(
	name = 'efficientnetv2_s',
	kwargs = dict(
		__isdict__ = True,
		num_classes = 1000,
		width_mult = 1,
		dropout = 0.1,
		p = 0.2
	)
)

# Configure the dataset
data = dict(
	name = 'ImageNet',
	root = '/path/to/imagenet',
	workers_per_gpu = 4
)

# Regularization configuration
## The batch size (per-gpu)
train_batch = dict(
	type = 'int',
	divisor = 8,
	start = 128,
	end = 128
)
val_batch = dict(
	type = 'int',
	divisor = 8,
	start = 64,
	end = 64
)
## The rand augment megnitude
augment_magnitude = dict(
	type = 'int',
	start = 5,
	end = 15
)
## The mixup alpha
mixup_alpha = dict(
	type = 'float',
	start = 0,
	end = 0
)
## The dropout rate
dropout = dict(
	type = 'float',
	start = 0.1,
	end = 0.3
)
## Image size
image_size = dict(
	type = 'int',
	start = 128,
	end = 300
)
## Modify epochs
stages = [0, 87, 87 * 2, 87 * 3]

## Optimization
epochs = 350
optim = dict(
	epochs = epochs,
	weight_decay = 1e-5,
	momentum = 0.9,
	lr = dict(
		scheduler = 'cosine',
		lr = 0.5,
		warmup_epochs = 10,
		base_lr = 0.001,
		epochs = epochs
	)
)
