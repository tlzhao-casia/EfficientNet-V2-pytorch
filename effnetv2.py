import math

import torch
import torchvision

def _make_divisible(c, d):
	minc = c
	outc = max(minc, (c + d / 2) // d * d)
	if outc < 0.9 * c:
		outc += d
	return outc

class SELayer(torch.nn.Module):
	def __init__(self, inc, outc):
		super().__init__()
		self.pool = torch.nn.AdaptiveAvgPool2d(1)
		midc = _make_divisible(inc // 4, 8)
		self.fc = torch.nn.Sequential(
			torch.nn.Linear(outc, midc),
			torch.nn.SiLU(),
			torch.nn.Linear(midc, outc),
			torch.nn.Sigmoid()
		)

	def forward(self, input):
		n, c, _, _ = input.shape
		out = self.pool(input)
		out = out.reshape(n, c)
		out = self.fc(out)
		out = out.reshape(n, c, 1, 1)
		return input * out

class FusedMBConv(torch.nn.Module):
	def __init__(self, inc, outc, expansion_ratio, stride = 1, servival_ratio = 1):
		super().__init__()
		self.inc = inc
		self.outc = outc
		self.expansion_ratio = expansion_ratio
		self.stride = stride
		self.servival_ratio = servival_ratio

		self.identity = (stride == 1 and inc == outc)

		midc = _make_divisible(inc * expansion_ratio, 8)
		self.convs = torch.nn.Sequential(
			torch.nn.Conv2d(inc, midc, 3, stride = stride, padding = 1, bias = False),
			torch.nn.BatchNorm2d(midc),
			torch.nn.SiLU(),
			torch.nn.Conv2d(midc, outc, 1, bias = False),
			torch.nn.BatchNorm2d(outc)
		)

	def forward(self, input):
		out = self.convs(input)
		if not self.identity:
			return out
		out = torchvision.ops.stochastic_depth(out, 1 - self.servival_ratio, mode = 'row', training = self.training)
		return input + out
		
class MBConv(torch.nn.Module):
	def __init__(self, inc, outc, expansion_ratio, stride = 1, servival_ratio = 1):
		super().__init__()
		self.inc = inc
		self.outc = outc
		self.expansion_ratio = expansion_ratio
		self.stride = stride
		self.servival_ratio = servival_ratio

		self.identity = (stride == 1 and inc == outc)

		midc = _make_divisible(inc * expansion_ratio, 8)
		self.convs = torch.nn.Sequential(
			torch.nn.Conv2d(inc, midc, 1, bias = False),
			torch.nn.BatchNorm2d(midc),
			torch.nn.SiLU(),
			torch.nn.Conv2d(midc, midc, 3, padding = 1, stride = stride, groups = midc, bias = False),
			torch.nn.BatchNorm2d(midc),
			torch.nn.SiLU(),
			SELayer(inc, midc),
			torch.nn.Conv2d(midc, outc, 1, bias = False),
			torch.nn.BatchNorm2d(outc)
		)

	def forward(self, input):
		out = self.convs(input)
		if not self.identity:
			return out
		out = torchvision.ops.stochastic_depth(out, 1 - self.servival_ratio, mode = 'row', training = self.training)
		return input + out

class EffnetV2(torch.nn.Module):
	def __init__(self, config, num_classes = 1000, width_mult = 1, dropout = 0):
		'''
			config = [
				[Block, n, s, t, c, p]
				...
			]
		'''	
		super().__init__()
		self.num_classes = num_classes
		# The first convolution layer
		inc = _make_divisible(width_mult * 24, 8)
		features = [
			torch.nn.Sequential(
				torch.nn.Conv2d(3, inc, 3, padding = 1, stride = 2, bias = False),
				torch.nn.BatchNorm2d(inc),
				torch.nn.SiLU()
			)
		]
		for block, n, s, t, c, p in config:
			outc = _make_divisible(c * width_mult, 8)
			for _ in range(n):
				features.append(
					block(inc, outc, t, s if _ == 0 else 1, 1 - p)
				)
				inc = outc
		features.append(
			torch.nn.Sequential(
				torch.nn.Conv2d(inc, 1792, 1, bias = False),
				torch.nn.BatchNorm2d(1792),
				torch.nn.SiLU()
			)
		)
		self.features = torch.nn.Sequential(*features)
		self.pool = torch.nn.AdaptiveAvgPool2d(1)
		self.fc = torch.nn.Sequential(
			torch.nn.Dropout(p = dropout, inplace = True),
			torch.nn.Linear(1792, num_classes)
		)

		# Init parameters
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_out')
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)
			elif isinstance(m, torch.nn.BatchNorm2d):
				torch.nn.init.ones_(m.weight)
				torch.nn.init.zeros_(m.bias)
			elif isinstance(m, torch.nn.Linear):
				init_range = 1. / math.sqrt(m.out_features)
				torch.nn.init.uniform_(m.weight, -init_range, init_range)
				torch.nn.init.zeros_(m.bias)

	def forward(self, input):
		out = self.features(input)
		out = self.pool(out)
		out = out.flatten(1)
		out = self.fc(out)
		return out


def efficientnetv2_s(num_classes = 1000, width_mult = 1, dropout = 0, p = 0.2):
	config = [
		[FusedMBConv, 2, 1, 1, 24, p],
		[FusedMBConv, 4, 2, 4, 48, p],
		[FusedMBConv, 4, 2, 4, 64, p],
		[MBConv, 6, 2, 4, 128, p],
		[MBConv, 9, 1, 6, 160, p],
		[MBConv, 15, 2, 6, 256, p]
	]

	return EffnetV2(config, num_classes, width_mult, dropout)

if __name__ == '__main__':
	input = torch.randn(1, 3, 300, 300, dtype = torch.float32)
	net = efficientnetv2_s()
	print(net)
	out = net(input)
	# print(sum([param.numel() for param in (*net.parameters(), *net.buffers())]))
	print(sum([param.numel() for param in net.parameters()]))
