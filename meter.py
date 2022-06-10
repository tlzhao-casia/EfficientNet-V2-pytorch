import torch

class AvgMeter(object):
	def __init__(self):
		self.sum = 0
		self.n = 0
		self.avg = 0

	def update(self, value):
		self.sum += value
		self.n += 1
		self.avg = self.sum / self.n

	def clear(self):
		self.sum = 0
		self.n = 0
		self.avg = 0

class AccMeter(object):
	def __init__(self, topks = [1, 5]):
		self.n = 0
		self.corrects = [0 for _ in topks]
		self.accs = [0 for _ in topks]
		self.topks = sorted(topks)

	def update(self, output, target):
		k = self.topks[-1]

		_, classes = torch.topk(output, k, dim = 1)
		correct = classes.eq(target.reshape(-1, 1))

		for i, k in enumerate(self.topks):
			self.corrects[i] += correct[..., :k].sum().item()

		self.n += len(target)

		self.accs = [c / self.n for c in self.corrects]

	def clear(self):
		self.n = 0
		self.corrects = [0 for _ in self.topks]
		self.accs = [0 for _ in self.topks]

	def accs_str(self):
		return ' '.join(['Acc@{}: {:.2f}. '.format(k, acc * 100) for k, acc in zip(self.topks, self.accs)])
