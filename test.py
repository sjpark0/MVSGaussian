import tqdm

class MyIndexable():
	def __init__(self):
		self.data = []
        # self.data = [i for i in range(10)]
		for i in range(10):
			self.data += [i]
		
	def __getitem__(self, index):
		#index, cnt = index_meta
		ret = {}
		ret["img"] = self.data[index]
		return ret


obj = MyIndexable()
for b in tqdm.tqdm(obj):
	print(b)
	