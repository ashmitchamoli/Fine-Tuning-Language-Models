import torch
from transformers import Trainer

class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, dataframe):
		self.dataframe = dataframe
	
	def __getitem__(self, index):
		output = {}
		output['input_ids'] = self.dataframe.loc[index, 'article']
		output['labels'] = self.dataframe.loc[index, 'highlights']

		return output
	
	def __len__(self):
		return len(self.dataframe)

def trainModel(model, train_dataset, val_dataset, training_args = None, compute_metrics = None):
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		compute_metrics=compute_metrics
	)

	trainer.train()

	return trainer