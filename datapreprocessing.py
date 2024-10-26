import pandas as pd
import os
import pickle
from alive_progress import alive_bar as aliveBar
from transformers import AutoTokenizer

DATA_DIR = "./data/cnn_dailymail"

trainLimit = 21000
testLimit = 3000
valLimit = 6000

trainDf = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).sample(trainLimit).reset_index(drop=True)
testDf = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).sample(testLimit).reset_index(drop=True)
valDf = pd.read_csv(os.path.join(DATA_DIR, "validation.csv")) .sample(valLimit).reset_index(drop=True)

print(trainDf.head())
print(len(trainDf), len(testDf), len(valDf))

tokenizer = AutoTokenizer.from_pretrained("gpt2",
										  pad_token="<PAD>")

with aliveBar(len(trainDf) + len(testDf) + len(valDf)) as bar:
	def tokenizeDf(row):
		row['article'] = tokenizer(row['article'], padding='max_length', truncation=True, max_length=256, return_tensors="pt").input_ids.squeeze(0)
		row['highlights'] = tokenizer(row['highlights'], padding='max_length', truncation=True, max_length=256, return_tensors="pt").input_ids.squeeze(0)
		bar()
		return row	

	trainDf = trainDf.apply(tokenizeDf, axis=1)
	testDf = testDf.apply(tokenizeDf, axis=1)
	valDf = valDf.apply(tokenizeDf, axis=1)

print(trainDf.head())
print(testDf.head())
print(valDf.head())

with open(os.path.join(DATA_DIR, "train.pkl"), "wb") as f:
	pickle.dump(trainDf, f)

with open(os.path.join(DATA_DIR, "test.pkl"), "wb") as f:
	pickle.dump(testDf, f)

with open(os.path.join(DATA_DIR, "val.pkl"), "wb") as f:
	pickle.dump(valDf, f)