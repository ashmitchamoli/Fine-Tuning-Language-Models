from peft import get_peft_model, get_peft_config, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

from utils import trainModel, CustomDataset

def finetuneGPT2Small(
		model,
		tokenizer,
		batchSize : int,
		epochs : int,
		learningRate : float,
		gradClipping : float = 1.0
):
	pass

# load gpt2 small model and its tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

rank = 8

peftConfig = LoraConfig(
	task_type=TaskType.CAUSAL_LM,
	r=8,
	lora_alpha=32,
	lora_dropout=0.1,
)

loraModel = get_peft_model(model, peft_config=peftConfig)
loraModel.print_trainable_parameters()

trainDf = pickle.load(open("./data/cnn_dailymail/train.pkl", "rb"))
valDf = pickle.load(open("./data/cnn_dailymail/val.pkl", "rb"))

trainDataset = CustomDataset(trainDf)
valDataset = CustomDataset(valDf)

trainer = trainModel(loraModel, trainDataset, valDataset)