# 下载模型到本地
from transformers import AutoTokenizer, AutoModel

# model_name = "bert-large-uncased"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# model.save_pretrained('utils/models/bert-large-uncased')
# tokenizer.save_pretrained('utils/models/bert-large-uncased')

# Use local path
model_path = ''

# Load model and tokenizer from local directory
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded from local directory.")