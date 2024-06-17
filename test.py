from transformers import (
   BertTokenizerFast,
   AutoModelForMaskedLM,
   AutoModelForCausalLM,
   AutoModelForTokenClassification,
)

# masked language model (ALBERT, BERT)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForMaskedLM.from_pretrained('ckiplab/albert-tiny-chinese') # or other models above

# casual language model (GPT2)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese') # or other models above

# nlp task model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForTokenClassification.from_pretrained('ckiplab/albert-tiny-chinese-ws') # or other models above