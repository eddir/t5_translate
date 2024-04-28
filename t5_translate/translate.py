import time

from transformers import T5ForConditionalGeneration, T5Tokenizer

start = time.time()

device = 'cpu' #or 'cpu' for translate on cpu

model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

prefix = 'translate to ru: '
src_text = prefix + "it flies horizontally, completely ignores air resistance, and therefore accelerates to crazy speeds"

# translate Russian to Chinese
input_ids = tokenizer(src_text, return_tensors="pt")

generated_tokens = model.generate(**input_ids.to(device))

result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(result)

print(f"Time taken: {time.time() - start}")