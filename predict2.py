import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
# device='cpu'
device='cuda'
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
model = AutoModelForSeq2SeqLM.from_pretrained('/home/dezs/projects/Correct-Vietnamese-Accent-using-T5-Vision-Transformer/ModelStep8')
model.to(device)

# Define the text you want to predict
# text = "Đây la hanh vi lấy cap đien rat nguy hiem, co the gay ra sự co cháy nổ làm gián đoạn cung cap đien, lam cháy thiết bi và dac biet rất dễ xay ra tai nạn diện."
# text = "Chao mung ban den voi thanh pho Can Tho."
# text = "Viet Nam la mot quoc gia phat trien."
# Preprocess the text

print("Write something that need correction:")
text = input()

inputs = tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt")

# Generate prediction
outputs = model.generate(
    input_ids=inputs['input_ids'].to(device),
    max_length=512,
    attention_mask=inputs['attention_mask'].to(device),
)

# Decode the output and print
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Result: \n"+ output_text)
