from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/home/dezs/Projects/Nienluan2/myModelMini")
model = AutoModelForSeq2SeqLM.from_pretrained("path_to_your_model")

from datasets import Dataset

# Prepare your data
input_lines = []
label_lines = []
with open('your_test_data.tsv') as file:
  for line in file:
    line = line.strip().split('\t')
    input = line[0]
    input_lines.append(input +'</s>')
    label_lines.append(line[1])

dict_obj = {'inputs': input_lines, 'labels': label_lines}
dataset = Dataset.from_dict(dict_obj)
test_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=4,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
)

trainer.predict(test_dataset)


# Draw a graph

import matplotlib.pyplot as plt

# Assuming that 'history' is your training history
def plot_graphs(history):
    # Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()

# Call the function
plot_graphs(trainer.state.log_history)
