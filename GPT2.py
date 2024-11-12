import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json

# Load parsed data
with open('formatted.json', 'r') as openfile:
    parsed_data = json.load(openfile)

class PromptResponseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        response = item['response']
        
        # Combine prompt and response
        input_text = f"{prompt}\n{response}"
        
        # Encode using the tokenizer
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Resize the model's token embeddings if necessary
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Move model to MPS device if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Create the dataset
dataset = PromptResponseDataset(parsed_data, tokenizer)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./GPT2_fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./GPT2_fine_tuned_model")
tokenizer.save_pretrained("./GPT2_fine_tuned_model")

# Function to generate a response
def generate_response(prompt):
    # Move input to the same device as the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    response_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "Hey Daniel if you have free time we can talk more about the math society thing"