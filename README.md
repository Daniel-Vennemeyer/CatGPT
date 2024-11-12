# Discord Fine-Tuned GPT Model aka CatGPT

This project fine-tunes a GPT-2 language model on parsed Discord messages to emulate the conversational style of a specific individual-- Cat Luong. By overfitting the model with a smaller dataset, we achieve a highly personalized tone that captures specific stylistic nuances for realistic, customized responses.

## Watch Demo [here](https://youtu.be/ZW07jbZEec0) 

## Key Architectural Decisions

### 1. **Model Selection: GPT-2**
   - **Choice**: GPT-2 was chosen as the base model due to its versatility and relatively small size compared to larger LLMs like GPT-3, making it easier to fine-tune with a smaller dataset.
   - **Reason**: GPT-2’s architecture supports transfer learning on custom datasets, and its manageable size allows for feasible deployment on most hardware while retaining enough capacity to capture intricate conversational patterns.

### 2. **Intentional Overfitting**
   - **Choice**: The training strategy involves deliberately overfitting the model by running through many epochs (50) with a small batch size (1).
   - **Reason**: Overfitting was used to ensure that the model memorizes the specific conversational style and phrases from the dataset. This approach is suitable for stylistic emulation since we want the model to respond in a precise, personalized manner rather than generalizing across various conversation styles.

### 3. **Training Parameters**
   - **Epochs and Batch Size**: We set `num_train_epochs` to 100 and `per_device_train_batch_size` to 2 to maximize the model’s exposure to each example individually, encouraging memorization.
   - **Learning Rate**: A relatively low learning rate (`5e-5`) was selected to fine-tune gently without disrupting GPT-2’s pre-trained language capabilities.

### 4. **Prompt-Response Data Structuring**
   - **Choice**: Each conversation segment from the parsed Discord data is structured into prompt-response pairs, with all prompts consolidated under the label "Prompt" and responses labeled as "Response".
   - **Reason**: The prompt-response format mirrors conversational exchanges, aligning well with GPT-2’s autoregressive nature, which predicts subsequent tokens based on preceding context. This structure helps the model learn turn-taking dynamics and response styles specific to the individual.

### 5. **Custom Dataset Class**
   - **Choice**: A custom `PromptResponseDataset` class was implemented to handle the specific data structure required for fine-tuning on Discord messages.
   - **Reason**: This class consolidates prompt-response pairs into a single training example with inputs formatted to the max token length, making it suitable for GPT-2’s sequence-based architecture.

### 6. **Attention Mask and Tokenizer Adjustment**
   - **Choice**: Since GPT-2 doesn’t include a default padding token, the end-of-sequence (`eos_token`) was reused as a padding token.
   - **Reason**: This avoids errors related to padding and ensures consistent handling of input lengths, which can be critical for devices like MPS that may interpret padding differently from EOS markers.

## Project Structure

- **GPT2.py**: Main script for loading data, fine-tuning the GPT-2 model, and generating responses.
- **data/**: Directory containing the parsed prompt-response data files.
- **logs/**: Logging directory for monitoring training progress.
- **GPT2_fine_tuned_model/**: Directory to store the final fine-tuned model and tokenizer.
- **raw_data.txt**: NOT INCLUDED-- Text file holding the raw training data of all of Cat Luong and my personal discord messages as well as many messages he sent to larger groups.

## Installation

To set up this project, first clone the repository and install the necessary Python packages.

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt