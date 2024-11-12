import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model_path = "./GPT2_fine_tuned_model"  # Path to your saved fine-tuned model
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set pad token if necessary
tokenizer.pad_token = tokenizer.eos_token

# Function to generate response from the model
def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    response_ids = model.generate(
        input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Streamlit App Layout
st.title("Discord-Style CAT-bot")
st.write("A chatbot fine-tuned to respond in the style of Cat Luong.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input box
user_input = st.text_input("You:", placeholder="Type a message and press Enter...")

# Generate response if user submits input
if user_input:
    # Generate and display response
    bot_response = generate_response(user_input)
    for item in [user_input, "Rai:", "Maerose:", "Daniel Vennemeyer:", "Sp3ial person(Anay)():", "/.BeastieNate5:", "Vishesh Parwani:", "aaronwho:"]:
        bot_response = bot_response.replace(item, "")
    # Append user and bot responses to the chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("CatGPT", bot_response))

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.write(f"**{speaker}:** {message}")
    else:
        st.write(f"**{speaker}:** {message}")

# Button to clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []