import re
from datetime import datetime
import json

def parse_discord_messages(file_path):
    parsed_data = []
    current_prompt = ""
    current_response = ""
    messages = []
    current_message = ''

    with open(file_path, 'r') as file:
        for line in file:
            # Detect new message start based on line with sender name and date
            if re.match(r".*? — \d{2}/\d{2}/\d{4} \d{1,2}:\d{2} (AM|PM)", line):
                if current_message:
                    # Save the previous complete message
                    messages.append(current_message.strip())
                # Start new message block
                current_message = line
            else:
                # Continue appending lines for multiline messages
                current_message += line

        # Add the last message if it exists
        if current_message:
            messages.append(current_message.strip())

    # Process each complete message
    for message in messages:
        match = re.match(r"(.*?) — (\d{2}/\d{2}/\d{4} \d{1,2}:\d{2} (AM|PM))\n(.+)", message, re.DOTALL)
        if match:
            sender = match.group(1).strip()
            date_str = match.group(2).strip()
            content = match.group(4).strip()

            # Parse the date to verify formatting, if needed
            datetime.strptime(date_str, "%m/%d/%Y %I:%M %p")

            # Check if the current message is from "CrazyMacAroNi"
            if sender in ["CrazyMacAroNi", "Cat", "Cat Luong"]:
                # If CrazyMacAroNi responds, consider it as a response to the previous messages
                if current_prompt:
                    current_response = content
                    parsed_data.append({"prompt": current_prompt.strip(), "response": current_response.strip()})
                    current_prompt = ""  # Reset the prompt after capturing the prompt-response pair
            else:
                # If it's from someone else, add it to the current prompt
                current_prompt += f"{sender}: {content}\n"

    return parsed_data



# Converts discord messages into prompt-response format given the input data is in the following format
"""Cat — 04/05/2024 3:39 PM
Wait daniel the cincyhacks thingy you told me about is tomorrow right?
Daniel Vennemeyer — 04/05/2024 3:40 PM
Yes, tomorrow and Sunday
Cat — 04/05/2024 3:40 PM
Rai and Arnav already on it so I'm not needed anymore right?
Daniel Vennemeyer — 04/05/2024 3:41 PM
Yep, unless you want to"""

parsed_data = parse_discord_messages("raw_data.txt")

# Display the parsed prompt-response pairs
with open("formatted.json", "w") as outfile:
    json.dump(parsed_data, outfile)