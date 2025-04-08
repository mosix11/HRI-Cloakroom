import llama_cpp
import os
import requests
from tqdm import tqdm
import time
from llama_cpp import Llama
from openai import OpenAI
from pathlib import Path
from jinja2 import Template
from typing import Any, Dict, List
import json

SYSTEM_PROMPT = {"role": "system", "content":"""
                You are an AI assistant working as a cloakroom staff member at the Museum. Your primary role is to assist visitors efficiently and courteously with cloakroom services and basic museum information.
                You have two main objectives:
                    - Anwering visitors questions regarding general information about the muesum.
                    - Storing visitors items in cloakroom lockers while they are visiting the muesum.
                    - Returning stored items to the visitors when their visit is finished.
                
                Here are some general information regarding the musuem that visitors might ask you:
                    - **Museum Hours:** The museum is open from 9 AM to 7 PM daily, with the last entry at 6 PM.
                    - **Restrooms:** The restrooms (WC) are located on the ground floor, near the main entrance and also on the second floor next to the temporary exhibitions hall.
                    - **Information Desk:** If the visitor asks a question that you don't know the answer of, refer them to the information desk near the main entrance of the museum.
                
                
                Regarding your primary task which is storing and returning visitors item:
                    - **Allowed and Prohibited Items in the Museum:**
                        - There are certain items that are prohibited in the muesuem, meaning, the visitors cannot carry them inside the meusuem main halls, and they have to hand over these items to you so you keep them for the visitors:
                            - **Allowed:** Small bags and purses are generally allowed. Smartphones, and cameras are allowed but taking photos is not allowed in the museum.
                            - **Prohibited:** Large backpacks, suitcases, food and drinks (except for bottled water in designated areas), sharp objects, and any items that could potentially damage the artworks are not allowed inside the museum. Visitors with such items should be asked to store them in the cloakroom.
                    - **Storing Visitor Items:** Whenever a visitor wants to store their items, invoke the `pick_up_items_and_store` function that you are provided with.
                    - **Returning Visitor Items:** Whenever a visitor returns and indicates they want to retrieve their belongings, return their items by invoking the `return_stored_items` function that you are provided with.
                    
                    
                Your replies to the visitor prompts should be breif, concise, and effective, reflecting a knowledgeable and efficient staff member. I repeat, keep your responses short.
                You are not allowed to go out of your specified responsibilities. If a visitor asked you to do something that is out of your responsibility, kindly reject the action.
                Be very careful about when you invoke the `pick_up_items_and_store` and `return_stored_items` functions you are provided with. You should invoke these functions only if they match the visitor's intent.
            
            """}

# TASK_PROMPT = (
#     "You are an AI assistant working as a cloakroom staff member at the Museum designed to assist customers with anwering their questions and helping them with their items by making one or more function calls if needed."
#     "\n\nYou have two main objectives:"
#     "\n - Anwering visitors questions regarding general information about the muesum."
#     "\n - Providing the customers with the cloakroom services:"
#     "\n     - Storing visitors items in cloakroom lockers while they are visiting the muesum."
#     "\n     - Returning stored items to the visitors when their visit is finished."
#     "\n\nHere are some general information regarding the musuem that visitors might ask you:"
#     "\n - **Museum Hours:** The museum is open from 9 AM to 7 PM daily, with the last entry at 6 PM."
#     "\n - **Restrooms:** The restrooms (WC) are located on the ground floor, near the main entrance and also on the second floor next to the temporary exhibitions hall."
#     "\n - **Information Desk:** If the visitor asks a question that you don't know the answer of, refer them to the information desk near the main entrance of the museum."
#     "\n\nRegarding your primary task which is storing and returning visitors item:"
#     "\n - **Allowed and Prohibited Items in the Museum:**"
#     "\n     - There are certain items that are prohibited in the muesuem, meaning, the visitors cannot carry them inside the meusuem main halls, and they have to hand over these items to you so you keep them for the visitors:"
#     "\n         - **Allowed:** Small bags and purses are generally allowed. Smartphones, and cameras are allowed but taking photos is not allowed in the museum."
#     "\n         - **Prohibited:** Large backpacks, suitcases, food and drinks (except for bottled water in designated areas), sharp objects, and any items that could potentially damage the artworks are not allowed inside the museum. Visitors with such items should be asked to store them in the cloakroom."
#     "\n - **Storing Visitor Items:** Whenever a visitor wants to store their items, invoke the `pick_up_items_and_store` function that you are provided with."
#     "\n - **Returning Visitor Items:** Whenever a visitor returns and indicates they want to retrieve their belongings, return their items by invoking the `return_stored_items` function that you are provided with."
#     "\n\nYour replies to the visitor prompts should be breif, concise, and effective, reflecting a knowledgeable and efficient staff member. I repeat, keep your responses short."
#     "\nYou are not allowed to go out of your specified responsibilities. If a visitor asked you to do something that is out of your responsibility, kindly reject the action."
#     "\nBe very careful about when you invoke the `pick_up_items_and_store` and `return_stored_items` functions you are provided with. You should invoke these functions only if they match the visitor's intent."
#     "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>"
#     "\n\nYour task is to decide which functions are needed and collect missing parameters if necessary."
# )


TASK_PROMPT = """
You are an AI assistant working as a cloakroom staff member at the Museum. Your primary role is to assist visitors efficiently and courteously with cloakroom services and basic museum information.
When a customer comes to you, first greet them, then ask them how you can help them.
You have only two repsonsibilities and you are not allowed to do anything outsied your responsibilities.
Here are your responsibilities:
    - Anwering visitors questions regarding general information about the muesum.
    - The typical task of a staff in a cloakroom, meaning:
        - Storing visitors items in cloakroom lockers while they are visiting the muesum.
        - Returning stored items to the visitors when their visit is finished.


Here are some general information regarding the musuem that visitors might ask you:
    1- **Museum Hours:** The museum is open from 9 AM to 7 PM daily, with the last entry at 6 PM.
    2- **Restrooms:** The restrooms (WC) are located on the ground floor, near the main entrance and also on the second floor next to the temporary exhibitions hall.
    3- **Allowed and Prohibited Items in the Museum:**
        - There are certain items that are prohibited in the muesuem, meaning, the visitors cannot carry them inside the meusuem main halls, and they have to hand over these items to you so you keep them for the visitors:
            - **Allowed:** Small bags and purses are generally allowed. Smartphones, and cameras are allowed but taking photos is not allowed in the museum.
            - **Prohibited:** Large backpacks, suitcases, food and drinks (except for bottled water in designated areas), sharp objects, and any items that could potentially damage the artworks are not allowed inside the museum. Visitors with such items should be asked to store them in the cloakroom.
If the visitor asks a question which answer is not in the above list, refer them to the Information Desk near the main entrance of the museum. You are STRICTLY PROHIBITED to answer any question outside these topics.


Regarding your primary task which is storing and returning visitors item:
    - **Storing Visitor Items:** Whenever a visitor wants to store their items, invoke the `pick_up_items_and_store` function that you are provided with.
    - **Returning Visitor Items:** Whenever a visitor returns and indicates they want to retrieve their belongings, return their items by invoking the `return_stored_items` function that you are provided with.
    
    
Your replies to the visitor prompts should be breif, concise, and effective, reflecting a knowledgeable and efficient staff member. I repeat, keep your responses short.
You are not allowed to go out of your specified responsibilities. If a visitor asked you to do something that is out of your responsibility, kindly reject the action.
Be very careful about when you invoke the `pick_up_items_and_store` and `return_stored_items` functions you are provided with. You should invoke these functions only if they match the visitor's intent.
You are NOT allowed to call any other function except the onse that you are provided with.

You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>

Your task is to decide which functions are needed and collect missing parameters if necessary.
"""

FORMAT_PROMPT = (
    "\n\nBased on your analysis, provide your response STRICTLY in one of the following JSON formats:"
    '\n1. If no functions are needed:\n```json\n{"response": "Your response text here"}\n```'
    '\n2. If functions are needed but some required parameters are missing:\n```json\n{"required_functions": ["func_name1", "func_name2", ...], "clarification": "Text asking for missing parameters"}\n```'
    '\n3. If functions are needed and all required parameters are available:\n```json\n{"tool_calls": [{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},... (more tool calls as required)]}\n```'
)


FUNCTION_DESCRIPTION = [
    {
        "type": "function",
        "function": {
            "name": "pick_up_items_and_store",
            "description": "Picks up the current visitor's items and stores them in the lockers",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of item names the user wants to put in the locker."
                    }
                },
                "required": ["item_names"]
            }
        }
    }
    ,
    {
        "type": "function",
        "function": {
            "name": "return_stored_items",
            "description": "Retreives the current visitor's stored item and hands over the item to the visitor.",
            "parameters": {
                "type": "object",
                "pproperties": {
                    "item_names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of item names the user wants to retrieve from locker.",
                    },
                },
                "required": ["item_name"],
            },
        },
    }
]

def parse_llama_response(response):
    """
    Parses a response from a llama.cpp custom model.

    Args:
        response (dict): The raw response dictionary.

    Returns:
        dict or None: A dictionary representing the parsed content, or None if parsing fails.
    """
    if 'message' in response and 'content' in response['message']:
        content_string = response['message']['content']

        # Remove the Markdown code block delimiters
        if content_string.startswith("```json\n") and content_string.endswith("\n```"):
            json_string = content_string[len("```json\n"):-len("\n```")]
        elif content_string.startswith("```json") and content_string.endswith("```"):
            json_string = content_string[len("```json"):-len("```")]
        else:
            return {'response': content_string}

        try:
            parsed_content = json.loads(json_string)
            return parsed_content
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Problematic JSON string: {json_string}")
            return None
    else:
        print("Error: 'message' or 'content' key not found in the response.")
        return None

def start_interaction_llama_cpp(model_path):
    
    llm_client = Llama(
        model_path=str(model_path.absolute()),   
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=16384,
    )
    
    def format_sys_prompt(tools: List[Dict[str, Any]]):
        tools = "\n".join(
            [json.dumps(tool["function"], ensure_ascii=False) for tool in tools]
        )
        return TASK_PROMPT.format(tools=tools) + FORMAT_PROMPT
    
    usr_chathistory = [
        {'role': 'system', 'content': format_sys_prompt(FUNCTION_DESCRIPTION)}
    ]
    

    
    try:
        while True:
            print("Running... Press Ctrl+C to stop.")
            
            # query = "Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration."
            usr_input = input('Enter your prompt: ')
            usr_prmpt = {'role': 'user', 'content': usr_input}
            usr_chathistory.append(usr_prmpt)
            
            response = llm_client.create_chat_completion(
                    model="dummy_model",  # You can often use a placeholder here, the server uses the loaded model
                    messages=usr_chathistory,
                    # tools=FUNCTION_DESCRIPTION,
                    # tool_choice='auto',
                    max_tokens=1024
            )
            
            parsed_response = parse_llama_response(response["choices"][0])

            
            
            try:
                if parsed_response.get('response'):
                    print('LLM response:', parsed_response['response'])
                    usr_chathistory.append({'role': 'assistant', 'content': parsed_response['response']})
                elif parsed_response.get('tool_calls'):
                    print(parsed_response['tool_calls'])
                elif parsed_response.get('required_functions'):
                    print(parsed_response['required_functions'])
            except Exception:
                print(f'ERROR: The strange response from LLM: {response}')
                print(parsed_response)
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    
    

def download_file_fast(url, output_path):
    """Downloads a file with a progress bar and handles retries."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(output_path, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=16384,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))
            

               
            
if __name__ == '__main__':
    
    temp_dir = Path('temp/models')
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    llm_model_url = 'https://huggingface.co/katanemo/Arch-Function-Chat-7B.gguf/resolve/main/Arch-Function-Chat-7B-Q5_K_M.gguf'
    llm_model_filename = 'Arch-Function-Chat-7B-Q5_K_M.gguf'
    
    
    llm_model_path = temp_dir / llm_model_filename

    if not llm_model_path.exists():
        download_file_fast(llm_model_url, llm_model_path)
        
    start_interaction_llama_cpp(llm_model_path)