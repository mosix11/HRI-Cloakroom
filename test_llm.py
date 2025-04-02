import llama_cpp
# import langchain
# print("Llama-Cpp Version:", llama_cpp.__version__)
# print("LangChain Version:", langchain.__version__)

# pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=on"

import os
import requests
from tqdm import tqdm
import time
from llama_cpp import Llama
from openai import OpenAI
from pathlib import Path
from jinja2 import Template

# import gc
# import torch

# gc.collect()
# torch.cuda.empty_cache()

# Constants
# model_url = "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q6_K_L.gguf"
# model_filename = "microsoft_Phi-4-mini-instruct-Q6_K_L.gguf"




# def download_file_fast(url, output_path):
#     """Downloads a file with a progress bar and handles retries."""
#     headers = {"User-Agent": "Mozilla/5.0"}
#     response = requests.get(url, stream=True, headers=headers)
#     total_size = int(response.headers.get("content-length", 0))
#     chunk_size = 1024 * 1024  # 1MB chunks

#     with open(output_path, "wb") as file, tqdm(
#         desc=f"Downloading {os.path.basename(output_path)}",
#         total=total_size,
#         unit="B",
#         unit_scale=True,
#         unit_divisor=16384,
#     ) as bar:
#         for data in response.iter_content(chunk_size=chunk_size):
#             file.write(data)
#             bar.update(len(data))

# Download model if not exists
# if not os.path.exists(model_path):
#     print("Model file not found. Downloading...")
#     download_file_fast(model_url, model_path)
#     print("Download complete.")

# Load model with llama-cpp
# print("Loading model...")
# llm = Llama(
#     model_path=model_path,   
#     n_gpu_layers=-1,
#     verbose=False,
#     n_ctx=4096
# )  # adjust context window if needed


# def busy_wait(seconds):
#     end_time = time.perf_counter() + seconds
#     while time.perf_counter() < end_time:
#         pass  # Keep the CPU busy

# # Example: busy wait for 2 seconds
# busy_wait(10)

# # Run a sample prompt
# output = llm("Q: What is the capital of France?\nA:", max_tokens=32, stop=["Q:", "\n"])
# print(output["choices"][0]["text"].strip())


# pip install huggingface-hub

# llm = Llama.from_pretrained(
#     repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
#     filename="*q8_0.gguf",
#     verbose=False
# )


# FUNCTION_DESCRIPTIONS = [
#             {
#             "type": "function",
#             "function": {
#                 "name": "get_current_weather",
#                 "description": "Get the current weather in a given city.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "city_name": {
#                             "type": "string",
#                             "description": "Name of the city we want to know its weather.",
#                         }
#                     },
#                     "required": ['city_name']
#                 }
#             }
#         },
# ]

# FUNCTION_DESCRIPTIONS = [
#     {
#         "name": "get_current_weather",
#         "description": "Retrieve the current weather in a given city.",
#         "arguments": {
#             "type": "dict",
#             "properties": {
#                 "city_name": {
#                     "type": "string",
#                     "description": "Name of the city we want to know its weather.",
#                 }
#             },
#             "required": ['city_name']
#         }
#     }
# ]

FUNCTION_DESCRIPTIONS = [
    {
        "name": "pick_up_items_and_store",
        "description": "Picks up the current visitor's items and stores them in the lockers",
        "arguments": {
            "type": "dict",
            "properties": {
                # "object_descriptions": {
                #     "type": "array",
                #     "description": "A list of item descriptions (given by the user) being picked up.",
                #     "items": {
                #         "type": "string"
                #     }
                # }
            },
            "required": []       
        }
    },
    {
        "name": "return_stored_items",
        "description": "Retreives the current visitor's stored items and hands over these items to the visitor.",
        "arguments": {
            "type": "dict",
            "properties": {},
            "required": []
        }
    }
]

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

# system_prompt= f"""
# You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
# If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
# You should only return the function call in tools call sections.

# If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
# You SHOULD NOT include any other text in the response.
# Here is a list of functions in JSON format that you can invoke.\n{tools}\n
# """


from jinja2 import Environment, FileSystemLoader
import json
import ast


def extract_llm_message(raw_text):
    # Remove "assistant\n\n" if present
    # prefix = "assistant\n\n"
    # # print(raw_text)
    # if raw_text.startswith(prefix):
    #     message_part = raw_text[len(prefix):]
    # else:
    #     message_part = raw_text

    # # Try to parse if it's a quoted literal string
    # try:
    #     message = ast.literal_eval(message_part)
    # except (ValueError, SyntaxError):
    #     print('EERRRRROOORRR:')
    #     print(ValueError)
    #     print(SyntaxError)
    #     # Fallback: treat as raw text (e.g., function call, command, malformed string)
    #     message = message_part.strip()

    # return message
    prefix1 = "assistant\n\n"
    prefix2 = "assistant\n"
    text = raw_text.replace(prefix1, '')
    text = text.replace(prefix2, '')
    
    if raw_text.startswith("\'"):
        text = text[1:]
    if raw_text.endswith("\'"):
        text = text[:-1]
    if text == '':
        print(raw_text)
        raise RuntimeError('Empty text outputted by the model!')
    return text

def parse_llm_response(response, usr_chathistory):
    message = extract_llm_message(response["choices"][0]['text'])
    function_call = False
    if 'pick_up_items_and_store' in message.lower():
        print('pick_up_items_and_store() called')
        usr_chathistory.append(
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'function': {
                            'name': 'pick_up_items_and_store',
                            'arguments': {}
                        }
                    }
                ]
            }
        )
        
        function_response = {
            'success': True,
            'message': 'Successfully stored the visitors items in the locker.'
        }
        
        usr_chathistory.append(
            {
                'role': 'tool',
                'content': function_response
            }
        )
        
        function_call = True
        
    elif 'return_stored_items' in message.lower():
        print('return_stored_items() called')
        usr_chathistory.append(
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'function': {
                            'name': 'return_stored_items',
                            'arguments': {}
                        }
                    }
                ]
            }
        )
        
        function_response = {
            'success': True,
            'message': 'Successfully returned the items to the visitor.'
        }
        usr_chathistory.append(
            {
                'role': 'tool',
                'content': function_response
            }
        )
        
        function_call = True
        
    else:
        print('LLM response:', message)
        usr_chathistory.append({'role': 'assistant', 'content': message})
    
    return usr_chathistory, function_call


def start_interaction_llama_cpp(model_path):
    

    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['tojson'] = lambda val, **kwargs: json.dumps(val, **kwargs)
    template = env.get_template('watt.jinja')


    
    llm_client = Llama(
        model_path=str(model_path.absolute()),   
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=16384,
        
    )

    # usr_chathistory = [
    #     {'role': 'system', 'content': """
    #     You are a helpful AI assitant.
        
    #     """}

    # ]
    
    usr_chathistory = [
        {'role': 'system', 'content': SYSTEM_PROMPT}
    ]
    

    
    try:
        while True:
            print("Running... Press Ctrl+C to stop.")
            
            # query = "Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration."
            usr_input = input('Enter your prompt: ')
            usr_prmpt = {'role': 'user', 'content': usr_input}
            usr_chathistory.append(usr_prmpt)
            
            prompt = template.render(messages=usr_chathistory, tools=FUNCTION_DESCRIPTIONS, tools_in_user_message=False, date_string="31 Mar 2025")
            # print('\n\n\n\n', prompt, '\n\n\n\n')
            response = llm_client(prompt=prompt, max_tokens=512)
            
            usr_chathistory, function_call = parse_llm_response(response, usr_chathistory)
            
            if function_call:
                prompt = template.render(messages=usr_chathistory, tools=FUNCTION_DESCRIPTIONS, tools_in_user_message=False, add_generation_prompt=True, date_string="31 Mar 2025")
                # print('\n\n\n\n', prompt, '\n\n\n\n')
                response = llm_client(prompt=prompt, max_tokens=512)
                usr_chathistory, function_call = parse_llm_response(response, usr_chathistory)
                
                if function_call:
                    print('EEERRRRROOOOOORRRRRR: The model called two functions consecutively!')

            # print(response)            

            # Safely parse the quoted message
            
            # print(message)
            
            
            
            # print(response)
            # response = llm_client.create_chat_completion(
            #         model="dummy_model",  # You can often use a placeholder here, the server uses the loaded model
            #         messages=usr_chathistory,
            #         tools=FUNCTION_DESCRIPTIONS,
            #         tool_choice='auto',
            #         max_tokens=512
            # )
            # print(response)
            # llm_response = response['choices'][0]['message']

            # if llm_response.tool_calls:
            #     print(llm_response.tool_calls)
            # elif llm_response.content:
            #     llm_message = response.choices[0].message.content
            #     print('LLM response:', llm_message)
            #     usr_chathistory.append({'role': 'assistant', 'content': llm_message})
            # else:
            #     print(f'ERROR: The strange response from LLM: {llm_response}')
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    

def start_interaction_OpenAI():
    llm_client = OpenAI(
        base_url='http://127.0.0.1:9585/v1',
        api_key="dummy_key"
    )




    usr_chathistory = [
        {'role': 'system', 'content': """
        You are a helpful AI assitant.
        
        You're provided with a couple of functions which you have to use when you are asked about things that you do not know in that context.
        Do not provide answers with your own knowledge in the context that you can use functions.
        I repeat, do not use your own knowledge. You have to use function calls if they relate to user's query.
        For information regarding weather, user the function `get_current_weather`.
        
        """}
    ]

    try:
        while True:
            print("Running... Press Ctrl+C to stop.")
            
            usr_input = input('Enter your prompt: ')
            usr_prmpt = {'role': 'user', 'content': usr_input}
            usr_chathistory.append(usr_prmpt)
            
            response = llm_client.chat.completions.create(
                    model="dummy_model",  # You can often use a placeholder here, the server uses the loaded model
                    messages=usr_chathistory,
                    tools=FUNCTION_DESCRIPTIONS,
                    tool_choice='auto',
                    max_tokens=512
            )
            llm_response = response.choices[0].message

            if llm_response.tool_calls:
                print(llm_response.tool_calls)
            elif llm_response.content:
                llm_message = response.choices[0].message.content
                print('LLM response:', llm_message)
                usr_chathistory.append({'role': 'assistant', 'content': llm_message})
            else:
                print(f'ERROR: The strange response from LLM: {llm_response}')
            
    except KeyboardInterrupt:
        print("\nStopped by user.")


from llama_cpp.llama_tokenizer import LlamaHFTokenizer

def start_interaction_functionary(model_path):
    Llama.from_pretrained(
        repo_id="meetkai/functionary-small-v2.4-GGUF",
        filename="functionary-small-v2.4.Q4_0.gguf",
        chat_format="functionary-v2",
        tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.4-GGUF"),
        n_gpu_layers=-1
    )

    llm_client = Llama(
        model_path=str(model_path.absolute()),   
        n_gpu_layers=-1,
        verbose=False,
        n_ctx=16384,
    )

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
    base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
    interaction_pkg_dir = base_dir / Path('interaction')
    interaction_pkg_dir.mkdir(exist_ok=True, parents=True)

    # llm_model_url = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q6_K.gguf"
    # llm_model_filename = "unsloth_Phi-4-mini-instruct-Q6_K.gguf"
    
    # llm_model_url = 'https://huggingface.co/Mungert/functionary-v4r-small-preview-GGUF/resolve/main/functionary-v4r-small-preview-q4_k_m.gguf'
    # llm_model_filename = 'Mungert_functionary-v4r-small-preview-q4_k_m.gguf'
    
    # llm_model_url = 'https://huggingface.co/mradermacher/watt-tool-8B-GGUF/resolve/main/watt-tool-8B.Q4_K_M.gguf'
    # llm_model_filename = 'mradermacher_watt-tool-8B.Q4_K_M.gguf'
    
    llm_model_url = 'https://huggingface.co/eaddario/Watt-Tool-8B-GGUF/resolve/main/Watt-Tool-8B-Q4_K_M.gguf'
    llm_model_filename = 'eaddario_Watt-Tool-8B-Q4_K_M.gguf'

    llm_model_path = interaction_pkg_dir / Path('models') / llm_model_filename
    llm_model_path.parent.mkdir(exist_ok=True, parents=True)

    if not llm_model_path.exists():
        download_file_fast(llm_model_url, llm_model_path)
        
    start_interaction_llama_cpp(llm_model_path)
    'python3 -m llama_cpp.server --host 127.0.0.1 --port 9585 --model ~/.hri_cloakroom/interaction/models/unsloth_Phi-4-mini-instruct-Q6_K.gguf --n_gpu_layers -1 --n_ctx 16384 --verbose False'
    'python3 -m llama_cpp.server --host 127.0.0.1 --port 9585 --model ~/.hri_cloakroom/interaction/models/Mungert_functionary-small-v3.2-q4_k_m.gguf --n_gpu_layers -1 --n_ctx 16384 --verbose False'