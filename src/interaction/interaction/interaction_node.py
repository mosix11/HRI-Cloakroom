#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from interfaces.msg import ASRTranscript, VisionInfo, DetectedFace, DetectedObject, BoundingBox2D
from interfaces.action import Speak
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, String

import llama_cpp
from llama_cpp import Llama
from openai import OpenAI
import lmdb
import os
import json
import base64

from pathlib import Path
import requests
import pickle
from tqdm import tqdm
from dotenv import load_dotenv


from jinja2 import Environment, FileSystemLoader
import ast

load_dotenv()


class InteractionNode(Node):
    
    def __init__(self):
        super().__init__('interaction_node')
        self.get_logger().info("Starting Interaction Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        interaction_pkg_dir = base_dir / Path('interaction')
        interaction_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        # llm_model_url = 'https://huggingface.co/mradermacher/watt-tool-8B-GGUF/resolve/main/watt-tool-8B.Q4_K_M.gguf'
        # llm_model_filename = 'mradermacher_watt-tool-8B.Q4_K_M.gguf'
        
        llm_model_url = 'https://huggingface.co/eaddario/Watt-Tool-8B-GGUF/resolve/main/Watt-Tool-8B-Q4_K_M.gguf'
        llm_model_filename = 'eaddario_Watt-Tool-8B-Q4_K_M.gguf'
        llm_model_path = interaction_pkg_dir / Path('models') / llm_model_filename
        llm_model_path.parent.mkdir(exist_ok=True, parents=True)
        
        if not llm_model_path.exists():
            self.download_file_fast(llm_model_url, llm_model_path)
            
        template_dir = interaction_pkg_dir / Path('models')
        env = Environment(loader=FileSystemLoader(template_dir))
        env.filters['tojson'] = lambda val, **kwargs: json.dumps(val, **kwargs)
        self.prmpt_template = env.get_template('watt.jinja')
            
        self.llm_client = Llama(
            model_path=str(llm_model_path.absolute()),   
            n_gpu_layers=-1,
            verbose=False,
            n_ctx=16384,
            
        )
        
        self.db_env = self.initialize_db(interaction_pkg_dir)
        
        # self.llm_client = OpenAI(
        #     base_url='http://127.0.0.1:9585/v1',
        #     api_key="dummy_key"
        # )

        self._TTS_action_client = ActionClient(self, Speak, 'interaction/speak')
        
        self.create_subscription(
            ASRTranscript,
            'audio/transcription',
            self.transcription_callback,
            10
        )

        self.create_subscription(
            VisionInfo,
            'vision/vision_info',
            self.vision_callback,
            10
        )
        
        self.SPEAKING_LOCK = False
        self.LLM_INFERENCE_LOCK = False
        self.PERFORMING_ACTION_LOCK = False
        
        self.current_detected_faces = []
        self.current_detected_objects = []
        
        self.SYSTEM_PROMPT = {"role": "system", "content":"""
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
        

        
        self.FUNCTION_DESCRIPTIONS = [
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
        # self.test_db(self.db_env, 'test_user')
        
    
    # def test_db(self, db_env, user_id):
    #     self.update_chat_history(db_env, user_id, {"role": "user", "content": "Hello, are you available?"})
    #     self.update_chat_history(db_env, user_id, {"role": "assistant", "content": "Yes, how can I help you?"})

    #     # For demonstration purposes, create a dummy CompressedImage.
    #     # In a real scenario, this would come from a ROS 2 subscriber callback.
    #     dummy_image = CompressedImage()
    #     dummy_image.header = Header()
    #     # Set dummy timestamp values (these should come from your sensor_msgs.msg.Time or builtin_interfaces.msg.Time)
    #     dummy_image.header.stamp.sec = 161803398
    #     dummy_image.header.stamp.nanosec = 314159265
    #     dummy_image.header.frame_id = "camera_frame"
    #     dummy_image.format = "jpeg"
    #     dummy_image.data = b'\xff\xd8\xff\xe0'  # Example JPEG header bytes

    #     # Append the image to the user's record
    #     self.add_user_image(db_env, user_id, dummy_image)

    #     # Retrieve and print the updated record with chat history and images
    #     updated_record = self.get_user_record(db_env, user_id)
    #     self.get_logger().info(f"Updated record: {updated_record}")
        
    def _can_listen(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)
    
    def _can_see(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)

    def _can_speak(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)
    
    def _perform_action(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)
        
        
    def vision_callback(self, msg: VisionInfo):
        self.get_logger().info(f"Received Vision Info: {len(msg.faces)} Faces and {len(msg.objects)} Objects")
        if self._can_see():
            self.current_detected_faces = msg.faces
            self.current_detected_objects = msg.objects

    def transcription_callback(self, msg: ASRTranscript):
        self.get_logger().info(f"Received ASR Transcript: '{msg.transcript}' (datetime: {msg.timestamp})")
        if self._can_listen():
            current_prmt = msg.transcript
            current_prmt_dt = msg.timestamp
        
            if len(self.current_detected_faces) == 0:
                self.get_logger().warning('No user is currently in the frame so the robot cannot associate the transcribed speech to a user!')
                return
            
            
            self.LLM_INFERENCE_LOCK = True
            
            # TODO select the user based on `talking` status
            # For now we use the first person in the list
            curr_user = self.current_detected_faces[0]
            # TODO only select the objects belonging to the user
            # For now we use all the objects appearing in the scene
            user_objs = self.current_detected_objects
            self.prompt_llm(current_prmt, current_prmt_dt, curr_user, user_objs)
        else:
            if self.SPEAKING_LOCK:
                self.get_logger().warning("System is currently applying TTS.")
            if self.LLM_INFERENCE_LOCK:
                self.get_logger().warning("System is currently processing the conversation.")
        # self.send_TTS_goal(msg.transcript)

    
    def perform_pick_action(self, objects):
        self.get_logger().log('Performing pick action ...')
        self.PERFORMING_ACTION_LOCK = True
        # TODO implement action
        
        if len(objects) == 0:
            success = False
            message = 'There is no item in robot\'s vision range. Please bring your item\'s to closer so the robot can see it.'
        else:
            success = True        
            message = 'Successfully stored the visitors items in the locker.'
            
        self.PERFORMING_ACTION_LOCK = False
        return {'sucsess': success, 'message': message}
    
    def perform_return_action(self, belongings):
        self.get_logger().log('Performing return action ...')
        self.PERFORMING_ACTION_LOCK = True
        # TODO implement action
        
        if len(belongings) == 0:
            success = False
            message = 'The user has no items stored in the lockers.'
        else:
            success = True
            message = 'Successfully returned the items to the visitor.'

        self.PERFORMING_ACTION_LOCK = False
        return {'sucsess': success, 'message': message}
    
    
    def extract_llm_message(self, raw_text):
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
    
    def parse_llm_response_and_perform_actions(self, response, usr_chathistory, curr_objs, usr_belongings):
        message = self.extract_llm_message(response["choices"][0]['text'])
        function_call = False
        if 'pick_up_items_and_store' in message.lower():
            self.LLM_INFERENCE_LOCK = False
            self.get_logger().log('pick_up_items_and_store() called')
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
            
            function_response = self.perform_pick_action(curr_objs)
            
            usr_chathistory.append(
                {
                    'role': 'tool',
                    'content': function_response
                }
            )
            
            function_call = True
            
        elif 'return_stored_items' in message.lower():
            self.LLM_INFERENCE_LOCK = False
            self.get_logger().log('return_stored_items() called')
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
            
            function_response = self.perform_return_action(usr_belongings)
            
            usr_chathistory.append(
                {
                    'role': 'tool',
                    'content': function_response
                }
            )
            
            function_call = True
            
        else:
            self.LLM_INFERENCE_LOCK = False
            self.get_logger().log('LLM response:', message)
            usr_chathistory.append({'role': 'assistant', 'content': message})
        
        return usr_chathistory, function_call

    def prompt_llm(self, prmpt, prmpt_dt, user, curr_objs):
        usr_id = user.user_id
        usr_gender = 'Male' if user.gender == 1 else 'Female'
        usr_record = self.get_user_record(self.db_env, usr_id)
        usr_chathistory = usr_record['chat_history']
        usr_belongings = usr_record['obj_imgs']
        
        usr_prmpt = {
                'role': 'user',
                'content': prmpt
            }
        
        if len(usr_chathistory) == 0:
            self.update_chat_history(self.db_env, usr_id, messages=[self.SYSTEM_PROMPT])
            usr_chathistory.extend([self.SYSTEM_PROMPT])
        
        usr_chathistory.append(usr_prmpt)
        
        try:
            prompt = self.prmpt_template.render(messages=usr_chathistory, tools=self.FUNCTION_DESCRIPTIONS, tools_in_user_message=False, date_string="31 Mar 2025")
            response = self.llm_client(prompt=prompt, max_tokens=512)
            
            usr_chathistory, function_call = self.parse_llm_response_and_perform_actions(response, usr_chathistory, curr_objs, usr_belongings)
            
            if function_call:
                self.LLM_INFERENCE_LOCK = True
                prompt =  self.prmpt_template.render(messages=usr_chathistory, tools=self.FUNCTION_DESCRIPTIONS, tools_in_user_message=False, add_generation_prompt=True, date_string="31 Mar 2025")
                # print('\n\n\n\n', prompt, '\n\n\n\n')
                response = self.llm_client(prompt=prompt, max_tokens=512)
                usr_chathistory, function_call = self.parse_llm_response_and_perform_actions(response, usr_chathistory)
                
                if function_call:
                    print('EEERRRRROOOOOORRRRRR: The model called two functions consecutively!')
                
            self.send_TTS_goal(usr_chathistory[-1]['content'])

        except Exception as e:
            self.get_logger().error(f"An error occurred: {e}")
        finally:
            self.LLM_INFERENCE_LOCK = False
        
        # try:
        #     response = self.llm_client.chat.completions.create(
        #         model="dummy_model",  # You can often use a placeholder here, the server uses the loaded model
        #         messages=usr_chathistory,
        #         tools=self.FUNCTION_DESCRIPTIONS,
        #         tool_choice="auto",
        #         max_tokens=256
        #     )
            
        #     # self.get_logger().info(response.choices[0].message.content)
        #     llm_response = response.choices[0].message

        #     if llm_response.tool_calls:
        #         for tool_call in llm_response.tool_calls:
        #             if tool_call["type"] == "function":
        #                 function_name = tool_call["function"]["name"]
        #                 function_args_str = tool_call["function"]["arguments"]
        #                 try:
        #                     function_args = json.loads(function_args_str)
        #                     self.get_logger().info(f"LLM wants to call function: {function_name} with arguments: {function_args}")
        #                     # self.execute_function_call(user_id, function_name, function_args)
                            
        #                     self.LLM_INFERENCE_LOCK = False
        #                     if function_name == 'pick_up_objects':
        #                         result = self.perform_pick_action(curr_objs)
        #                     elif function_name == 'return_objects':
        #                         result = self.perform_return_action(usr_belongings)
        #                     else:
        #                         self.get_logger().error(f'The function `{function_name}` is an invalid funciton!')
                                
        #                     print(result)
        #                     # Optionally, you might want to send the result of the function call back to the LLM
        #                     # to inform it about the outcome. This would involve adding another message
        #                     # to the self.user_chat_history with the role "tool_response".
                            
                            
                            
        #                 except json.JSONDecodeError as e:
        #                     self.get_logger().error(f"Failed to decode function arguments: {e}")
        #                     # Optionally, inform the LLM about the error

        #     elif llm_response.content:
        #         llm_text_response = llm_response.content
        #         self.get_logger().info(f"LLM Response: '{llm_text_response}'")
        #         assistant_response = {"role": "assistant", "content": llm_text_response}
        #         self.update_chat_history(self.db_env, usr_id, [usr_prmpt, assistant_response])
        #         usr_chathistory.append(assistant_response)
                
        #         self.LLM_INFERENCE_LOCK = False
                
        #         self.send_TTS_goal(llm_text_response)
        #     else:
        #         self.get_logger().error(f'LLM has output a strange form of response!: {llm_response}')

        # except Exception as e:
        #     self.get_logger().error(f"An error occurred: {e}")
        # finally:
        #     self.LLM_INFERENCE_LOCK = False
        
    def send_TTS_goal(self, text):
        self.SPEAKING_LOCK = True
        action_goal = Speak.Goal()
        action_goal.text = text
        self._TTS_action_client.wait_for_server()
        future = self._TTS_action_client.send_goal_async(action_goal)
        future.add_done_callback(self.TTS_goal_response_callback)
        
    def TTS_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("TTS goal was rejected.")
            self.SPEAKING_LOCK = False
            return
        
        self.get_logger().info("TTS goal accepted, waiting for result...")
        goal_handle.get_result_async().add_done_callback(self.TTS_get_result_callback)
    
    def TTS_get_result_callback(self, future):
        result = future.result().result
        self.SPEAKING_LOCK = False
        if result.success:
            self.get_logger().info("TTS finished playing audio. Resuming processing.")
            # Continue with further actions or processing here.
        else:
            self.get_logger().error("TTS failed to play audio.")
        
        
    def download_file_fast(self, url, output_path):
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
        
        
    def get_user_record(self, db_env, user_id: str):
        """
        Retrieve the record for a given user, which includes both chat history and images.
        If no record exists, returns a default record with empty lists.
        """
        key = user_id.encode('utf-8')
        with db_env.begin() as txn:
            data = txn.get(key)
            if data is None:
                # Return an empty record if it doesn't exist.
                return {"chat_history": [], "obj_imgs": []}
            return json.loads(data.decode('utf-8'))

    def save_user_record(self, db_env, user_id: str, record: dict):
        """
        Save the complete record (chat history and images) for the given user.
        """
        key = user_id.encode('utf-8')
        with db_env.begin(write=True) as txn:
            txn.put(key, json.dumps(record).encode('utf-8'))
            
    def delete_user_record(db_env, user_id: str):
        """
        Delete the user's record (both chat history and images) from the database.
        """
        key = user_id.encode('utf-8')
        with db_env.begin(write=True) as txn:
            txn.delete(key)

    def update_chat_history(self, db_env, user_id: str, messages: dict):
        """
        Append a new message to the user's chat history.
        
        Example message:
        {"role": "user", "content": "Hello, how are you?"}
        {"role": "assistant", "content": "I'm doing well, thank you!"}
        """
        record = self.get_user_record(db_env, user_id)
        record["chat_history"].extend(messages)
        self.save_user_record(db_env, user_id, record)

    def add_user_image(self, db_env, user_id: str, image: CompressedImage):
        """
        Append a new image to the user's record.
        
        The image is converted to a dictionary that contains:
        - header: including a timestamp (converted from sec and nanosec) and frame_id,
        - format: the image format (e.g., 'jpeg'),
        - data: base64 encoded image data.
        """
        record = self.get_user_record(db_env, user_id)
        
        # Convert the image header. Depending on your ROS version, the time fields might differ.
        # Here we assume the header stamp has 'sec' and 'nanosec' attributes.
        header_dict = {
            "stamp": image.header.stamp.sec + image.header.stamp.nanosec * 1e-9,
            "frame_id": image.header.frame_id
        }
        
        image_dict = {
            "header": header_dict,
            "format": image.format,
            "data": base64.b64encode(image.data).decode('utf-8')
        }
        
        record["obj_imgs"].append(image_dict)
        self.save_user_record(db_env, user_id, record)
            
    def initialize_db(self, dir: Path):
        db_path = dir / Path('db/user_chat_history.lmdb')
        db_path.parent.mkdir(exist_ok=True, parents=True)
        db_env = lmdb.open(str(db_path.absolute()), map_size=int(1e8)) # 0.1GB max size
        return db_env
        
def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down Interaction node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()