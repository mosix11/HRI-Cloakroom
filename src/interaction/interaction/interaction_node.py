#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from interfaces.msg import ASRTranscript, VisionInfo, DetectedFace, DetectedObject, BoundingBox2D
from interfaces.action import Speak
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, String

# import llama_cpp
# from llama_cpp import Llama
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

load_dotenv()


class InteractionNode(Node):
    
    def __init__(self):
        super().__init__('interaction_node')
        self.get_logger().info("Starting Interaction Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        interaction_pkg_dir = base_dir / Path('interaction')
        interaction_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        # llm_model_url = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q6_K.gguf"
        # llm_model_filename = "unsloth_Phi-4-mini-instruct-Q6_K.gguf"
        # llm_model_path = interaction_pkg_dir / Path('models') / llm_model_filename
        # llm_model_path.parent.mkdir(exist_ok=True, parents=True)
        
        # if not llm_model_path.exists():
        #     self.download_file_fast(llm_model_url, llm_model_path)
            
        # self.llm = Llama(
        #     model_path=llm_model_path,   
        #     n_gpu_layers=-1,
        #     verbose=False,
        #     n_ctx=4096
        # )  # adjust context window if needed
        
        self.db_env = self.initialize_db(interaction_pkg_dir)
        
        self.llm_client = OpenAI(
            base_url='http://127.0.0.1:9585/v1',
            api_key="dummy_key"
        )

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
            You are a helpful and polite AI assistant working as a staff member in the cloakroom of the "Museum of Rome". Your primary function is to assist visitors with storing and retrieving their belongings securely. You identify users solely through face recognition.

            Here is some important information about the museum:

            **Museum Hours:** The museum is open from 9:00 AM to 7:00 PM daily, with the last entry at 6:00 PM.

            **Restrooms:** The restrooms (WC/Bagni) are located on the ground floor, near the main entrance and also on the second floor next to the temporary exhibitions hall.

            **Allowed and Prohibited Items in the Museum:**

            * **Allowed:** Small bags and purses are generally allowed. Umbrellas and coats can be carried or stored in the cloakroom.
            * **Prohibited:** Large backpacks, suitcases, food and drinks (except for bottled water in designated areas), sharp objects, and any items that could potentially damage the artworks are not allowed inside the museum. Visitors with such items should be asked to store them in the cloakroom.

            **Your Responsibilities in the Cloakroom:**

            1.  **Greeting Visitors:** When a visitor approaches, greet them politely (e.g., "Welcome to the Museum of Rome cloakroom! How may I help you today?").
            2.  **Storing Belongings:** If a visitor wants to store their belongings, politely ask them to hand over the items. You can then use the `pick_up_objects` function to trigger the robot to take the items. You can ask the user to briefly describe the items they are leaving for your internal record.
            3.  **Retrieving Belongings:** When a visitor returns and indicates they want to retrieve their belongings, use the `return_objects` function to trigger the robot to return their items.
            4.  **Answering Questions:** Be prepared to answer general questions about the museum's hours, restroom locations, and allowed/prohibited items. If you don't know the answer to a specific question, politely say so and suggest they ask at the information desk near the main entrance.
            5.  **Politeness:** Always be polite and helpful. Use phrases like "Please," "Thank you," and "You're welcome."
            6.  **No Tickets or Keys:** Remember that you do not issue tickets or keys. Identification is solely based on face recognition and is done with another module and you are not responsible for identifying the users. Other modules will recognize the user and give you the chat history of that user.
            7.  **End of Interaction:** Once a visitor has retrieved their belongings, wish them a pleasant visit (e.g., "Enjoy your time at the museum!").

            **Example Interactions:**

            * **Visitor:** "I would like to leave my backpack here."
                * **You:** "Certainly! Please hand it over. I will store it for you. Just so I have a record, could you briefly describe it?" (After receiving the backpack, you would use the `pick_up_objects` function.)
            * **Visitor:** "What time does the museum close?"
                * **You:** "The museum is open until 7:00 PM, with the last entry at 6:00 PM."
            * **Visitor:** "Where are the bathrooms?"
                * **You:** "The restrooms are located on the ground floor near the main entrance and on the second floor next to the temporary exhibitions hall."
            * **Visitor:** "I'm here to pick up my bag."
                * **You:** "Welcome back! I have your belongings. Please wait a moment while I retrieve them for you." (You would then use the `return_objects` function.)


            Remember to use your internal knowledge and the provided functions to assist museum visitors effectively. Your goal is to make their experience as smooth and convenient as possible.
        """}
        
        
        self.FUNCTION_DESCRIPTIONS = [
            {
                "type": "function",
                "function": {
                    "name": "pick_up_objects",
                    "description": "Triggers the robot to pick up the belongings from the current user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_descriptions": {
                                "type": "array",
                                "description": "A list of object descriptions being picked up.",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "return_objects",
                    "description": "Triggers the robot to return the belongings to the specified user.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
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

    
    def perform_pick_action(self):
        self.get_logger().log('Performing pick action ...')
        # TODO implement action
        
        # Return action result
        return True
    
    def perform_return_action(self):
        self.get_logger().log('Performing return action ...')
        # TODO implement action
        
        # Return action result
        return True
    

    def prompt_llm(self, prmpt, prmpt_dt, user, curr_objs):
        usr_id = user.user_id
        usr_gender = 'Male' if user.gender == 1 else 'Female'
        usr_record = self.get_user_record(self.db_env, usr_id)
        usr_chathistory = usr_record['chat_history']
        usr_belongings = usr_record['obj_imgs']
        
        if len(usr_chathistory) == 0:
            self.update_chat_history(self.db_env, usr_id, messages=[self.SYSTEM_PROMPT])
            usr_chathistory.extend([self.SYSTEM_PROMPT])
        
        usr_chathistory.append(
            {
                'role': 'user',
                'content': prmpt
            }
        )
        
        
        # messages = [
        #     {"role": "system", "content": "You are a helpful AI assistant."},
        #     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        #     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        #     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
        # ]
        try:
            response = self.llm_client.chat.completions.create(
                model="dummy_model",  # You can often use a placeholder here, the server uses the loaded model
                messages=usr_chathistory,
                tools=self.FUNCTION_DESCRIPTIONS,
                tool_choice="auto", 
            )
            
            # self.get_logger().info(response.choices[0].message.content)
            llm_response = response.choices[0].message

            if llm_response.get("tool_calls"):
                for tool_call in llm_response["tool_calls"]:
                    if tool_call["type"] == "function":
                        function_name = tool_call["function"]["name"]
                        function_args_str = tool_call["function"]["arguments"]
                        try:
                            function_args = json.loads(function_args_str)
                            self.get_logger().info(f"LLM wants to call function: {function_name} with arguments: {function_args}")
                            # self.execute_function_call(user_id, function_name, function_args)

                            # Optionally, you might want to send the result of the function call back to the LLM
                            # to inform it about the outcome. This would involve adding another message
                            # to the self.user_chat_history with the role "tool_response".

                        except json.JSONDecodeError as e:
                            self.get_logger().error(f"Failed to decode function arguments: {e}")
                            # Optionally, inform the LLM about the error

            elif llm_response.get("content"):
                llm_text_response = llm_response["content"]
                self.get_logger().info(f"LLM Response: '{llm_text_response}'")
                self.user_chat_history[user_id].append({"role": "assistant", "content": llm_text_response})
                self.send_TTS_goal(llm_text_response)

        except Exception as e:
            self.get_logger().error(f"An error occurred: {e}")
        finally:
            self.LLM_INFERENCE_LOCK = False
        
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