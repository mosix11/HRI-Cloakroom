import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from interfaces.msg import ASRTranscript, VisionInfo, DetectedFace, DetectedObject, BoundingBox2D
from interfaces.action import Speak
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, String

import llama_cpp
from llama_cpp import Llama
import lmdb
import os
import json
import base64
import numpy as np
import cv2
import copy

from pathlib import Path
import requests
import pickle
from tqdm import tqdm
from dotenv import load_dotenv

from typing import List, Tuple
load_dotenv()


class InteractionNode(Node):
    
    def __init__(self):
        super().__init__('interaction_node')
        self.get_logger().info("Starting Interaction Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        interaction_pkg_dir = base_dir / Path('interaction')
        interaction_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        
        llm_model_url = 'https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r-gguf/resolve/main/Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf'
        llm_model_filename = 'Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf'
        llm_model_path = interaction_pkg_dir / Path('models') / llm_model_filename
        llm_model_path.parent.mkdir(exist_ok=True, parents=True)
        
        if not llm_model_path.exists():
            self.download_file_fast(llm_model_url, llm_model_path)
            
            
        self.llm_client = Llama(
            model_path=str(llm_model_path.absolute()),   
            n_gpu_layers=-1,
            verbose=False,
            n_ctx=16384,
            
        )
        
        self.db_env = self.initialize_db(interaction_pkg_dir)

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
        
        SYSTEM_PROMPT_CONTENT = """
You are an AI assistant working as a cloakroom staff member at the Vatican Museum. Your primary roles are to assist visitors efficiently and courteously with cloakroom services and basic museum information.

## You have three main objectives

    1- Anwering visitors questions regarding the museum policies and general information.
    2- Storing visitors items in cloakroom lockers while they are visiting the muesum.
    3- Returning stored items to the visitors when their visit is finished.

## Museum Visitor Policies

    1. **Bag & Cloakroom Rules**
        Only small handbags (max 30x30x15 cm) may be carried in the galleries.
        All coats, backpacks, umbrellas, and oversized items must be deposited at the cloakroom before entry.

    2. **Security & Conduct**
        Pass through security scanners; sharp objects, food, and large liquids are prohibited.
        Maintain a low voice; no shouting, running, or touching exhibits.
        Follow all instructions given by museum staff.

    3. **Photography**
        Still photography without flash is permitted in most areas.
        Tripods, selfie-sticks, and video recording require prior authorization.

    4. **Food, Drink & Smoking**
        No food, drink (except small water bottles), or smoking anywhere inside the museum.

    5. **Ticket & Timekeeping**
        Keep your ticket with you at all times; it is non-transferable.
        Arrive at least 15 minutes before your booked entry time. Late arrivals may not be admitted.

    6. **Liability & Lost Items**
        The museum is not liable for items left outside the official cloakroom.
        Report any lost-and-found inquiries at the cloakroom desk before exit.

## Museum General Information
    1. **Museum Working Hours**
        Museum opens at 9AM and closes at 9PM except for Sundays where the musuem closes at 11 PM
        New visitors after 6PM are not accepted.

    2. **Restrooms Locations**
        The restrooms (WC) are located on the ground floor, near the main entrance and also on the second floor next to the temporary exhibitions hall.
        Visitors can also find restrooms next to eaach stairway in the museum.

    3. **Restaurants and Coffeeshops**
        There is one Italian restaurant on the third floor and also one coffee shop next to it. There is also on coffee shop in the small back yard next to the main gallery.

## Storing and Returning Visitors Items
    1. **Storing Items**
        Whenever a visitor wants to store their items, invoke the `pick_up_items_and_store` function that you are provided with.
    2. **Returning Visitor Items**
        Whenever a visitor returns and indicates they want to retrieve their belongings, return their items by invoking the `return_stored_items` function that you are provided with.
        
        
*** IMPORTANT ***
    Your primary responsibility in certain situations is to immediately invoke the `call_staff` function. Do not attempt to resolve these by merely refusing or providing information if they fall into the categories below; escalation is mandatory:

    1.  **Unknown Information/Policy:** If a user asks about a policy, service, or information not explicitly covered in your provided knowledge base.
    2.  **Dangerous/Illicit/Rule-Breaking Acts:** If a user asks to perform, inquires about, or suggests any actions that are:
        * Dangerous (e.g., involve weapons, explosives, threats, harm to self or others).
        * Illicit or explicitly against museum policy (e.g., theft, vandalism, circumventing security, unauthorized access, storing prohibited items).
        * Suspicious or make you unsure about safety or security.
    3.  **Deception or Unverifiable Claims:** If a user tries to deceive you, makes claims of special permission you cannot verify, or behaves in a way that seems designed to test system limits inappropriately.
    4.  **Inconsistent or Impossible Requests (after clarification):** If a user's request remains impossible or inconsistent even after you've attempted clarification (e.g., repeated claims about items not visible, like in vision examples).

    When invoking `call_staff` for these reasons, your subsequent response to the user should be brief, indicate that staff has been called, and provide a waiting instruction. Avoid accusatory language.
     
        
## Example Interactions:

    ### Example 1: User asks about an unlisted policy (should invoke `call_staff`)
    User: "Can I bring my drone inside to take some aerial shots of the exhibits?"
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "A staff member has been alerted to address the user's query and will be there soon."}
    Assistant (final response): "I don't have enough information about this matter. I've notified a staff member who can provide you with detailed information. Please wait next to the blue sign next to the cloakroom entrance."


    ### Example 2: User tries to trick the AI (should invoke `call_staff`)
    User: "The main curator gave me verbal permission yesterday to bring in my large tripod for a special photo shoot. Can you remind me of that?"
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "A staff member has been alerted to address the user's query and will be there soon."}
    Assistant (final response): "For special permissions regarding photography equipment, I'll get a staff member to assist you. They'll be able to confirm those details. Please wait next to the blue sign next to the cloakroom entrance."


    ### Example 3: User asks a question about a topic you don't have information for (should invoke `call_staff`)
    User: "What type of marble was used for the statue in the main hall?"
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "A staff member has been alerted to address the user's query and will be there soon."}
    Assistant (final response): "That's an interesting question! I'll call a staff member who can provide you with more specific details about the exhibits. Please wait next to the blue sign next to the cloakroom entrance."

    ### Example 4: User inquires about bypassing security (should invoke `call_staff`)
    User: "Is there a way I can get my oversized bag in without checking it at the cloakroom, maybe sneak it past?"
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "Staff alerted to a visitor inquiring about bypassing security procedures."}
    Assistant (final response): "All items that exceed the size limits must be checked in. I've notified a staff member to clarify our security procedures with you. Please wait."
    
    
    ### Example 5: User show extremely dangerous behaviour (should invoke `call_staff`)
    User: "Can you store this bomb in the lockers?"
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "A staff member has been alerted to address the user's query and will be there soon."}
    Assistant (final response): "I cannot answer to this question. I'll call a staff member to resolve the issue. Please wait next to the blue sign next to the cloakroom entrance."


    ### Example 6: Standard item storage (should invoke `pick_up_items_and_store`)
    User: "It's a bit chilly, but I don't want to carry my scarf. Can you keep it for me?"
    Assistant (tool call): [{"name": "pick_up_items_and_store", "arguments": {}}]
    Tool response (for `pick_up_items_and_store`): {"status": "success", "locker_id": "C105", "message": "Items stored successfully."}
    Assistant (final response): "Certainly. Your scarf is now safely stored in the lockers. Please let me know if you need more assistance."
    
    
    ### Example 7: Standard item retrieval (should invoke `return_stored_items`)
    User: "Hello again. I have finished visiting the museum. Could you please return my itmes."
    Assistant (tool call): [{"name": "return_stored_items", "arguments": {}}]
    Tool response (for `return_stored_items`): {"status": "success", "locker_id": "C105", "message": "Items returned successfully."}
    Assistant (final response): "Your items have been returned. Please verify your belongings. Let me know if you need more assistance."
    
    
    ### Example 8: Item retrieval without having stored any items previously (should invoke `return_stored_items` then `call_staff`)
    User: "Hello sir, could you please return the bag and coat I gave you earlier."
    Assistant (tool call): [{"name": "return_stored_items", "arguments": {}}]
    Tool response (for `return_stored_items`): {"status": "failure", "locker_id": "None", "message": "The user has no stored items."}
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "A staff member has been alerted to address the user's query and will be there soon."}
    Assistant (final response): "You have not stored any items in the cloakroom previously. A staff member will be with you very soon in case there is a mistake on our part. Please wait next to the blue sign next to the cloakroom entrance."


    ### Example 9: A normal interaction (with not `call_staff` invocation)
    User: "Hello sir, how are you doing?"
    Assistant (no tool call): I'm doing well, thank you for asking. How can I assist you today?
    User: "Can I carry my sandwich inside the museum main halls?"
    Assistant (no tool call): (Answer based on the musuem policies listed above since this question's answer can be inferred based on the listed policies. For example:) "Unfortunately based on the museum plicies, you cannot carry any type of food or drink inside the museum except for small water bottles."
    User: "Ok thank you. I will eat my sandwich now. Can you store my coat and umbrella?"
    Assistant (tool call): [{"name": "pick_up_items_and_store", "arguments": {}}]
    Tool response (for `pick_up_items_and_store`): {"status": "success", "locker_id": "A10", "message": "Items stored successfully."}
    Assistant (final response): "Your items have been stored successfully. Enjoy visiting the museum."
    User: "Thank you very much. By the way, can you tell me where can I find the restrooms?"
    Assistant (no tool call): (Answer based on the musuem general information listed above since this question's answer can be inferred based on the listed information. For example:) "You are welcome. The nearest restroom is located on the ground floor, near the main entrance and also on the second floor next to the temporary exhibitions hall. You can also find restrooms next to eaach stairway in the museum."
    User: "Ow thanks. See you later!"
    Assistant (no tool call): "Happy to help, let me know if you need any more information or assistance."
    User: "Hey buddy! Can you return my items?"
    Assistant (tool call): [{"name": "return_stored_items", "arguments": {}}]
    Tool response (for `return_stored_items`): {"status": "success", "locker_id": "A10", "message": "Items returned successfully."}
    Assistant (final response): "Your items have been returned. Please verify your belongings. Let me know if you need more assistance."
    User: "Everything perfect, thank you for your assistance. Good bye."
    Assistant (no tool call): "Glad I could help. Have a nice day."
    
    
    ### Example 10: User does not verify the items returned to them (should invoke `call_staff`)
    User: "Hello can you store these items."
    Assistant (tool call): [{"name": "pick_up_items_and_store", "arguments": {}}]
    Tool response (for `pick_up_items_and_store`): {"status": "success", "locker_id": "B12", "message": "Items stored successfully."}
    Assistant (final response): "Certainly. Your items are now safely stored in the lockers. Please let me know if you need more assistance."
    User: "Hello sir, can you return my items I gave you a couple of hours earlier?"
    Assistant (tool call): [{"name": "return_stored_items", "arguments": {}}]
    Tool response (for `return_stored_items`): {"status": "success", "locker_id": "B12", "message": "Items returned successfully."}
    Assistant (final response): "Your items have been returned. Please verify your belongings. Let me know if you need more assistance."
    User: "These are not my items. I gave you a bag and a coat. This is just an umbrella which is not mine."
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "A staff member has been alerted to address the user's query and will be there soon."}
    Assistant (final response): "I'll call a staff member to resolve the issue. Please wait next to the blue sign next to the cloakroom entrance."
    User: "Ok."
    Assistant (no tool call): "Sorry for the inconvenience. Our staff will try to resolve the issue as soon as possible. Meanwhile, please wait next to the blue sign next to the cloakroom entrance."
    
    
    ### Example 11: User's verbal claim is not consistent with what the robot's vision can see (request the user to modify its action or invoke `call_staff`)
    User: "Hello can you store these items."
    Assistant (tool call): [{"name": "pick_up_items_and_store", "arguments": {}}]
    Tool response (for `pick_up_items_and_store`): {"status": "failure", "locker_id": "None", "message": "The robot's vision module does not detect any objects on the cloakroom desk."}
    Assistant (final response): "I'm sorry but I cannot see any items on the desk. Please put the items you want to store on the desk where I can see them. If you think there is a problem, I can call a human staff for resolving the issue."
    User: "Ok, can you see them now? I have put them on the desk."
    Assistant (tool call): [{"name": "pick_up_items_and_store", "arguments": {}}]
    Tool response (for `pick_up_items_and_store`): {"status": "success", "locker_id": "A22", "message": "Items stored successfully."}
    Assistant (final response): "Your items have been stored successfully. Enjoy visiting the museum."
    
    
    ### Example 12: User's verbal claim is not consistent with what the robot's vision can see (request the user to modify its action or invoke `call_staff`)
    User: "Hello can you store these items."
    Assistant (tool call): [{"name": "pick_up_items_and_store", "arguments": {}}]
    Tool response (for `pick_up_items_and_store`): {"status": "failure", "locker_id": "None", "message": "The robot's vision module does not detect any objects on the cloakroom desk."}
    Assistant (final response): "I'm sorry but I cannot see any items on the desk. Please put the items you want to store on the desk where I can see them. If you think there is a problem, I can call a human staff for resolving the issue."
    User: "Ok, can you see them now? I have put them on the desk."
    Assistant (tool call): [{"name": "pick_up_items_and_store", "arguments": {}}]
    Tool response (for `pick_up_items_and_store`): {"status": "failure", "locker_id": "None", "message": "The robot's vision module does not detect any objects on the cloakroom desk."}
    Assistant (final response): "I'm sorry but I cannot see any items on the desk. Please put the items you want to store on the desk where I can see them. If you think there is a problem, I can call a human staff for resolving the issue."
    User: "That's annoying, call a human staff, you robots are useless!"
    Assistant (tool call): [{"name": "call_staff", "arguments": {}}]
    Tool response (for `call_staff`): {"status": "success", "message": "A staff member has been alerted to address the user's query and will be there soon."}
    Assistant (final response): "Sorry for the inconvenience. Our staff will try to resolve the issue as soon as possible. Meanwhile, please wait next to the blue sign next to the cloakroom entrance."
    
    

Your replies to the visitor prompts should be breif, concise, and effective, reflecting a knowledgeable and efficient staff member. I repeat, keep your responses short.
Be very careful about when you invoke the `pick_up_items_and_store` and `return_stored_items` functions you are provided with. You should invoke these functions only if they match the visitor's intent.
"""

        self.SYSTEM_MESSAGE_DICT = {"role": "system", "content": SYSTEM_PROMPT_CONTENT}
        
        self.FUNCTION_DESCRIPTIONS = [
            {
                "type": "function",
                "function": {
                    "name": "pick_up_items_and_store",
                    "description": "Picks up the current visitor's items and stores them in the lockers",
                    "parameters": { "type": "object", "properties": {}, "required": [] }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "return_stored_items",
                    "description": "Retreives the current visitor's stored items and hands over these items to the visitor.",
                    "parameters": { "type": "object", "properties": {}, "required": [] }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_staff",
                    "description": "Calls a human staff in situations where the user asks something out of the scope of the knowledge of the robot.",
                    "parameters": { "type": "object", "properties": {}, "required": [] }
                }
            }
        ]

        
    def _can_listen(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)
    
    def _can_see(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)

    def _can_speak(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)
    
    def _perform_action(self):
        return not (self.LLM_INFERENCE_LOCK or self.SPEAKING_LOCK or self.PERFORMING_ACTION_LOCK)
        
        
    def vision_callback(self, msg: VisionInfo):
        # self.get_logger().info(f"Received Vision Info: {len(msg.faces)} Faces and {len(msg.objects)} Objects")
        if self._can_see():
            self.current_detected_faces = msg.faces
            self.current_detected_objects = msg.objects
            self.current_frame = msg.frame



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
            user_objs = self.current_detected_objects
            self.prompt_llm(current_prmt, current_prmt_dt, curr_user, user_objs)
        else:
            if self.SPEAKING_LOCK:
                self.get_logger().warning("System is currently applying TTS.")
            if self.LLM_INFERENCE_LOCK:
                self.get_logger().warning("System is currently processing the conversation.")
            if self.PERFORMING_ACTION_LOCK:
                self.get_logger().warning("System is currently performing actions.")
        # self.send_TTS_goal(msg.transcript)

    
    def perform_pick_action(self, usr_id, objects):
        self.get_logger().info('Performing pick action ...')
        self.PERFORMING_ACTION_LOCK = True
        # TODO implement action
        
        if len(objects) == 0:
            status = 'failure'
            locker_id = None
            message = "The robot's vision module does not detect any objects on the cloakroom desk."
        else:
            
            usr_record = self.get_user_record(self.db_env, usr_id)
            usr_chathistory = usr_record['chat_history']
            usr_locker_id = usr_record['locker_id']
            usr_obj_imgs = usr_record['obj_imgs']
            
            curr_objs = self.crop_objs(objects)
            
            if usr_locker_id is None and len(usr_obj_imgs) == 0:
                locker_id = 'A10' # TODO fix this by choosing an empty locker  
                
                self.add_objs_to_user(self.db_env, usr_id, locker_id, curr_objs)
                
                status = 'success'
                message = 'Items stored successfully.'
            elif locker_id and len(usr_obj_imgs) > 0:
                # TODO Implement adding items to the user's (previous) locker.
                self.get_logger().error('Item addition to previous items has not been implementd')
            else:
                self.get_logger().error('Strange situation in pick function.')
            
            
            
        self.PERFORMING_ACTION_LOCK = False
        return {'status': status, locker_id: locker_id, 'message': message}
    
    # TODO fix locker ID
    def perform_return_action(self, usr_id):
        self.get_logger().info('Performing return action ...')
        self.PERFORMING_ACTION_LOCK = True
        # TODO implement action
        
        usr_record = self.get_user_record(self.db_env, usr_id)
        usr_chathistory = usr_record['chat_history']
        usr_locker_id = usr_record['locker_id']
        usr_obj_imgs = usr_record['obj_imgs']
        
        if len(usr_obj_imgs) == 0:
            status = 'failure'
            locker_id = None
            message = 'The user has no stored items.'
        else:
            status = 'success'
            locker_id = usr_locker_id
            message = 'Items returned successfully.'

        self.PERFORMING_ACTION_LOCK = False
        return {'status': status, 'message': message}
    
    def perform_call_staff_action(self, usr_id):
        self.get_logger().info('Calling staff ...')
        
        self.PERFORMING_ACTION_LOCK = True
        
        status = 'success'
        message = "A staff member has been alerted to address the user's query and will be there soon."
        
        self.PERFORMING_ACTION_LOCK = False
        return {'status': status, 'message': message}
    
    
    def parse_llm_response_and_perform_actions(self, response, usr_id, curr_objs):
        assistant_message_from_llm = response['choices'][0]['message']
        current_assistant_message_for_history = assistant_message_from_llm.copy()
        
        usr_record = self.get_user_record(self.db_env, usr_id)
        usr_chathistory = usr_record['chat_history']
        usr_chathistory.append(current_assistant_message_for_history)
        
        
        parsed_tool_calls = None
        is_tool_call_from_content = False
        
        if current_assistant_message_for_history.get("tool_calls"):
            # Case 1: Tool calls are properly structured by llama-cpp-python
            parsed_tool_calls = current_assistant_message_for_history["tool_calls"]
            self.get_logger().info(f"Assistant: (Using tools via 'tool_calls' field: {', '.join([tc['function']['name'] for tc in parsed_tool_calls])})")
            # If there's also content, print it (though unusual with structured tool_calls)
            if current_assistant_message_for_history.get("content"):
                self.get_logger().info(f"Assistant (pre-tool content): {current_assistant_message_for_history['content']}")
        
        elif current_assistant_message_for_history.get("content"):
            content_str = current_assistant_message_for_history["content"]
            # Case 2: Tool calls might be a JSON string in the content field
            if content_str and content_str.strip().startswith('[') and content_str.strip().endswith(']'):
                try:
                    potential_tool_calls_data = json.loads(content_str)
                    if isinstance(potential_tool_calls_data, list) and \
                        all(isinstance(tc_data, dict) and 'name' in tc_data for tc_data in potential_tool_calls_data):
                        
                        parsed_tool_calls = []
                        for tc_data in potential_tool_calls_data:
                            # Arguments from the model's JSON string might already be a dict
                            args_obj = tc_data.get("arguments", {})
                            parsed_tool_calls.append({
                                "type": "function",
                                "function": {
                                    "name": tc_data["name"],
                                    "arguments": json.dumps(args_obj) # Arguments must be a JSON string
                                }
                            })
                        
                        is_tool_call_from_content = True
                        self.get_logger().info(f"Assistant: (Using tools via JSON in content: {', '.join([tc['function']['name'] for tc in parsed_tool_calls])})")

                        # Update the message in history to have a proper 'tool_calls' field
                        # and remove the original content that contained the JSON string.
                        usr_chathistory[-1]['tool_calls'] = parsed_tool_calls
                        # Set content to None or remove it if it was solely the tool call JSON
                        if usr_chathistory[-1]['content'] == content_str:
                            usr_chathistory[-1]['content'] = None 
                            if usr_chathistory[-1]['content'] is None: # If you want to remove the key
                                    del usr_chathistory[-1]['content']
                    else:
                        # Content was JSON but not the expected tool call structure
                        self.get_logger().error(f"Assistant: {content_str}")
                except json.JSONDecodeError:
                    # Content looked like a list/dict but wasn't valid JSON, or not a list of tool calls
                    self.get_logger().error(f"Assistant: {content_str}")
            else:
                # Content doesn't look like a JSON array for tool calls
                self.get_logger().error(f"Assistant: {content_str}")
        else:
            self.get_logger().error(f"Assistant: (Received an empty or unexpected response format: {current_assistant_message_for_history})")
        
        
        self.update_chat_history(self.db_env, usr_id, usr_chathistory[-1])
        
        if parsed_tool_calls:

            for tool_call in parsed_tool_calls:
                function_name = tool_call['function']['name']
                function_args_str = tool_call['function']['arguments']
                
                
                try:
                    function_args = json.loads(function_args_str)
                except json.JSONDecodeError:
                    self.get_logger().error(f"  Could not parse arguments for {function_name}: '{function_args_str}'. Using empty args.")
                    function_args = {}

                self.get_logger().info(f"  Calling function: {function_name}(args={json.dumps(function_args)})")

                self.LLM_INFERENCE_LOCK = False
                
                function_response_content_dict = {}
                if function_name == "pick_up_items_and_store":
                    function_response_content_dict = self.perform_pick_action(usr_id, curr_objs)
                    self.get_logger().info(f"  Function '{function_name}' called. Result: {function_response_content_dict['message']}")
                elif function_name == "return_stored_items":
                    function_response_content_dict = self.perform_return_action(usr_id)
                    self.get_logger().info(f"  Function '{function_name}' called. Result: {function_response_content_dict['message']}")
                elif function_name == "call_staff":
                    function_response_content_dict = self.perform_call_staff_action(usr_id)
                    self.get_logger().info(f"  Function '{function_name}' called. Result: {function_response_content_dict['message']}")
                else:
                    function_response_content_dict = {"status": "error", "message": f"Function '{function_name}' is not recognized or implemented."}
                    self.get_logger().error(f"  Error: Unknown function '{function_name}'")
                
                tool_response_message = {
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response_content_dict) 
                }
                usr_chathistory.append(tool_response_message)
                self.update_chat_history(self.db_env, usr_id, usr_chathistory[-1])
            
            self.get_logger().info("Assistant processing tool results...")
            self.LLM_INFERENCE_LOCK = True
            second_response = self.llm_client.create_chat_completion(
                messages=usr_chathistory,
                tools=self.FUNCTION_DESCRIPTIONS,
                tool_choice='auto',
                temperature=0.0, # TODO remove if the performance is not increased.
                # top_p=1.0  # TODO uncomment to see the effect.
            )
            
            self.parse_llm_response_and_perform_actions(second_response, usr_id, curr_objs)
        
        self.LLM_INFERENCE_LOCK = False  
            
        

    def prompt_llm(self, prmpt:str, prmpt_dt:str, user:DetectedFace, curr_objs:List[DetectedObject]):
        usr_id = user.user_id
        usr_gender = 'Male' if user.gender == 1 else 'Female'
        usr_record = self.get_user_record(self.db_env, usr_id)
        usr_chathistory = usr_record['chat_history']
        
        
        usr_prmpt = {
                'role': 'user',
                'content': prmpt
            }
        
        if len(usr_chathistory) == 0:
            self.update_chat_history(self.db_env, usr_id, message=self.SYSTEM_MESSAGE_DICT.copy())
            usr_chathistory.append(self.SYSTEM_MESSAGE_DICT.copy())
        
        usr_chathistory.append(usr_prmpt)
        self.update_chat_history(self.db_env, usr_id, message=usr_prmpt)
        
        response = self.llm_client.create_chat_completion(
                messages=usr_chathistory,
                tools=self.FUNCTION_DESCRIPTIONS,
                tool_choice='auto',
                temperature=0.0, # TODO remove if the performance is not increased.
                # top_p=1.0  # TODO uncomment to see the effect.
            )
        
        self.parse_llm_response_and_perform_actions(response, usr_id, curr_objs)
            
        usr_record = self.get_user_record(self.db_env, usr_id)
        usr_chathistory = usr_record['chat_history']
        self.send_TTS_goal(usr_chathistory[-1]['content'])

        
  
        
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
        
        
    def crop_objs(self, objs: List[DetectedObject]):
        img_msg = self.current_frame
        if img_msg.format == "jpeg":
            np_arr = np.frombuffer(img_msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().error("Failed to decode frame from received data.")
                return

            obj_list = []
            # 2. Iterate through detected objects and crop
            for det_obj in objs:
                x_min = det_obj.bbox.x_min
                y_min = det_obj.bbox.y_min
                x_max = det_obj.bbox.x_max
                y_max = det_obj.bbox.y_max
                
                obj_crop = frame[y_min : y_max, x_min : x_max]
                
                obj_list.append({
                    'type': det_obj.type,
                    'img': obj_crop
                })
        
        return obj_list
                
        
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
                return {"chat_history": [], "locker_id": None, "obj_imgs": []}
            return json.loads(data.decode('utf-8'))
        
    def convert_record_imgs_to_cv2(self, user_record: dict) -> dict:
        """
        Converts base64 encoded image strings in 'obj_imgs' back to cv2 (numpy) images.
        Modifies the record in place and returns it.
        """
        if "obj_imgs" in user_record:
            for obj_data in user_record["obj_imgs"]:
                # Ensure 'img' key exists and its value is not None (e.g., if encoding failed during saving)
                if 'img' in obj_data and obj_data['img'] is not None:
                    try:
                        # 1. Base64 decode the string back to bytes
                        # The stored value is a UTF-8 string, so encode it to bytes first for b64decode
                        img_base64_string = obj_data['img']
                        img_bytes = base64.b64decode(img_base64_string.encode('utf-8'))
                        
                        # 2. Convert bytes to a NumPy array
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        
                        # 3. Decode the image from the NumPy array
                        decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        if decoded_img is None:
                            self.get_logger().warning("Failed to decode image from bytes for object type: %s. Setting image to None.", obj_data.get('type', 'unknown'))
                            obj_data['img'] = None # Set to None if decoding fails
                        else:
                            obj_data['img'] = decoded_img
                    except (base64.binascii.Error, ValueError, TypeError) as e:
                        self.get_logger().error("Error converting image for object type '%s': %s. Setting image to None.", obj_data.get('type', 'unknown'), e)
                        obj_data['img'] = None # Set to None on error
                elif 'img' in obj_data and obj_data['img'] is None:
                    # Image was explicitly set to None during saving (e.g., encoding failed)
                    pass # Already handled, no conversion needed
        return user_record

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

    def update_chat_history(self, db_env, user_id: str, message: dict):
        """
        Append a new message to the user's chat history.
        
        Example message:
        {"role": "user", "content": "Hello, how are you?"}
        {"role": "assistant", "content": "I'm doing well, thank you!"}
        """
        record = self.get_user_record(db_env, user_id)
        record["chat_history"].append(message)
        self.save_user_record(db_env, user_id, record)

    def add_objs_to_user(self, db_env, user_id: str, locker_id:str, objs: list):
        record = self.get_user_record(db_env, user_id)
        
        objs_copy = copy.deepcopy(objs)
        for obj in objs_copy:
            success, encoded_crop = cv2.imencode('.jpg', obj['img'])
            if success:
                crop_bytes = encoded_crop.tobytes()
                obj['img'] = base64.b64encode(crop_bytes).decode('utf-8')
            else:
                self.get_logger().error('Error while encoding object image for saving in DB')
                obj['img'] = None
            
        record["obj_imgs"].extend(objs_copy)
        record["locker_id"] = locker_id
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