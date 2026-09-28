import os
import cv2
import time
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw

from minestudio.models import RocketPolicy, load_rocket_policy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import load_callbacks_from_config

from sam2.build_sam import build_sam2_camera_predictor
import re
from openai import OpenAI
from io import BytesIO
import requests
import base64
from xml.etree import ElementTree as ET

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

SEGMENT_MAPPING = {
    "Hunt": 0, "Use": 3, "Mine": 2, "Interact": 3, "Craft": 4, "Switch": 5, "Approach": 6, "None": -1
}


# NOOP_ACTION = {
#     "ESC": 0,
#     "back": 0,
#     "drop": 0,
#     "forward": 0,
#     "hotbar.1": 0,
#     "hotbar.2": 0,
#     "hotbar.3": 0,
#     "hotbar.4": 0,
#     "hotbar.5": 0,
#     "hotbar.6": 0,
#     "hotbar.7": 0,
#     "hotbar.8": 0,
#     "hotbar.9": 0,
#     "inventory": 0,
#     "jump": 0,
#     "left": 0,
#     "right": 0,
#     "sneak": 0,
#     "sprint": 0,
#     "swapHands": 0,
#     "camera": np.array([0, 0]),
#     "attack": 0,
#     "use": 0,
#     "pickItem": 0,
# }

class Session:
    
    def __init__(self, model_path: str, sam_path: str, name_file_mapping: dict):
        start_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.current_image = np.array(start_image)
        self.model_path = model_path
        self.sam_path = sam_path
        self.name_file_mapping = name_file_mapping
        self.clear_points()
        
        self.sam_choice = 'base'
        self.load_sam()
        
        self.tracking_flag = True
        self.points = []
        self.points_label = []
        self.able_to_track = False
        self.segment_type = "Approach"
        self.obj_mask = np.zeros((224, 224), dtype=np.uint8)
        self.calling_rocket = False
        self.num_steps = 0
    
    def clear_points(self):
        self.points = []
        self.points_label = []
    
    def clear_obj_mask(self):
        self.obj_mask = np.zeros((224, 224), dtype=np.uint8)
    
    def clear_agent_memory(self):
        self.num_steps = 0
        if hasattr(self, "agent"):
            self.state = self.agent.initial_state()

    def load_sam(self):
        
        ckpt_mapping = {
            'large': [os.path.join(self.sam_path, "sam2_hiera_large.pt"), "sam2_hiera_l.yaml"],
            'base': [os.path.join(self.sam_path, "sam2_hiera_base_plus.pt"), "sam2_hiera_b+.yaml"],
            'small': [os.path.join(self.sam_path, "sam2_hiera_small.pt"), "sam2_hiera_s.yaml"], 
            'tiny': [os.path.join(self.sam_path, "sam2_hiera_tiny.pt"), "sam2_hiera_t.yaml"]
        }
        sam_ckpt, model_cfg = ckpt_mapping[self.sam_choice]
        # first realease the old predictor
        if hasattr(self, "predictor"):
            del self.predictor
        self.predictor = build_sam2_camera_predictor(model_cfg, sam_ckpt)
        print(f"Successfully loaded SAM2 from {sam_ckpt}")
        self.able_to_track = False

    def segment(self):
        if len(self.points) > 0 and len(self.points_label) > 0:
            self.able_to_track = True
            self.predictor.load_first_frame(self.current_image)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, 
                obj_id=0,
                points=self.points,
                labels=self.points_label,
            )
        else:
            out_obj_ids, out_mask_logits = self.predictor.track(self.current_image)
        self.obj_mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy() # 360, 640
        self.clear_points()
        return self.obj_mask
    
    def reset(self, env_name: str):
        self.image_history = []
        # self.env = MinecraftWrapper(env_name, prev_action_obs=False)
        self.env = MinecraftSim(
            preferred_spawn_biome="plains", 
            callbacks=load_callbacks_from_config(self.name_file_mapping[env_name]),
        )
        self.obs, self.info = self.env.reset()
        for i in range(30): #! better init
            time.sleep(0.1)
            noop_action = self.env.noop_action()
            self.obs, self.reward, terminated, truncated, self.info = self.env.step(noop_action)
        
        self.reward = 0
        if os.path.exists(self.model_path):
            agent = load_rocket_policy(self.model_path)
        else:
            agent = RocketPolicy.from_pretrained(self.model_path)
        self.agent = agent.to("cuda")
        self.agent.eval()
        self.clear_agent_memory()
        self.current_image = self.info["pov"]
        self.image_history.append(self.current_image)
        return self.current_image
    
    def apply_mask(self):
        image = self.current_image.copy()
        color = COLORS[ SEGMENT_MAPPING[self.segment_type] ]
        color = np.array(color).reshape(1, 1, 3)[:, :, ::-1]
        obj_mask = (self.obj_mask[..., None] * color).astype(np.uint8)
        image = cv2.addWeighted(image, 1.0, obj_mask, 0.5, 0.0)
        return image
    
    def step(self, input_action=None):
        if input_action is not None:
            action = input_action
        else:
            obj_id = torch.tensor( SEGMENT_MAPPING[self.segment_type] )
            obj_mask = self.obj_mask.astype(np.uint8)
            obj_mask = cv2.resize(obj_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            obj_mask = torch.tensor(obj_mask, dtype=torch.uint8)
            obs = {
                'image': self.obs['image'], 
                'segment': {
                    'obj_id': obj_id, 
                    'obj_mask': obj_mask, 
                }
            }
            action, self.state = self.agent.get_action(obs, self.state, input_shape="*")

        self.obs, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.current_image = self.info["pov"]
        image = self.current_image
        if self.able_to_track and self.tracking_flag:
            self.segment()
            image = self.apply_mask()
        else:
            time.sleep(0.01)
        self.num_steps += 1
        self.image_history.append(image)
        return image

def encode_image_base64(image: np.array) -> str:
    # Convert the image to a base64 string
    img = Image.fromarray(image)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    return img_base64

class Pointer:
    '''
    Pointer is a model based on molmo, which can output point location of the object in the image.
    '''
    def __init__(self, model_id, model_url:str="http://127.0.0.1:9162/v1", api_key:str="EMPTY"):
        self.model_id = model_id
        self.model_url = model_url
        self.api_key = api_key

    def post_init(self):
        if self.model_url == None or self.model_url == 'huggingface':
            self.client = None
            self.load_molmo_from_hf(model_id)
        else:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.model_url,
            )
            models = client.models.list()
            print(models)
            model = models.data[0].id
            assert model == self.model_id, f"Model {self.model_id} not found in current model_url {self.model_url}"
            print(f"Using model {self.model_id} based on url {self.model_url}")
            self.client = client

    def load_molmo_from_hf(self, model_id):
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto')
        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto')

    def openai_generation(self, prompt:str, image) -> str:
        if hasattr(self, "client") == False:
            print("Initializing OpenAI client")
            self.post_init()
        image_base64 = encode_image_base64(image)
        chat_completion_from_base64 = self.client.chat.completions.create(
            messages=[{
                "role":"user",
                "content": [{"type": "text", "text": prompt},
                        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            model=self.model_id,
        )
        return chat_completion_from_base64.choices[0].message.content

    def parse_coordinates(self, xml_string: str):
        # Parse the XML string
        root = ET.fromstring(xml_string)
        # Initialize an empty list to store coordinates
        coordinates = []
        # Iterate over the attributes of the XML node
        for attr_name, attr_value in root.attrib.items():
            # Check if the attribute is an 'x' or 'y' coordinate by matching the pattern
            if attr_name.startswith('x'):
                # Get the corresponding 'y' coordinate
                y_attr_name = 'y' + attr_name[1:]  # Assume 'y' coordinate has the same index
                if y_attr_name in root.attrib:
                    # Append the (x, y) tuple to the coordinates list
                    coordinates.append((float(attr_value), float(root.attrib[y_attr_name])))
        return coordinates

    # def gen_point(self, image:Image, object_name:str):
    def gen_point(self, image:Image, prompt:str):
        # prompt = f"Check whether {object_name} in this image. If {object_name} exists in the image, pinpoint all {object_name}. If not, output 'None'."
        self.molmo_result = self.openai_generation(prompt, image)
        print("Pointing result: ", self.molmo_result)
        if 'none' in self.molmo_result.lower():
            return None
        else:
            points = self.parse_coordinates(self.molmo_result)
            points = [(int(x/100*640), int(y/100*360)) for x, y in points]
            return points 

def reset_fn(env_name, session):
    image = session.reset(env_name)
    return image, session

def step_fn(act_key, session):
    # action = NOOP_ACTION.copy()
    action = self.env.noop_action()
    if act_key != "null":
        action[act_key] = 1
    image = session.step(action)
    # image = Image.fromarray(image)
    return image, session

def loop_step_fn(steps, session):
    for i in range(steps):
        image = session.step()
        status = f"Running Agent `Rocket` steps: {i+1}/{steps}. "
        # image = Image.fromarray(image)
        yield image, session.num_steps, status, session

def clear_memory_fn(session):
    image = session.current_image
    session.clear_agent_memory()
    return image, "0", session

def get_points_with_draw(image, label, session, evt: gr.SelectData):
    points = session.points
    point_label = session.points_label
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (0, 255, 0) if label == 'Add Points' else (255, 0, 0)
    points.append([x, y])
    point_label.append(1 if label == 'Add Points' else 0)
    cv2.circle(image, (x, y), point_radius, point_color, -1)
    return image, session

def clear_points_fn(session):
    session.clear_points()
    return session.current_image, session

def segment_fn(session):
    session.segment()
    image = session.apply_mask()
    return image, session

def clear_segment_fn(session):
    session.clear_obj_mask()
    session.tracking_flag = False
    return session.current_image, False, session

def set_tracking_mode(tracking_flag, session):
    session.tracking_flag = tracking_flag
    return session

def set_segment_type(segment_type, session):
    session.segment_type = segment_type
    return session

def play_fn(session):
    image = session.step()
    return image, session

def make_video_fn(session, make_video, save_video, progress=gr.Progress()):
    images = session.image_history
    if len(images) == 0:
        return session, make_video, save_video
    filepath = "rocket.mp4"
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filepath, fourcc, 20.0, (w, h))
    for image in progress.tqdm(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    video.release()
    session.image_history = []
    return session, gr.Button("Make Video", visible=False), gr.DownloadButton("Save Video", value=filepath, visible=True)

def save_video_fn(session, make_video, save_video):
    return session, gr.Button("Make Video", visible=True), gr.DownloadButton("Save Video", visible=False)

def choose_sam_fn(sam_choice, session):
    session.sam_choice = sam_choice
    session.load_sam()
    return session
def molmo_fn(molmo_text, molmo_session,session):
    img = Image.fromarray(session.current_image)
    output_text = molmo_session.generate(img, molmo_text)
    return output_text

def extract_points(data):
    # 匹配 x 和 y 坐标的值，支持 <points> 和 <point> 标签
    pattern = r'x\d?="([-+]?\d*\.\d+|\d+)" y\d?="([-+]?\d*\.\d+|\d+)"'
    points = re.findall(pattern, data)
    # 将提取到的坐标转换为浮点数
    points = [(float(x)/100*640, float(y)/100*360) for x, y in points]
    
    return points

def add_points_fn(image, text, session):
    new_points = extract_points(text)
    points = session.points
    point_label = session.points_label
    for x, y in new_points:
        point_radius, point_color = 5, (0, 255, 0) 
        x,y = int(x),int(y)
        points.append([x, y])
        point_label.append(1)
        cv2.circle(image, (x, y), point_radius, point_color, -1)
    return image, session