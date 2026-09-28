'''
Date: 2024-11-21 21:38:32
LastEditors: zhengxinyue
LastEditTime: 2024-12-06 16:39:13
FilePath: /MineStudio/minestudio/benchmark/auto_eval/video_comparison.py
'''
import cv2  # We're using OpenCV to read video
import base64
import time
from openai import OpenAI
import os
import requests
import shutil
from PIL import Image
from io import BytesIO
import json
import datetime
import openai
import requests
import argparse

with open('./prompt/compare_rating_prompts.txt', 'r', encoding='utf-8') as file:  
    system_content = file.read()

def fetch_gpt4(query): # gpt4
    print('fetching gpt4 ...')
    client = OpenAI(api_key='empty')
    completion = client.chat.completions.create(
        model="gpt-4o", #gpt-4o-mini gpt-4o
        messages=query,
        temperature=0.7
    )
    res = completion.choices[0].message.content
    return res


def assess_video(task_name, frames, frame_1, rule_file):
    with open(rule_file, 'r', encoding='utf-8') as file:  
        grading_rule = file.read()
    query = [
        {
        "role": "system",
        "content": system_content
        },
        {
        "role": "user", "content":  
        f'The task name is ' + task_name + ' '
        + f'You should follow the following grading criteria to compare the performance of agents in videos A and B' + grading_rule +'\n'
        + f'Here are the image frames of the video A '
        }]

    query.append({"role": "user", "content": [{
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{frame}"
          },
        } for frame in frames
        ]})

    query.append({"role": "user", "content": f'Here are the image frames of the video B '})

    query.append({"role": "user", "content": [{
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{frame}"
          },
        } for frame in frame_1
        ]})
    ans = fetch_gpt4(query)
    print(ans)
    answer = {"role": "assistant", "content": f'{ans}'}
    return ans

def save_data_json(ans, task_name, video_path_a, video_path_b):
    result_dict= {}
    keys_to_extract = [  
        "Task Progress",  
        "Action Control",  
        "Error Recognition and Correction",  
        "Creative Attempts",  
        "Task Completion Efficiency",  
        "Material Selection and Usage"  
    ]  

    for line in ans.strip().split('\n'):  
        for key in keys_to_extract:  
            if line.startswith(f'- {key}: '):   
                value = (line.split(': ', 1)[1].strip()) 
                
                if value: 
                    result_dict[key] = value  
                    break  
    result_dict['video_1_path'] = video_path_a
    result_dict['video_2_path'] = video_path_b
    task_name = task_name.replace(' ', '_')
    out_file = os.path.join('vlm_rating_res', task_name+'.json')
    with open(out_file, 'w') as f:
        json.dump([result_dict, ans], f, indent = 4)

    return result_dict  

def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        # frame = cv2.resize(orig_frame, (224, 224))
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    base64Frames1 = base64Frames[0::25]
    print(len(base64Frames1), "frames read.")
    assert(len(base64Frames1)<150)
    return base64Frames1


def main(video_path_a, video_path_b, rule_file_path):

    task_name = os.path.splitext(os.path.basename(rule_file_path))[0].replace('_', ' ')
    video_a = process_video(video_path_a)
    video_b = process_video(video_path_b)
    ans = assess_video(task_name, video_a, video_b, rule_file_path)
    save_data_json(ans, task_name, video_path_a, video_path_b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and assess two videos.")
    parser.add_argument('--video_path_a', type=str, help='Path to the first video file.')
    parser.add_argument('--video_path_b', type=str, help='Path to the second video file.')
    parser.add_argument('--criteria_path', type=str, help='Path to the criteria file.')
    
    args = parser.parse_args()
    
    main(args.video_path_a, args.video_path_b, args.criteria_path)
   