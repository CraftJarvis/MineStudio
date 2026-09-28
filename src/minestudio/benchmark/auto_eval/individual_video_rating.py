'''
Date: 2024-11-21 21:38:32
LastEditors: zhengxinyue
LastEditTime: 2024-12-06 16:38:55
FilePath: /MineStudio/minestudio/benchmark/auto_eval/individual_video_rating.py
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
import argparse


with open('./prompt/single_rating_prompt.txt', 'r', encoding='utf-8') as file:  
    system_content = file.read()

def fetch_gpt4(query):
    print('fetching gpt4 ...')
    client = OpenAI(api_key='empty')
    
    completion = client.chat.completions.create(
        model="gpt-4o", #gpt-4o-mini gpt-4o
        messages=query,
        temperature=0.5
    )
    res = completion.choices[0].message.content
    return res

def assess_video(task_name, frames, video_path_a, criteria_files_path):
    task_name = task_name.replace(' ', '_')
    try:
        with open(criteria_files_path, 'r', encoding='utf-8') as file:  
            grading_rule = file.read()
    except:
        print("no task file")
        return None
    query = [
        {
        "role": "system",
        "content": system_content
        },
        {
        "role": "user", "content":  
        f'The task name is ' + task_name + ' '
        + f'You should follow the following grading criteria to score the performance of agents in videos' + grading_rule +'\n'
        + f'Here are the image frames of the video A '
        }]

    query.append({"role": "user", "content": [{
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{frame}"
          },
        } for frame in frames
        ]})

    ans = fetch_gpt4(query)
    print(ans)
    save_data_json(ans, video_path_a, task_name)
    answer = {"role": "assistant", "content": f'{ans}'}
    return ans

def save_data_json(ans, video_path_a, task_name):
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
    result_dict['video_path'] = video_path_a
    result_dict['task_name'] = task_name
    
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
    if (len(base64Frames1)>60):
        base64Frames1 = base64Frames[0::70]

    return base64Frames1

def find_mp4_files(directory):
    mp4_files = []
    task_list = []
    video_name = []
    for root, dirs, files in os.walk(directory):

        files = files[:5]
        for file in files:
            if file.endswith('.mp4'):
                parent_dir = os.path.basename(root)
                full_path = os.path.join(root, file)
                video_name.append(file)
                mp4_files.append(full_path)
                task_list.append(parent_dir)
    return mp4_files, task_list, video_name


def main(video_path_a, rule_file_path):
    
    task_name = os.path.splitext(os.path.basename(rule_file_path))[0].replace('_', ' ')
    print(task_name)
    video_a = process_video(video_path_a)
    task_name = task_name.replace('_', ' ')
    assess_video(task_name, video_a, video_path_a, rule_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and assess videos.")
    parser.add_argument('--video_path', type=str, help='Path to the MCU videos directory.')
    parser.add_argument('--criteria_path', type=str, help='Path to the rule file.')
    
    args = parser.parse_args()
    
    if args.video_path and args.criteria_path:
        main(args.video_path, args.criteria_path)
    else:
        print("Please provide both --video_path and --criteria_files_path.")

