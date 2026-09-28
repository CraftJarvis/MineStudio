'''
Date: 2024-11-21 21:38:32
LastEditors: zhengxinyue
LastEditTime: 2024-12-06 16:38:33
FilePath: /MineStudio/minestudio/benchmark/auto_eval/batch_video_rating.py
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
import pdb
import json
import datetime
import argparse


with open('./prompt/single_rating_prompt.txt', 'r', encoding='utf-8') as file:  
    system_content = file.read()
metric = []

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
    # pdb.set_trace()
    task_name = task_name.replace(' ', '_')
    try:
        with open(f'{criteria_files_path}/{task_name}.txt', 'r', encoding='utf-8') as file:  
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
    # pdb.set_trace()
    ans = fetch_gpt4(query)
    print(ans)
    save_data_json(ans, video_path_a, task_name)
    answer = {"role": "assistant", "content": f'{ans}'}
    return ans

def save_data_json(ans, video_path_a, task_name):
    result_dict= {}
    metric_dict = {}
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
                    result_dict[key] = int(value) 
                    metric_dict[key] = int(value) 
                    break  
    metric.append(metric_dict)
    result_dict['video_path'] = video_path_a
    result_dict['task_name'] = task_name
    
    out_file = os.path.join('vlm_rating_res', task_name+'.json')
    with open(out_file, 'w') as f:
        json.dump([result_dict, ans], f, indent = 4)

    return result_dict  


def cal_metric():
    print('Begain calculating the results')
    values_dict = {}
    for d in metric:
        for key, value in d.items():
            if key not in values_dict:
                values_dict[key] = []
            values_dict[key].append(value)
    average_dict = {key: f"{sum(values) / len(values):.2f}/10" for key, values in values_dict.items()}
    for key, value in average_dict.items():
        print(f"{key}: {value}")


def process_video(task_name, video_path):
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

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                if file.endswith('.mp4'):
                    parent_dir = item
                    full_path = os.path.join(item_path, file)
                    video_name.append(file)
                    mp4_files.append(full_path)
                    task_list.append(parent_dir)
    return mp4_files, task_list, video_name


def main(videos_path, criteria_files_path):
    mp4_files, task_list, video_name = find_mp4_files(videos_path)

    for i in range(len(mp4_files)):
        video_path_a = mp4_files[i]
        task_name = task_list[i]
        print(task_name)
        video_a = process_video(task_name, video_path_a)
        task_name = task_name.replace('_', ' ')
        assess_video(task_name, video_a, video_path_a, criteria_files_path)
    cal_metric()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and assess videos.")
    parser.add_argument('--videos_path', type=str, help='Path to the MCU videos directory.')
    parser.add_argument('--criteria_files_path', type=str, help='Path to the rule file.')
    
    args = parser.parse_args()
    
    if args.videos_path and args.criteria_files_path:
        main(args.videos_path, args.criteria_files_path)
    else:
        print("Please provide both --videos_path and --criteria_files_path.")
        
