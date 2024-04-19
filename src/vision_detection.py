#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vlm_vision_service.srv import ProcessImage, ProcessImageResponse
import cv2
from cv_bridge import CvBridge, CvBridgeError
import base64
import requests
import json
import re
import ast
from typing import List
import os

def make_api_call_to_clean_res(response) -> List:
    api_key = rospy.get_param("/api_key")  
    url = 'https://api.openai.com/v1/completions'

    detected_items = get_res(response)
    prompt_text = f"I have detected the following items in the image: {detected_items}\n\n Write the items in a list as follows: ['item1', 'item2', 'item3', 'item4']"

    
    payload = {
        "model": "text-davinci-003",  
        "prompt": prompt_text,
        "max_tokens": 100,
        "temperature": 0.5
    }

    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    
    req = requests.post(url, data=json.dumps(payload), headers=headers)
    if req.status_code == 200:
        print("API call to OpenAI GPT-3 went well!")
        response_data = req.json()
        print(response_data)
    else:
        print("API call to OpenAI GPT-3 model failed:", req.text)

    
    response_text = response_data['choices'][0]['text'].strip()
    reg_list = re.findall(r'\[.*?\]', response_text)
    if reg_list:
        
        return json.loads(reg_list[0])
    else:
        return []


def is_last_step(item ,items: list):
    
    if item == items[-1]:
        return 1
    return 0


def send_detected_food_items(items: list):
    i=0
    for item in items:
        
        message = f"{item},✌️,{is_last_step(item, items)}"
        
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(message.encode(), ('127.0.0.1', 666 + i))
            print(f"Message sent: {message}")
            print("Sent to port ", 666 + i)
            i += 1

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_image(request):
   
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(request.image, "bgr8")
    except CvBridgeError as e:
        print(e)

    image_data = image_to_base64(cv_image)
    gpt_vision_response = make_api_call_gpt_vision(image_data)
    print(gpt_vision_response)
    res = make_api_call_to_clean_res(gpt_vision_response)
    res = ast.literal_eval(res[0])  
    # create a publisher that publishes a topic with the food results

    return ProcessImageResponse(detected_items=res)

def make_api_call_gpt_vision(base64_image):
    api_key = rospy.get_param("/api_key")  
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Which food items can you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    # the print(gpt response) {'id': 'chatcmpl-9Fo5G9fffLTovIBAQ6QgysvcnbDn1', 'object': 'chat.completion', 'created': 1713554470, 'model': 'gpt-4-1106-vision-preview', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'There are no food items visible in this image. The picture shows electronic equipment, specifically DJ equipment, including a mixer, turntables, and a laptop. There are also various cables and a set of keys on the table.'}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1122, 'completion_tokens': 45, 'total_tokens': 1167}, 'system_fingerprint': None}

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print("API call to GPT Vision failed:", response.text)
        return None

def vision_service():
    rospy.init_node('vision_service_node')
    s = rospy.Service('vlm_process_image', ProcessImage, process_image)
    rospy.spin()

if __name__ == "__main__":
    vision_service()

