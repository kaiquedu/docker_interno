import io
import time
import base64
import runpod
import requests
import random
from PIL import Image, PngImagePlugin
from requests.adapters import HTTPAdapter, Retry

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    endpoint = inference_request['endpoint']
    if endpoint in ['img2img', 'txt2img']:
        try:
            model = inference_request['model']
            inference_request['override_settings'] = {
                "sd_model_checkpoint": model,
                "sampling_method": "DPM++ 2M",
                "scheduler": "karras",
                "steps": 50,
                "refiner": {
                    "model": model,
                    "switch_at": 0.8
                },
                "resize_mode": "auto",  # Auto-detect size from img2img
                "cfg_scale": 25,
                "strength": 0.5,
                "seed": random.randint(0, 2**32 - 1),  # Random seed
                "controlnet": {
                    "unit_0": {
                        "enabled": True,
                        "control_type": "all",
                        "control_weight": 1,
                        "start_control_step": 0,
                        "end_control_step": 1,
                        "control_mode": "balanced",
                        "batch_option": "all_units_for_all_images"
                    }
                }
            }
        except Exception as e:
            print("Error setting override_settings: ", e)
            inference_request['override_settings'] = {
                "sd_model_checkpoint": 'model_indoor.safetensors'
            }
        
        json_data = automatic_session.post(
            url=f'http://127.0.0.1:3000/sdapi/v1/{endpoint}',
            json=inference_request,
            timeout=600
        ).json()

        # Face swapping logic if applicable
        try:
            roop_img = inference_request['roop_img']
            for i in json_data['images']:
                index_of_i = json_data['images'].index(i)
                payload = {
                    "source_image": roop_img,
                    "target_image": i,
                    "source_faces_index": [0],
                    "face_index": [0],
                    "upscaler": "4x_Struzan_300000",
                    "scale": 2,
                    "upscale_visibility": 1,
                    "face_restorer": "CodeFormer",
                    "restorer_visibility": 1,
                    "restore_first": 1,
                    "model": "inswapper_128.onnx",
                    "gender_source": 0,
                    "gender_target": 0,
                    "save_to_file": 0,
                    "result_file_path": ""
                }
                face_swaped_image = requests.post(
                    'http://127.0.0.1:3000/reactor/image',
                    headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                    json=payload
                ).json()['image']

                json_data['images'][index_of_i] = face_swaped_image
            return json_data
        except Exception as e:
            print("Error during face-swapping: ", e)
            return json_data
    else:
        return {'error'}


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    # Run inference on the input event
    json_result = run_inference(event["input"])

    # Return the output that you want to be returned, such as pre-signed URLs to output artifacts
    return json_result


if __name__ == "__main__":
    # Wait until the service is ready
    wait_for_service(url='http://127.0.0.1:3000/sdapi/v1/txt2img')

    print("WebUI API Service is ready. Starting RunPod...")

    # Start the RunPod serverless function with the handler function
    runpod.serverless.start({"handler": handler})
