from openai import OpenAI
import os
from datetime import datetime
import json
from dotenv import load_dotenv
from ThreadObject import ThreadObject
load_dotenv(override=True)


def save_threads_to_file(msg_objs:ThreadObject, filename="threads.json"):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump([msg_obj.to_dict() for msg_obj in msg_objs], file, indent=4, ensure_ascii=False)

def load_threads_from_file(filename="threads.json"):
    with open(filename, 'r', encoding='utf-8') as file:
        msg_objs_data = json.load(file)
        return [ThreadObject.from_dict(data) for data in msg_objs_data]


# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_all_messages(thread_id):
    messages = []
    has_more = True
    after = None
    last_created_at = None

    while has_more:
        response = client.beta.threads.messages.list(thread_id=thread_id, after=after)
        messages.extend(response.data)
        has_more = response.has_more
        if has_more:
            after = response.last_id

    message_list = []
    run_ids = []
    model = "Not Found"
    last_created_at = None
    for message in reversed(messages):
        if message.role in ["assistant", "user"]:
            for content_item in message.content:
                if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                    tokens=0
                    if hasattr(message, 'run_id') and message.run_id not in run_ids and message.run_id is not None:
                        run_ids.append(message.run_id)
                        run_details = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=message.run_id)
                        tokens += run_details.usage.total_tokens
                        model = run_details.model
                    message_list.append({"role": message.role, "message": content_item.text.value, "tokens": tokens})
        first_created_at = message.created_at
        if last_created_at is None:
            last_created_at = message.created_at


            
    return message_list, last_created_at, first_created_at, model


def build_msg_obj(thread_id):
    messages_list, started_time, finished_time, model = fetch_all_messages(thread_id)
    started_time = datetime.fromtimestamp(started_time).strftime('%Y%m%d%H%M%S')
    finished_time = datetime.fromtimestamp(finished_time).strftime('%Y%m%d%H%M%S')
    msg_obj = ThreadObject(
        thread_id=thread_id,
        messages=messages_list,
        started_time=started_time,
        finished_time=finished_time,
        model=model
    )
    return msg_obj

def process_threads_array(threads_array):
    msg_objs = []
    for thread_id in threads_array:
        msg_obj = build_msg_obj(thread_id)
        msg_objs.append(msg_obj)
        print(f"Thread {thread_id} processed successfully")

    save_threads_to_file(msg_objs)

THREADS = """thread_IVkHiheVmFWrBjHmrebioZcN
thread_4OBTOTUoH1XoYU79rmJog418
thread_7khxfCmJWGdrGxW6x6WJkjZ9
thread_AzACzSotBxVFvPZsJRuMAeNz
thread_ogI7Ug8hxOKSBOMLaElLFQX3
thread_HtQmyestgJC5qmHBAQCxXvC3
thread_z6CvtO2OgHZJdGMS70OlyMhu
thread_nb8uKqcV1FSkF1Ffx9WM99rD
thread_PeGELsH7Ma29HTcMTQdjm61y
thread_qhvMCm7jK3J0i963yguVkLlE
thread_PAudSphEK3hrRIpmpUItujUU
thread_7VeRhaGiGHylNCm8S1NQ06l0
thread_eDMIbx5vJYlnMD43OkKADOwl
thread_TiqYlOUpsWTlcsX2hNjJdAtR
thread_plYC6cZcKsdlKYqcQNNNh9Fi
thread_xUs9uTwYahZtyxJe1G9JWhhf
thread_z91orLaP042lj04CleU2oSMB
thread_i2TD50CKC8TKlSf6J3CscLtj
thread_CZMW4VWhxrKWvUv1iNjvfeJt
thread_5UgJ3jovLpI72x7Owx3ovmeG
thread_Gp984R2DuONmpyXOgh5XLEWx
thread_ohRJIQMObfR6jKWkgaU47M4V
thread_SjGrQq1hRFNHGvgJVslH2Yrf
thread_m0HRDPOOnRbsSGpaSYG5h8ol
thread_mcqhlaluShCmIrzAPhHzDFvf
thread_P5qu0nplY4IvQkViHa4aLvaU
thread_Lhb7UzcpULhSQAMOMQ8Y8Lyr
thread_HWZEM9jul3guVwhY2UT2RP0S
thread_9NK7pFE9VMAjp3hsEMeFIvOD
thread_6tBB04N7esQ7C4PqhD8noZ1n
thread_UUuNCqQl1ovE9D3NZvcsNvWB
thread_o4v7677UOVTWDN82R2UBTbhI
thread_8QMLqdELWVwXVeSDaNjxEeWT
thread_JKC6mBxmRBDiHfvO17QyPNNu
thread_ZSpBIyA06YWqkKKQYbsje007
thread_lOfjdwZ9NRLeFzhBbeHD4tnE
thread_XZq8w9pr6kUI0KMF0SpFzClu
thread_mmysZvlpMKI7nTg7Vn9Z2yux
thread_5NLPkrgHrFev7nazhofvhIXD
thread_WXfE6azEWLITQvLCTEmf4V3B
thread_3ht2IBOyb1KgXhOatbjRcbwA
thread_8XJniEX5lmxvKxbMGyxdmcAf
thread_PGr0ND9S0xn8Qpbktxwt4mOJ
thread_SuhGNUhMgcjFpqMjeA4hurBJ
thread_3ISaw80y5RTfc4b80szVTBbB
thread_PeaIL500kesFaSoKqWAlGQ7A
thread_rHuE7Swk5lwdMumF4A04rdOI
thread_z9Aiapyl1HY0xniezanrttve
thread_c4mKVxWXHwyYItzal0qTbzdM
thread_EYuOS7Vq08pIF3rZwO8lBkgX
thread_Vu57C6iQvrK6fXPoSq0RXwnx
thread_y81pN3trqruSsXWmsCx9eykL
thread_xdBLZAD6exRzNrXOCyCsgiZK
thread_Lsihq4xyIqJmhQo5gCJ3RcLg
thread_GOEPWnDsVehaNBlalzRiTwr1
thread_FQFEbjhK7XDvlqgxxHCucQSL
thread_rIFgP6MGNTphevqPtx72CRUR
thread_rYdGvz9ZNDKFMRCdlN2va8s7
thread_e12cYfu4fxbeN7uXNu8X3Dz3"""

threads_array = THREADS.split("\n")


# process_threads_array(threads_array)

