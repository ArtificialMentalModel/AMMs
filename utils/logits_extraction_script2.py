import sys
import subprocess
import os
import pandas as pd  # Added pandas import for Excel handling
import ast
import json
from get_prompts import prompts

installation_path = "/netscratch/muhkhan"
cache_dir = '/netscratch/muhkhan/.cache/huggingface/hub'

os.makedirs(cache_dir, exist_ok=True)

os.environ['HF_HOME'] = cache_dir
os.environ['XDG_CACHE_HOME'] = cache_dir

sys.path.insert(0, installation_path)

#subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", installation_path, "transformers"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", installation_path, "redis"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", installation_path, "sentence-transformers"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", installation_path, "torch"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", installation_path, "llama-cpp-python==0.1.78"])
#llama-cpp-python==0.1.78
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "transformers"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "redis"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "sentence-transformers"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "numpy"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface-hub"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "torch"])

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
#from datasets import load_dataset
from huggingface_hub import login
import redis
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from llama_cpp import Llama
import torch.nn.functional as F
from token_predictor_logits import predict_correct_token

#model = Llama(model_path="/netscratch/muhkhan/llama-7b.ggmlv3.q4_0.bin")
#model = AutoModel.from_pretrained("TheBloke/LLaMa-7B-GGML")
ITEM_QUESTION_EMBEDDING_FIELD='item_question_vector'
ITEM_ANSWER_EMBEDDING_FIELD='item_answer_vector'

#login(token=os.getenv('HF_TOKEN', ''))
print('sentence transformers downloading')
model_sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#redis_conn = redis.Redis(
#  host='your-redis-host',
#  port=your-redis-port,
#  password='your-redis-password'
#)
# Add your Hugging Face token here
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
# Load LLaMA model and tokenizer
#model_name = "meta-llama/Llama-2-13b-chat-hf"
#model_name = "ebadullah371/llama_2_7b_ft_chat_new_10_v1"
#model_name = "meta-llama/Llama-3.1-70B-Instruct"
#model_name = "meta-llama/Llama-3.1-70B-Instruct"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
print('Llama tokenizer')

tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HF_TOKEN'], cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=os.environ['HF_TOKEN'],
    cache_dir=cache_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Tokenizer and model initialized successfully.")
print('Model loaded.')

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def question_answer(prompt):
    #inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    messages=[
        {"role": "system", "content": prompts()},
        {"role": "user", "content": "How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5"},
        {"role": "assistant", "content": """{"score": 1}"""
        },
        {"role": "user", "content": "How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5"+ "\n\n" + prompt}

    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
    outputs = model.generate(
        #inputs["input_ids"].to('cuda')
        input_ids=input_ids,
        top_p=0.9,
        temperature=0.9,
        do_sample=False,
        max_new_tokens=500,
        return_dict_in_generate=True,
        output_scores=True
    )
    #output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
   # outputs = tokenizer.batch_decode(generated_ids)
    # Collect top-k words and their logits for each step
    top_k_words = []
    top_k = 5
    #print(outputs.scores)
    output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print('Response: ', output_text)
    generated_tokens = outputs.sequences[0][input_ids.size(1):]

    for i, token_id in enumerate(generated_tokens):
        token = tokenizer.decode(token_id)
        logits = outputs.scores[i][0]  # Logits for the current token
        probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        token_prob = probabilities[token_id].item()
        result = {
            'token': token,
            'logit': logits[token_id].item(),
            'probability': token_prob
        }
        #print(result)
    #result = []
    #for i, token_id in enumerate(generated_tokens):
    #    token = tokenizer.decode(token_id)
    #    logits = outputs.scores[i][0]  # Logits for the current token
    #    probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities

    #    # Get the top_k alternatives
    #    top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
    #    alternatives = []
    #    for prob, idx in zip(top_k_probs, top_k_indices):
    #        alternatives.append({
    #            'token': tokenizer.decode(idx),
    #            'probability': prob.item()
    #        })

    #    result.append({
    #        'token': token,
    #        'alternatives': alternatives
    #    })
    result = []
    allowed_tokens = {'1', '2', '3', '4', '5'}  # Tokens of interest

    for i, token_id in enumerate(generated_tokens):
        token = tokenizer.decode(token_id)
        
        # Check if the token is in the allowed set
        if token in allowed_tokens:
            logits = outputs.scores[i][0]  # Logits for the current token
            probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities

            # Get the top_k alternatives
            top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
            alternatives = []
            for prob, idx in zip(top_k_probs, top_k_indices):
                alternatives.append({
                    'token': tokenizer.decode(idx),
                    'probability': prob.item()
                })

            # Append only if the token is in the allowed set
            result.append({
                'token': token,
                'alternatives': alternatives
            })

    print([result[0]])
    return output_text, [result[0]]



def extract_entity(text):
    int_idx = text.find('### Response: [/INST]')
    text_class = text[int_idx+len('### Response: [/INST]'):]   
    return text_class.strip()

def extract_entity_v2(text):
    start_idx = text.rfind('<assistant>') + len('<assistant>')
    end_idx = text.rfind('</assistant>')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        text_class = text[start_idx:end_idx].strip()
        return text_class
    return None

def extract_entity_v3(text):
    start_tag = '<assistant>'
    end_tag = '</assistant>'
    
    start_idx = text.find(start_tag)
    while start_idx != -1:
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx != -1:
            entity = text[start_idx:end_idx].strip()
            # Validate that it matches the expected dictionary-like structure
            if entity.startswith("{") and entity.endswith("}"):
                return entity
        # Move to the next occurrence
        start_idx = text.find(start_tag, end_idx)
    
    return None
import json

def extract_entity_v4(text):
    keyword = "assistant"
    keyword_idx = text.find(keyword)
    
    if keyword_idx != -1:
        # Find the start of the dictionary
        start_idx = text.find("{", keyword_idx)
        if start_idx != -1:
            # Find the end of the dictionary
            end_idx = text.find("}", start_idx)
            if end_idx != -1:
                end_idx += 1  # Include the closing brace
                entity = text[start_idx:end_idx].strip()
                try:
                    # Validate that it is a dictionary
                    parsed_entity = json.loads(entity)
                    if isinstance(parsed_entity, dict):
                        return parsed_entity
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON structures
    
    return None


topK=3
query="How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5."

# File paths
json_file = 'synthetic_data_updated_file.json'  # Replace with your JSON file path
output_csv = 'Synthetic_data_logits_llama3_8b_fewshot.csv'

# Load JSON data
with open(json_file, 'r') as file:
    data = json.load(file)

# Initialize a DataFrame for output
data_list = []

# Lookup table for reverse scores
lookup_table = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}

# Process each entry in JSON data
count = 1
for sample in data:
    Age = sample.get("Q1", "Unknown")
    Gender = sample.get("Q2", "Unknown")
    Employment_status = sample.get("Q3", "Unknown")
    Current_emotional_state = sample.get("Q6", "Unknown")
    Have_any_physical_disabilities = sample.get("Q7", "Unknown")
    Type_of_physical_activities = sample.get("Q8", "Unknown")
    How_many_days_do_you_do_exercise = sample.get("Q9", "Unknown")
    Overall_health_status = sample.get("Q12", "Unknown")
    Current_mobility = sample.get("Q13", "Unknown")
    question_answer_string = sample.get("Q32", "Unknown")

    # Calculate personality traits
    try:
        Q26 = lookup_table.get(int(sample.get("Q26").split('-')[0].strip()))
        Q21 = int(sample.get("Q21").split('-')[0].strip())
        Extroversion = (Q26 + Q21) / 2

        Q22 = lookup_table.get(int(sample.get("Q22").split('-')[0].strip()))
        Q27 = int(sample.get("Q27").split('-')[0].strip())
        Agreeableness = (Q22 + Q27) / 2

        Q28 = lookup_table.get(int(sample.get("Q28").split('-')[0].strip()))
        Q23 = int(sample.get("Q23").split('-')[0].strip())
        Conscientiousness = (Q28 + Q23) / 2

        Q24 = lookup_table.get(int(sample.get("Q24").split('-')[0].strip()))
        Q29 = int(sample.get("Q29").split('-')[0].strip())
        Emotional_Stability = (Q24 + Q29) / 2

        Q30 = lookup_table.get(int(sample.get("Q30").split('-')[0].strip()))
        Q25 = int(sample.get("Q25").split('-')[0].strip())
        Openness = (Q30 + Q25) / 2
    except (ValueError, AttributeError):
        Extroversion = Agreeableness = Conscientiousness = Emotional_Stability = Openness = "Unknown"


    prompt3 = f"""
        <system>
        You are a helpful assistant. 

    You will be given user data including the Big 5 personality traits and their demographic and health information. Based on this, the user is asked about the difficulty of performing an exercise and provides their own rating. Your job is to reflect the user's score based on their data.

    Remember:
    - The user is always correct.
    - The question: "How difficult does it look to perform 10 squats?"
    - Rate difficulty from 1 to 5.
    - You have the following user data:

    Extroversion: {Extroversion}/7 (General Norm: 4.44)
    Agreeableness: {Agreeableness}/7 (General Norm: 5.23)
    Conscientiousness: {Conscientiousness}/7 (General Norm: 5.4)
    Emotional Stability: {Emotional_Stability}/7 (General Norm: 4.83)
    Openness: {Openness}/7 (General Norm: 5.38)

    Age: {Age}
    Gender: {Gender}
    Employment status: {Employment_status}
    Current emotional state: {Current_emotional_state}
    Have any physical disabilities: {Have_any_physical_disabilities}
    Type of physical activities: {Type_of_physical_activities}
    How many days do you do exercise: {How_many_days_do_you_do_exercise}
    Overall health status: {Overall_health_status}
    Current mobility: {Current_mobility}

    User Response to the question: {question_answer_string}

    JUST Return a json like below:
    {{ 
        "score": 1 to 5
    }}
        </system>
        <user>
        How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5.</user>
        <assistant>
        """
    prompt = f"""
    Based on the Big 5 personality traits, here are the personalized scores of a user. I have also attached the general norm and their definitions. I will also give you the user's age, gender, profession, current emotional state, if they have physical disabilities or not, what type of physical activities they participate in, how much they exercise, their overall health status, their mobility status, etc.  Your task is to give me a rating to a question that this user would rate themselves based on these below traits. Remember, the idea is to predict what the user would do and see the Artificial Mental Model part of the person. \n\n
         
            I will give you user response to the question too. 

            I will give you some data in German. Give whole response in English.

            Remember, the user is always right.
         
            Just tell if the user will be able to perform the exercise (Yes or No) with a score you rate that user (1 being not difficult at all and 5 being extremely difficult). Also see what the user rated themselves.
         
            See the Big 5 scores and see the Artificial Mental Model of the user below. 
            Here is everything you need to know:
            Do not assume anything about the user. Only consider what is given to you below.\n\n
            •       Extroversion: {Extroversion}/7 (High) (General Norm: 4.44)
            •       Agreeableness: {Agreeableness}/7 (Medium Low) (General Norm: 5.23)
            •       Conscientiousness: {Conscientiousness}/7 (Medium High) (General Norm: 5.4)
            •       Emotional Stability: {Emotional_Stability}/7 (Medium High) (General Norm: 4.83)
            •       Openness: {Openness}/7 (High) (General Norm: 5.38)
            Here’s a brief overview of what each of the Big Five personality traits represents:
            •       Extroversion: Reflects your level of sociability and enthusiasm.
            •       Agreeableness: Indicates how cooperative and kind-hearted you are.
            •       Conscientiousness: Measures your reliability and attention to detail.
            •       Emotional Stability: Assesses your ability to remain stable and composed.
            •       Openness: Shows your willingness to embrace new experiences and ideas.\n\n

            Age: {Age}\n
            Gender: {Gender}\n
            Employment status: {Employment_status}\n
            Current emotional state: {Current_emotional_state}\n
            Have any physical disabilities: {Have_any_physical_disabilities}\n
            Type of physical activities: {Type_of_physical_activities}\n
            How many days do you do exercise: {How_many_days_do_you_do_exercise}\n
            Overall health status: {Overall_health_status}\n
            Current mobility: {Current_mobility}\n\n
         

            User Response to the question: {question_answer_string}
         
            JUST Return a json like below:
            {{ 
                "score": 1 to 5
            }}

    """
    print('****************************************')
    #print(prompt)
    print('****************************************')
    # Generate the model's response
    output_text, tokens = question_answer(prompt)

    #print(prompt)
    ## logits prompt
    #tokens_converted = ast.literal_eval(tokens)
    #tokens_ = [{"token": tokens_converted['token'], 'alternatives': tokens_converted['alternatives'][:2]}]
    #logit_prompt = predict_correct_token(tokens)

    #print(tokens)
    #output_text2, tokens2 = question_answer(logit_prompt)
    #print(tokens2)

    for i in range(3):
        print("IN FOR LOOP")
        if any(char.isdigit() for char in output_text):
            print("OUTPUT OK!")
            break
        else:
            print(f"{i}'th CALL")
            output_text, tokens = question_answer(prompt)

    print('OUTPUT RAW TEXT')
    print(output_text)
    # output_text, tokens = question_answer(prompt)
    #output_text = question_answer(prompt)
    #print(output_text)
    # output_text = extract_entity(output_text)
    output_text = extract_entity_v4(output_text)
    #output_text2 = extract_entity_v2(output_text2)
    print('OUTPUTTTT START')
    print(output_text)
    print(tokens)
    print(count)
    count += 1
    print('OUTPUTTTT END')
    data_list.append({
        "Actual Output": question_answer_string,
        "Model_Output": output_text,
        "Logits": tokens,
        "Prompt": prompt3
    })

df = pd.DataFrame(data_list)
df.to_csv(output_csv, index=False)

print(f"Data processing complete. Results saved to {output_csv}.")
