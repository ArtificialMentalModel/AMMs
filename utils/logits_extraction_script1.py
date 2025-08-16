import sys
import subprocess
import os
import pandas as pd  # Added pandas import for Excel handling
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

#model = Llama(model_path="/netscratch/muhkhan/llama-7b.ggmlv3.q4_0.bin")
#model = AutoModel.from_pretrained("TheBloke/LLaMa-7B-GGML")
ITEM_QUESTION_EMBEDDING_FIELD='item_question_vector'
ITEM_ANSWER_EMBEDDING_FIELD='item_answer_vector'

#login(token=os.getenv('HF_TOKEN', ''))
print('sentence transformers downloading')
model_sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

redis_conn = redis.Redis(
  host=os.getenv('REDIS_HOST', 'localhost'),
  port=int(os.getenv('REDIS_PORT', 6379)),
  password=os.getenv('REDIS_PASSWORD', '')
)
# Add your Hugging Face token here
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
# Load LLaMA model and tokenizer
#model_name = "meta-llama/Llama-2-13b-chat-hf"
#model_name = "ebadullah371/llama_2_7b_ft_chat_new_10_v1"
#model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name = "meta-llama/Llama-3.1-70B-Instruct"
model_name = "ebadullah371/llama_3_8b_ft_10_chat_v2"
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
        {"role": "assistant", "content": """{ 
                "score": 1 to 5
            }"""
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

    print(result)
    return output_text, result



def extract_entity(text):
    int_idx = text.find('Response:')
    text_class = text[int_idx+len('Response:'):]   
    return text_class.strip()
topK=3
query="How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5."

excel_file = 'Discover Your Inner Self - Deutsch_June 10, 2024_06.43 - Generated (8).xlsx'
tipi_df = pd.read_excel(excel_file, sheet_name='TIPI')
data_df = pd.read_excel(excel_file, sheet_name='Data')


tipi_df['Model_Output'] = ''
tipi_df['Prompt'] = ''
tipi_df['Logits'] = ''

for index, row in tipi_df.iterrows():
    Extroversion = row['Extroversion']
    Agreeableness = row['Agreeableness']
    Conscientiousness = row['Conscientiousness']
    Emotional_Stability = row['Emotional Stability']
    Openness = row['Openness']

    data_index = index + 2
    if data_index >= len(data_df):
        break 
    data_row = data_df.iloc[data_index]

    Age = data_row['Q1']
    Gender = data_row['Q2']
    Employment_status = data_row['Q3']
    Current_emotional_state = data_row['Q6']
    Have_any_physical_disabilities = data_row['Q7']
    Type_of_physical_activities = data_row['Q8']
    How_many_days_do_you_do_exercise = data_row['Q9']
    Overall_health_status = data_row['Q12']
    Current_mobility = data_row['Q13']
    question_1 = data_row['Q33']
    question_2 = data_row['Q35']
    question_answer_string = data_row['Q32']
    # Vectorize the query
    query_vector = model_sentence_transformer.encode(query).astype(np.float32).tobytes()

    # Prepare the query
    q = Query(f'(@INDEX:{{KNEE}})=>[KNN {topK} @{ITEM_QUESTION_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','INDEX', 'Question', 'Answer').dialect(2)
    params_dict = {"vec_param": query_vector}

    # Execute the query
    results = redis_conn.ft('QA_index').search(q, query_params = params_dict)

    result_string = ""
    # Print similar products found
    for row in results.docs:
        print('***************Q/A found ************')
        #print(row.INDEX)
        result_string = result_string + "Question:\n"+row.Question+"\nAnswer:\n"+row.Answer+"\n\n"
        #print(row.Question)
        #print(row.Answer)
        #print(row.vector_score)

    context = ""  # Set the context

    prompt = f"""Based on the Big 5 personality traits, here are the personalized scores of a user. I have also attached the general norm and their definitions. I will also give you the user's age, gender, profession, current emotional state, if they have physical disabilities or not, what type of physical activities they participate in, how much they exercise, their overall health status, their mobility status, etc.  Your task is to give me a rating to a question that this user would rate themselves based on these below traits. Remember, the idea is to predict what the user would do and see the Artificial Mental Model part of the person. \n\n
         
            I will give you user response to the question too. 

            I will give you some data in German. Give whole response in English.

            Remember, the user is always right.
         
            Just tell if the user will be able to perform the exercise with a score you rate that user (1 being not difficult at all and 5 being extremely difficult). Also see what the user rated themselves.
         
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
    User Query:\n{query}\n\n### Response:
    """

    prompt2 = f"""
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Instruction:
        How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5. No explanation is needed. JUST RETURN 1 SCORE, NOTHING ELSE.\n\n
        ### Input:
        Extroversion: {Extroversion}/7 (High) (General Norm: 4.44)
        Agreeableness: {Agreeableness}/7 (Medium Low) (General Norm: 5.23)
        Conscientiousness: {Conscientiousness}/7 (Medium High) (General Norm: 5.4)
        Emotional Stability: {Emotional_Stability}/7 (Medium High) (General Norm: 4.83)
        Openness: {Openness}/7 (High) (General Norm: 5.38)
        Age: {Age}\n
        Gender: {Gender}\n
        Employment status: {Employment_status}\n
        Current emotional state: {Current_emotional_state}\n
        Have any physical disabilities: {Have_any_physical_disabilities}\n
        Type of physical activities: {Type_of_physical_activities}\n
        How many days do you do exercise: {How_many_days_do_you_do_exercise}\n
        Overall health status: {Overall_health_status}\n
        Current mobility: {Current_mobility}\n\n
        ### Response:
        """
    print('****************************************')
    #print(prompt)
    print('****************************************')


    output_text, tokens = question_answer(prompt)
    #output_text = question_answer(prompt)
    print("Top-K words for each token in the response:")
    
    output_text = extract_entity(output_text)
    print('OUTPUTTTT START')
    print(output_text)
    print('OUTPUTTTT END')
    tipi_df.at[index, 'Model_Output'] = output_text
    tipi_df.at[index, 'Prompt'] = prompt
    tipi_df.at[index, 'Logits'] = tokens

with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    tipi_df.to_excel(writer, sheet_name='TIPI', index=False)

print("Processing complete. The outputs have been saved in the 'Model_Output' column of the TIPI sheet.")
