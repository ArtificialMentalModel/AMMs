import sys
import subprocess
import os


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



#model = Llama(model_path="/netscratch/muhkhan/llama-7b.ggmlv3.q4_0.bin")
#model = AutoModel.from_pretrained("TheBloke/LLaMa-7B-GGML")
ITEM_QUESTION_EMBEDDING_FIELD='item_question_vector'
ITEM_ANSWER_EMBEDDING_FIELD='item_answer_vector'




# Add your Hugging Face token here
login(token=os.getenv('HF_TOKEN', ''))
print('sentence tarsnformers downloading')
model_sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

redis_conn = redis.Redis(
  host=os.getenv('REDIS_HOST', 'localhost'),
  port=int(os.getenv('REDIS_PORT', 6379)),
  password=os.getenv('REDIS_PASSWORD', ''))


# Load LLaMA model and tokenizer
#model_name = "meta-llama/Llama-2-13b-chat-hf"
model_name = "meta-llama/Llama-2-7b-chat-hf"
print('Llama tokenizer')

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_auth_token=os.getenv('HF_TOKEN', ''))
print('Llama model')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, use_auth_token=os.getenv('HF_TOKEN', '')).to('cuda')


personality_scores_list = [
    [1.5, 6, 3, 3, 5],
    [4.5, 4.5, 6, 3, 5],
    [5, 6, 6.5, 5.5, 6.5],
    [4, 5, 3.5, 5, 6],
    [3.5, 6.5, 3.5, 3, 7]
]

# Function to dynamically generate the prompt based on personality scores
def question_answer(query, context, scores):
    # Define the Big Five trait names and general norms
    traits = [
        ["Extroversion", 4.44],
        ["Agreeableness", 5.23],
        ["Conscientiousness", 5.4],
        ["Emotional Stability", 4.83],
        ["Openness", 5.38]
    ]
    
    # Build the prompt dynamically by inserting the user's scores and general norms
    prompt = f"""Based on the Big 5 personality traits, here are the personalized scores of a user. I have also attached the general norm and their definitions. Your task is to give me a rating to a question that this user would rate themselves based on these below traits. Remember, the idea is to predict what the user would do and see the Artificial Mental Model part of the person. Also, I will give you some context regarding the question asked. If need be, you can use that too. Just return the score from 1 to 5 that you think the user will give. 1 means not difficult at all, 2 means Slightly difficult, 3 means Moderately difficult, 4 means Very difficult and 5 means Extremely difficult.\n\n"""
    
    for i, trait in enumerate(traits):
        trait_name, norm = trait
        user_score = scores[i]
        prompt += f"• {trait_name}: {user_score}/7 (User Score) (General Norm: {norm})\n"
    
    prompt += """\nHere’s a brief overview of what each of the Big Five personality traits represents:
        • Extroversion: Reflects your level of sociability and enthusiasm.
        • Agreeableness: Indicates how cooperative and kind-hearted you are.
        • Conscientiousness: Measures your reliability and attention to detail.
        • Emotional Stability: Assesses your ability to remain stable and composed.
        • Openness: Shows your willingness to embrace new experiences and ideas.
    \n\nUser Query:\n{query}\n\nContext:\n{context}\n\n### Response:"""
    
    # Tokenize and send the prompt to the model
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(input_ids=inputs["input_ids"].to('cuda'), attention_mask=inputs["attention_mask"], max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print('Response:', output_text)
    
    return output_text


# Example usage
topK = 3
query = "How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5."

# Vectorize the query
query_vector = model_sentence_transformer.encode(query).astype(np.float32).tobytes()

# Prepare the query for Redis
q = Query(f'(@INDEX:{{KNEE}})=>[KNN {topK} @{ITEM_QUESTION_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0, topK).return_fields('vector_score', 'INDEX', 'Question', 'Answer').dialect(2)
params_dict = {"vec_param": query_vector}

# Execute the query on Redis
results = redis_conn.ft('QA_index').search(q, query_params=params_dict)

result_string = ""
# Print similar products found
for product in results.docs:
    result_string += f"Question:\n{product.Question}\nAnswer:\n{product.Answer}\n\n"

# Take the first sublist as the personality scores for this example
for score in personality_scores_list:

    # Get the response by passing the query, context, and dynamic scores
    response = question_answer(query, result_string, score)

    print(response)