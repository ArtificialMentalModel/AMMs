import sys
import subprocess
import os
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "transformers"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "redis"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "sentence_transformers"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface-hub"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "torch"])
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from datasets import load_dataset
from huggingface_hub import login
import redis
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


# Add your Hugging Face token here
login(token=os.getenv('HF_TOKEN', ''))



ITEM_QUESTION_EMBEDDING_FIELD='item_question_vector'
ITEM_ANSWER_EMBEDDING_FIELD='item_answer_vector'





model_sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

redis_conn = redis.Redis(
  host=os.getenv('REDIS_HOST', 'localhost'),
  port=int(os.getenv('REDIS_PORT', 6379)),
  password=os.getenv('REDIS_PASSWORD', ''))
 

# Load LLaMA model and tokenizer
#model_name = "meta-llama/Llama-2-13b-chat-hf"
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')




def question_answer(query, context):

    prompt = f"""You are a specialized healthcare bot designed to provide guidance and support for leg and knee issues. I will provide you with the questions and answers people talked about related to the user query and then the actual user query you need to answer. Based on this information, you will respond to user queries with accurate, professional, and clear guidance. Follow these instructions:

    1. Use appropriate medical terminology: If a term is technical, explain it in simple terms for the user when needed.

    2. Base advice on data provided: Ensure that your answers consider any specific symptoms, or treatments shared in the data input. Tailor responses to individual situations.

    3. Clarify when something requires further medical consultation: While offering advice, make it clear when the user should consult with a healthcare provider for further evaluation.

    4. Avoid making definitive diagnoses: Provide possible explanations or conditions based on the symptoms, but avoid giving an exact diagnosis. Suggest potential causes and treatments while recommending further consultation for accuracy.

    5. Show empathy and concern: Your responses should not only be informative but also reassuring. Consider the user's emotional state and provide comfort where appropriate.
    
    
    \n\nUser Query:\n{query}\n\nContext:\n{context}\n\n### Response:"""
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    print("inputs")
    print(inputs)
    outputs = model.generate(input_ids=inputs["input_ids"].to('cuda'), attention_mask=inputs["attention_mask"], max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
    print("outputs")
    print(outputs)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print('output_text', output_text)

    print('Response: ', output_text)

    
    return output_text



topK=2
query="""I have a patient who is around 55 years old, desk worker, never done any sport, never overweight.
He comes because of pain around TH12, which only occurs after a while when lying down, but so severe that he has to sleep elevated."""

#vectorize the query
query_vector = model_sentence_transformer.encode(query).astype(np.float32).tobytes()

#prepare the query
#q = Query(f'(@INDEX:{{LEG|KNEE}})=>=>[KNN {topK} @{ITEM_QUESTION_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','INDEX', 'Question', 'Answer').dialect(2)
#params_dict = {"vec_param": query_vector}

q = Query(f'(@INDEX:{{KNEE}})=>[KNN {topK} @{ITEM_QUESTION_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','INDEX', 'Question', 'Answer').dialect(2)
params_dict = {"vec_param": query_vector}


#Execute the query
results = redis_conn.ft('QA_index').search(q, query_params = params_dict)

result_string = ""
#Print similar products found
for qa in results.docs:
    print('***************Q/A found ************')
    result_string = result_string + "Question:\n"+qa.Question+"\nAnswer:\n"+qa.Answer+"\n\n"


print(result_string)
response = question_answer(query, result_string)

print(response)
