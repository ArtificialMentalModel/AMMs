


import sys
import subprocess
import os
import random
import json

# subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "trl"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

import time
from random import randrange, sample, seed
import os
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import numpy as np
from rouge_score import rouge_scorer
import sys

installation_path = "/netscratch/muhkhan"
cache_dir = '/netscratch/muhkhan/.cache/huggingface/hub'

os.makedirs(cache_dir, exist_ok=True)

os.environ['HF_HOME'] = cache_dir
os.environ['XDG_CACHE_HOME'] = cache_dir

sys.path.insert(0, installation_path)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add your Hugging Face token to the environment variables
# Add your Hugging Face token here
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

# Load dataset
with open("synthetic_data_updated_file.json", "r") as file:
    json_data = json.load(file)

dataset = Dataset.from_dict({
    key: [row.get(key) for row in json_data] for key in json_data[0].keys()
})
print(f"Dataset size: {len(dataset)}")
print(f"Sample: {dataset[0]}")


def format_instruction(sample):
    # Convert raw features to their respective scores
    lookup_table = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
    Age = sample.get("Q1", "Unknown")
    Gender = sample.get("Q2", "Unknown")
    Employment_status = sample.get("Q3", "Unknown")
    Current_emotional_state = sample.get("Q6", "Unknown")
    Have_any_physical_disabilities = sample.get("Q7", "Unknown")
    Type_of_physical_activities = sample.get("Q8", "Unknown")
    How_many_days_do_you_do_exercise = sample.get("Q9", "Unknown")
    Overall_health_status = sample.get("Q12", "Unknown")
    Current_mobility = sample.get("Q13", "Unknown")

    Expectation_Alignment_Score = sample.get("Expectation Alignment Score", "Unknown")
    Pre_Surgery_Anxiety_Level = sample.get("Pre-Surgery Anxiety Level", "Unknown")
    WOMAC_Pain_Subscale = sample.get("WOMAC Pain Subscale", "Unknown")
    VAS_Pre_Surgery_Pain_Score = sample.get("VAS Pre-Surgery Pain Score", "Unknown")
    Post_Surgery_Pain_Reduction_Expectation = sample.get("Post-Surgery Pain Reduction Expectation", "Unknown")
    KOOS_Functionality_Subscale = sample.get("KOOS Functionality Subscale", "Unknown")
    Time_to_Expected_Mobility = sample.get("Time to Expected Mobility (days)", "Unknown")
    Lysholm_Score_Baseline = sample.get("Lysholm Score Baseline", "Unknown")
    IKDC_SKF_Baseline_Score = sample.get("IKDC-SKF Baseline Score", "Unknown")
    ACL_RSI_Score_Pre_Surgery = sample.get("ACL-RSI Score Pre-Surgery", "Unknown")
    MPFL_RSI_Pre_Surgery_Score = sample.get("MPFL-RSI Pre-Surgery Score", "Unknown")
    Support_System_Availability = sample.get("Support System Availability", "Unknown")
    Rehabilitation_Commitment = sample.get("Rehabilitation Commitment (hours/week)", "Unknown")
    question_answer_string = sample.get("Q32", "Unknown")

    question_1 = sample.get("Q33", "Unknown")
    question_2 = sample.get("Q35", "Unknown")

    Q26 = lookup_table.get(int(sample.get("Q26").split('-')[0].strip()))
    Q21 = int(sample.get("Q21").split('-')[0].strip())
    Extroversion = (Q26 + Q21)/2

    Q22 = lookup_table.get(int(sample.get("Q22").split('-')[0].strip()))
    Q27 = int(sample.get("Q27").split('-')[0].strip())
    Agreeableness = (Q22 + Q27)/2

    Q28 = lookup_table.get(int(sample.get("Q28").split('-')[0].strip()))
    Q23 = int(sample.get("Q23").split('-')[0].strip())
    Conscientiousness = (Q28 + Q23)/2

    Q24 = lookup_table.get(int(sample.get("Q24").split('-')[0].strip()))
    Q29 = int(sample.get("Q29").split('-')[0].strip())
    Emotional_Stability = (Q24 + Q29)/2

    Q30 = lookup_table.get(int(sample.get("Q30").split('-')[0].strip()))
    Q25 = int(sample.get("Q25").split('-')[0].strip())
    Openness = (Q30 + Q25)/2

    # Prepare system message (instructions)
    system_message = f"""
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
        Expectation Alignment Score: {Expectation_Alignment_Score}
        Pre-Surgery Anxiety Level: {Pre_Surgery_Anxiety_Level}
        WOMAC Pain Subscale: {WOMAC_Pain_Subscale}
        VAS Pre-Surgery Pain Score: {VAS_Pre_Surgery_Pain_Score}
        Post-Surgery Pain Reduction Expectation: {Post_Surgery_Pain_Reduction_Expectation}
        KOOS Functionality Subscale: {KOOS_Functionality_Subscale}
        Time to Expected Mobility: {Time_to_Expected_Mobility}
        Lysholm Score Baseline: {Lysholm_Score_Baseline}
        IKDC-SKF Baseline Score: {IKDC_SKF_Baseline_Score}
        ACL-RSI Score Pre-Surgery: {ACL_RSI_Score_Pre_Surgery}
        MPFL-RSI Pre-Surgery Score: {MPFL_RSI_Pre_Surgery_Score}
        Support System Availability: {Support_System_Availability}
        Rehabilitation Commitment: {Rehabilitation_Commitment}

        User response (self-rated): {question_answer_string}

        JUST Return a json like below:
            {{ 
                "score": 1 to 5
            }}
        """

    # The user message
    user_message = """
        How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5.
        """

    # The assistant message (the model will learn to produce this)
    assistant_message = f"""
        {{ 
            "score": {question_answer_string}
        }}
        """

    # Merge into a single training prompt
    # Using simple tags <system>, <user>, <assistant> to demarcate roles:
    chat_prompt = f"<system>{system_message}</system>\n<user>{user_message}</user>\n<assistant>{assistant_message}</assistant>"

    sample["formatted_text"] = chat_prompt
    return sample


dataset = dataset.map(format_instruction, cache_file_name=None)
print(dataset[0]["formatted_text"])

model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    cache_dir=cache_dir,
    device_map="auto",
    token=os.getenv("HF_TOKEN", ""),
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    token=os.getenv("HF_TOKEN", ""),
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj", 
        "up_proj", 
        "down_proj",
    ]
)

model = prepare_model_for_kbit_training(model)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    result = scorer.score(decoded_preds, decoded_labels)
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

args = TrainingArguments(
    output_dir="/netscratch/muhkhan/results/phi_3_ft",
    num_train_epochs=10,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    tf32=False,
    max_grad_norm=0.3,
    warmup_steps=5,
    lr_scheduler_type="linear",
    disable_tqdm=False,
    auto_find_batch_size=True,
    report_to="none",
    ddp_find_unused_parameters=False,
)

model = get_peft_model(model, peft_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="formatted_text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=lambda x: x["formatted_text"],
    args=args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()

# save model
trainer.save_model("phi_3_ft_10_model")
tokenizer.save_pretrained("phi_3_ft_10_model")
trainer.model.push_to_hub("phi_3_ft_10_model", token=os.getenv("HF_TOKEN", ""))
tokenizer.push_to_hub("phi_3_ft_10_model", token=os.getenv("HF_TOKEN", ""))
