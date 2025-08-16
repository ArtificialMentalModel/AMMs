import openai
import pandas as pd
from gpt4_few_shots_prompts import prompts
import json
import os

# Constants
input_file = 'StudyByModels/Discover Your Inner Self - English_4o_mini_fin copy.xlsx'
output_folder = 'results_zeroshot_fewshot_all_one_exercise'
os.makedirs(output_folder, exist_ok=True)

# Load data
tipi_df = pd.read_excel(input_file, sheet_name='TIPI')
data_df = pd.read_excel(input_file, sheet_name='Data')

# Add new columns
tipi_df['Model_Output_query_1'] = ""
tipi_df['Prompt_query_1'] = ""
tipi_df['Actual_Output_query_1'] = ""
tipi_df['Logits_query_1'] = ""

# Setup OpenAI client
# Add your OpenAI API key here
openai.api_key = os.getenv('OPENAI_API_KEY', '')


# Process only first row for now
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
    answer_ = data_row['Q32']

    print("******************")
    print(answer_)
    print("******************")

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
         

            User Response to the question: {answer_}

            JUST Return a json like below:
            {{ 
                "score": 1 to 5
            }}
    """

    completion = openai.ChatCompletion.create(
        response_format={"type": "json_object"},
        model="gpt-4o-mini",
        temperature=0.1,
        logprobs=True,
        top_logprobs=5,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5"}
        ]
    )

    logprobs = completion.choices[0].logprobs
    output_text = completion.choices[0].message.content
    print(output_text)

    tipi_df.at[index, 'Model_Output_query_1'] = output_text
    tipi_df.at[index, 'Prompt_query_1'] = prompt
    tipi_df.at[index, 'Actual_Output_query_1'] = answer_
    tipi_df.at[index, 'Logits_query_1'] = str(logprobs)

# Build output file name
model_name = "TIPI_gpt-4o-mini_with_mm"
lang = "german"
shot_type = "zeroshot"
output_filename = f"{model_name}_{shot_type}_{lang}.xlsx"
output_path = os.path.join(output_folder, output_filename)

# Save to new file
with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
    tipi_df.to_excel(writer, sheet_name='TIPI', index=False)

print(f"\n✅ Processing complete. File saved to: {output_path}")
