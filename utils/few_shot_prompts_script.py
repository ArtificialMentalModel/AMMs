



def prompts():
    prompt1 = f"""
            Based on the Big 5 personality traits, here are the personalized scores of a user. I have also attached the general norm and their definitions. I will also give you the user's age, gender, profession, current emotional state, if they have physical disabilities or not, what type of physical activities they participate in, how much they exercise, their overall health status, their mobility status, etc.  Your task is to give me a rating to a question that this user would rate themselves based on these below traits. Remember, the idea is to predict what the user would do and see the Artificial Mental Model part of the person. \n\n
         
            I will give you user response to the question too. 

            I will give you some data in German. Give whole response in English.

            Remember, the user is always right.
         
            Just tell if the user will be able to perform the exercise (Yes or No) with a score you rate that user (1 being not difficult at all and 5 being extremely difficult). Also see what the user rated themselves.
         
            See the Big 5 scores and see the Artificial Mental Model of the user below. 
            Here is everything you need to know:
            Do not assume anything about the user. Only consider what is given to you below.\n\n
            •       Extroversion: 3/7 (High) (General Norm: 4.44)
            •       Agreeableness: 4.5/7 (Medium Low) (General Norm: 5.23)
            •       Conscientiousness: 5/7 (Medium High) (General Norm: 5.4)
            •       Emotional Stability: 3/7 (Medium High) (General Norm: 4.83)
            •       Openness: 2.5/7 (High) (General Norm: 5.38)
            Here’s a brief overview of what each of the Big Five personality traits represents:
            •       Extroversion: Reflects your level of sociability and enthusiasm.
            •       Agreeableness: Indicates how cooperative and kind-hearted you are.
            •       Conscientiousness: Measures your reliability and attention to detail.
            •       Emotional Stability: Assesses your ability to remain stable and composed.
            •       Openness: Shows your willingness to embrace new experiences and ideas.\n\n

            Age: 18-24\n
            Gender: Female\n
            Employment status: Student\n
            Current emotional state: Relaxed\n
            Have any physical disabilities: No\n
            Type of physical activities: Other\n
            How many days do you do exercise: 1-2 days\n
            Overall health status: 3 - Good\n
            Current mobility: 3 - Good\n\n
         

            User Response to the question: 1 - not difficult at all
         
            
            {{ 
                "able_to_perform": YES/NO
            }}
            """
    return prompt1