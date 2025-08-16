def get_prompt_templates():
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
                "score": 1 to 5
            }}
            """

    prompt2 = f"""
            ### Instruction:

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
                "score": 1 to 5
            }}

            ### User Query:

            How difficult does it look to perform 10 squats? Please rate the difficulty on a scale from 1 to 5.

            ### Response:

            """
    prompt3 = f"""
        <system>
        You are a helpful assistant. 

        You will be given user data including the Big 5 personality traits and their demographic and health information. Based on this, the user is asked about the difficulty of performing an exercise and provides their own rating. Your job is to reflect the user's score based on their data.

        Remember:
        - The user is always correct.
        - The question: "How difficult does it look to perform 10 squats?"
        - Rate difficulty from 1 to 5.
        - You have the following user data:

        Extroversion: 3/7 (High) (General Norm: 4.44)
        Agreeableness: 4.5/7 (Medium Low) (General Norm: 5.23)
        Conscientiousness: 5/7 (Medium High) (General Norm: 5.4)
        Emotional Stability: 3/7 (Medium High) (General Norm: 4.83)
        Openness: 2.5/7 (High) (General Norm: 5.38)

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

        JUST Return a json like below:
        {{ 
            "score": 1 to 5
        }}
        </system>
        """
    prompt4 = f"""
        ### Instruction:
        You are an expert news analyst with advanced skills in understanding and interpreting news articles. 
        Your job is to carefully classify each article into one of 4 categories: '1-World', '2-Sports', '3-Business', '4-Sci/Tech'.
            
        JUST Return a json like below:
        {{ 
            "score": 1 to 4
        }}
        """
    return prompt4