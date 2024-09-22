#
# WANDB hackathon 9/22/24
# Evaluation of the open-ended questions framework
# 
# Hypothesis 1.
# Does iterative application of the same eval prompt converge?
#
# Hypothesis 2.
# Does generative A-B comparison converge?


import os


WANDB_API_KEY=os.environ["WANDB_API_KEY"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

import weave
from openai import OpenAI

from pydantic import BaseModel



base_prompt = """
Consider the following dialog between the user and the chatbot.
The bot's goal is to suggest a cheaper mobile plan based on the information the user provides.
The user's responses are not guaranteed to be consistent or coherent at all times.

Your goal is to generate a list of action items to create better user experience. These action items will be used to generate synthetic data to train the bot.
If there are no suggestions to generate, reply with "success"=True and return an empty list.

Example.

[dialog]
bot: Currently you pay $30 for your plan. 
bot: Correct?
user: correct.
bot: Sorry, I did not get this.
user: yes.
[/dialog]

Recommendation list:
["make sure the bot understands nuanced and indirect answers to yes/no questions"]

"""

iterative_prompt = """
Consider the following dialog between the user and the chatbot.
The bot's goal is to suggest a cheaper mobile plan based on the information the user provides.
The user's responses are not guaranteed to be consistent or coherent at all times.

This dialog was evaluated by an LLM and this evaluation is given below. 

You job is to assess the quality of this evaluation and respond with "success"=True and repeat the original action list if there is nothing significant to add.
If there is something missing in evaluation, respond with "success"=False and a new list of action items to create better user experience integrating the old list with new suggestions. Make sure the list items are unique and not repetitive.

"""


generative_prompt = """
Consider the following sample dialog between the user and the chatbot.
The bot's goal is to suggest a cheaper mobile plan based on the information the user provides.
The user's responses are not guaranteed to be consistent or coherent at all times.

Also consider the given list of problems in the bot behavior with respect to providing a good service. 
You job is to generate a synthetic dialog between the user and chatbot that would display all problems from the list.
Use sample dialog as a model and modify it as needed. Stop generation when all mentioned problems have been shown.
"""

contrastive_prompt = """
consider two sample dialogs between the user and chatbot. 
The bot's goal is to suggest a cheaper mobile plan based on the information the user provides.
The user's responses are not guaranteed to be consistent or coherent at all times.

Your job is to access the quality of the chatbot user service. Respond with "success"=True and empty action items list id the second dialog has issues similar to the first one. Otherwise, respond with "success"=True and list action items to improve the chatbot from dialog two, but only for problems not present in the first dialog.
"""

dialog = """
"bot": Hey there!
"bot": I'm a chatbot trained to help you find the best mobile phone plan for you. What can I do for you?
"user": hey
"bot": Hey there! I can find a better plan for you. Let me know when you're ready!
"user": do i still have the cheapest plan
"bot": Okay, let's start! First, I need some information about your current mobile phone plan:
"bot": Who's your current mobile phone provider?
"user": supermobile
"bot": Okay. Which mobile phone plan do you have?
"user": yellow basic 1000
"bot": How much did you pay in total last month?
"user": $32,29
"bot": Got it. How many minutes have you spent on the phone last month (national calls)?
"user": 652 minutes
"bot": How many text messages did you send last month?
"user": 23
"bot": How much data did you use last month?
"user": 1450
"bot": Do you often travel outside of Europe?
"user": yes
"bot": Okay, final question: How much are you willing to spend on a mobile phone plan per month?
"user": $10.00
"bot": Thanks, here are your information again:
"bot": Currently, you pay $32.29 for a supermobile yellow basic 1000 plan. Last month, you spent 652 minutes on the phone, sent 23 messages, and used 1.45 GB of data. You often travel outside of Europe. In total, you don't want to pay more than $10 
"bot": Is that correct?
"user": yes
"bot": Alright, I'm looking for a better plan for you now. Stay tuned!
"bot": So, here is the best plan for you: Ultrafone offers a plan called Orange L for $14 per month. It comes with unlimited national calls, but without inclusive international minutes and free text messages. You would also get 2 GB of high-speed data!
"bot": Sounds good? I hope I could help you find a better mobile phone plan. If you want to try again, you can ask me anytime. Have a nice day!
"user": cool
"bot": Hi! How can I help you?
"user": can i book an additional package for international calls
"bot": I can respond to messages like  'new mobile phone plan'
"user": byebye
"bot": Sorry, I didn't understand that. Try asking me things like 'new plan'
"""

class EvalResponse(BaseModel):
    success: str
    action_items: list[str] 


class EvalPrompt(BaseModel):
    success: str
    improved_prompt: str




client = OpenAI(api_key=OPENAI_API_KEY)




@weave.op()
def extract_actions(message: str, prompt: str) -> EvalResponse:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": message
            }
            ],
            response_format=EvalResponse
        )
    return response.choices[0].message.parsed


@weave.op()
def contrastive_eval(message: str, prompt: str) -> EvalPrompt:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": message
            }
            ],
            response_format=EvalPrompt
        )
    return response.choices[0].message.parsed



weave.init('wdb-hackathon')
result = extract_actions(message = f"[dialog]{dialog}[/dialog]", prompt = base_prompt)
print(result)


def incremental_eval(dialog: str, result: EvalResponse) -> EvalResponse: 
   improved_prompt = iterative_prompt
   counter = 0
   while result.success != "True":
      compound_message = f"""
      [dialog]
      {dialog}
      [/dialog]
   
      [prompt]
      {improved_prompt}
      [/prompt]

      [eval]
      {str(result.action_items)}
      [\eval]

      """   
      result = extract_actions(message = compound_message, prompt = improved_prompt)
      counter = counter + 1
   return counter, result
   

folder_path = '/Users/dkh/wb-hackathon/data'

# self-improving eval cycle

iterations=[]
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
            counter, result = incremental_eval(dialog = file_content, result=result)
            iterations.append(counter)
            print(f"Final result for {filename}: {result} in {counter} iterations")
            result.success="False"

print(f"Iterations per dialog: {iterations}")
print(f"Final recommendations: {result.action_items}")
