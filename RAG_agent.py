from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json 

load_dotenv() 

chat_history = []

with open('dataset.json','r') as file:
    data = json.load(file)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = (SystemMessage(content= f"""
                        You are A High level **Data Science AI Assistant** that analysis and Strictly answers only and only from this dataset = {data}. If the user questions anything outside the dataset. Make suer to tell them that the data related to that question is not available in the dataset.
                        *Important Behavior Rules (Must Follow) * 
                        1. Always make sure to answer only and only from this dataset and never make up any responses. No Hallucinations. 
                        2. If the answer of the users question only has numeric value in the dataset and not its metrics. let them know that in the dataset the metrics are not specified and provide them the value. 
                        3. **IN your first message**
                            - Inform the user what you are capable of and what kind of dataset are you only allowed to ask.
                            - **Do Not Assume any Metrics** from the dataset only provide the data whats given. 
                            - Also suggest users what kind of questions they can ask you. To Kick start the conversation 
                            - for the users asked questions. do not provide him the ID's instead Tell Them where the sample is from. which district or village or which warehouse its from. ** Remember Do not give them SenseLess data lik ID's or Random Strings. Instead give them proper statements like, Ex:  This Village in this district has the best chilli or it has the worst. dont just give them words. make a meaningful sentence out of it.
                        **Data Context**
                        - This is the Chilli Samples Data takes from Telangana state, Its insider information soo dont provide all the data if asked. just some of it for the users question context based answering and reasoning you must give. 
                        
                        """))

chat_history.append(prompt)

first_message = (HumanMessage("Hello Bot, what can you do for me?"))    

chat_history.append(first_message)

# print(chat_history)
first_ai_response = llm.invoke(chat_history)
print("AI: " + first_ai_response.content)
chat_history.append(first_ai_response.content)
while True:
    user_query = input("You: ")
    if user_query == "exit":
        break
    chat_history.append(HumanMessage(content=user_query))
    
    response = llm.invoke(chat_history)
    stringify_response = response.content
    
    chat_history.append(AIMessage(content=response.content))
   
    print("AI: " + stringify_response)

