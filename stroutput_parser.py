from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template='write a detailed report on topic {topic} ',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='write a 5 line summary on folling text {text}',
    input_variables=["text"]
)

# prompt1=template1.invoke({'topic':'valhalla'})

# result=model.invoke(prompt1)
# print(result.content)
# prompt2=template2.invoke({'text':result.content})

# response=model.invoke(prompt2)
# print(response.content)


parser=StrOutputParser()

chain= template1| model | parser | template2 | model | parser
result = chain.invoke({'topic':'valhalla'})
print(result)