from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)
parser=JsonOutputParser()

template1=PromptTemplate(
    template='give information about {topic} \n {format_instructions} ',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
# WITHOUT USING CHAINS
# prompt=template1.format(topic='people of valhalla')
# print(prompt)

# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)

# USING CHAINS


chain= template1 | model | parser
result= chain.invoke({'topic':'3 people of valhalla'})

print(result)