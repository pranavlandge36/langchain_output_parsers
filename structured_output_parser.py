from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import StructuredOutputParser , ResponseSchema
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)


schema=[
    ResponseSchema(name='Name',description='name of person '),
    ResponseSchema(name='Title',description='title of person'),
    ResponseSchema(name='About',description='Description of person'),
    ResponseSchema(name='Powers',description='describe in short powers of person'),
    ResponseSchema(name='Weapons',description='Weapons that the person uses')
]

parser=StructuredOutputParser.from_response_schemas(schema)

template1=PromptTemplate(
    template="""Give information about ONLY ONE person from Valhalla.

    Do NOT return multiple people.
    Do NOT add extra keys.
    {format_instructions}"""
    
   
    ,
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
# WITHOUT USING CHAINS
# prompt=template1.format(topic='one person from  valhalla')
# print(prompt)

# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)

# USING CHAINS


chain= template1 | model | parser
result= chain.invoke({'topic':'1 character from valhalla'})

print(result)