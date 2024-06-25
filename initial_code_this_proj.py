from langchain.llms import GooglePalm
api_key="AIzaSyAlg3FK3ItrQ9qtiKsxxJeZv2VbyZUBeLw"
llm=GooglePalm(google_api_key=api_key,temperature=0.7)#creativity_level
#poem=llm("write a 15 line poem on Cricket")
#print(poem)
from langchain.document_loaders.csv_loader import CSVLoader
loader=CSVLoader(file_path=r'C:\Users\Navin Kr Gupta\Downloads\codebasics_faqs.csv')
data=loader.load()
#print(data)
#above will create Document object having promt as the answers and source as the questions
#now that the data is loaded, we will create embedding
#there are n number of embeddings. we will be using instruct embedding of googlePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
instructor_embeddings = HuggingFaceInstructEmbeddings()
# e=embeddings.embed_query("What is your refund policy")
# print(len(e)) --768
# print(e[:5]) --[-0.04669319838285446, 0.009528379887342453, -0.003355831140652299, 0.023532895371317863, 0.03376814350485802] it is a list of vectors, if two list of vectors have a cosine similarity close to 1, they represent same sort of info
# print(type(e)) --it is a list
from langchain.vectorstores import FAISS
vectordb=FAISS.from_documents(documents=data,embedding=instructor_embeddings)
retriever=vectordb.as_retriever()
rdocs=retriever.get_relevant_documents("is there any discount for the courses available?")
#print(rdocs)  #these are relevant docs
from langchain.prompts import PromptTemplate

prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)
ques_ans=chain("Do you guys provide internship and also do you offer EMI payments?")
print(ques_ans)# it generates answer based on the csv loaded
