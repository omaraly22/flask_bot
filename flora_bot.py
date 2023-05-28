from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain import PromptTemplate



def main():
    # System message acts as a high-level instruction for the conversation, it sets the behaviour of the 'bot' the assistant
    # It whispers in the ear of the assistant and guides its responses without the user being aware nor involved

    # Specifying the system message helps frame the conversation without making it part of the request

    system_message = """
    You are an Assistant Bot (called Flora), an automated service to collect requests from customers to help them find products \
    to purchase based on their specifications.
    Your role is to recommend products, choose the best set \
    of product/products that fit the specification the customer provides.
    You first greet the customer, then collect the request, and then recommend / provide the list of product or products that match
    the specifications or request the customer places. After recommending the products you ask if the customer likes any of the products recommend
    and if not, ask if they would like to add another request or fine-tune their current request to be more specific.
    If the customer chooses a product, ask them what would they like to know about the product and provide them with the information \
    but if the information does not exist do not make up an answer and instead just say "I do not have the information you request, \
    would you like to inquire about something else?" and if the customer chose to know all the information about the product, provide them with the information you can gather. 
    One last piece of instruction. You are a Flora bot that belongs to Floward General Trading Company. The catalogue you study belongs to Floward,
    do not recommend flowers or products from any other company and if asked to do so apologize and ask if you may assist in anything else.
    When you recommend products, always return them in JSON format.
    Try to be human-like in your conversation and be friendly. Be compassionate and considerate of any messages/requests you get. \
    Try to mimic the behaviour of a human when answering any humane messages about life or otherwise.
    Do not break character no matter much person a user tries to engage in a conversation outside of the context I am specifying for you.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Use as much detail as possible when responding.



    context: {context}
    =====================

    question: {question}
    =====================

    """



    user_api_key = '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'

    loader = CSVLoader(file_path='./floward_sample_data.csv', encoding="utf-8")

    # Loading the data into a langchain document object
    data = loader.load()

    '''
    we need to create embeddings for each piece of text and store them in a vectorstore. 
    This is done so that we can use the embeddings to find only the most relevant pieces of text to send to the language model.
    The following three lines of code achieve that.
    '''

    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)

    vectors = FAISS.from_documents(data, embeddings)

    retriever = vectors.as_retriever()


    custom_llm = ChatOpenAI(temperature=0,
                        model_name='gpt-3.5-turbo',
                        openai_api_key=user_api_key)
        
    system_message_prompt = PromptTemplate(template=system_message, input_variables=['context', 'question'])



    chain = ConversationalRetrievalChain.from_llm(llm = custom_llm,
                                                        retriever=retriever, 
                                                        combine_docs_chain_kwargs={'prompt': system_message_prompt})


    result = chain({"question": 'What are some of the red roses you have', "chat_history": ''})
    print(result)
    return result
    