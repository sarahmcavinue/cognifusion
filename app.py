import os
import time
import numpy as np
import streamlit as st
from clean_data import CleanData
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


#############################   Templates     ##################################

# Given contexts for different influencer types
contexts = {
    "fitness_influencer": "Strategies for enhancing audience engagement through innovative fitness content, leveraging the latest fitness tracking technologies, and health trends.",
    "tech_influencer": "Insights into the latest tech gadgets, software tool enhancements, and effective methods for tech content creation and distribution.",
    "mom_blogger": "Tips on creating engaging content for parents, managing time effectively while producing family-friendly content, and strategies for dealing with dietary restrictions in children.",
    "travel_blogger": "Best practices for travel blogging, engaging with tourism boards, and using social media to enhance travel content visibility.",
    "skincare_blogger": "Advanced techniques in skincare blogging, addressing audience-specific skin concerns, and the importance of ethical brand partnerships."
}
user_question = "Provide me with a detailed 500-word report on recommendations on how I can increase my engagement?"

# Determine the context based on the question
if "skincare" in user_question and "ethical" in user_question:
    selected_context = contexts["skincare_blogger"]
elif "fitness" in user_question:
    selected_context = contexts["fitness_influencer"]
elif "tech" in user_question:
    selected_context = contexts["tech_influencer"]
elif "mom" in user_question:
    selected_context = contexts["mom_blogger"]
elif "travel" in user_question:
    selected_context = contexts["travel_blogger"]
else:
    selected_context = "Please specify the area of influence for more tailored advice."


# Defining the template with placeholders for context and question.
prompt_template = """
Human: As an AI with knowledge of social media influencers, make product suggestions and offer advice for the social media influencer in the Customer Discovery Interviews PDFs. Here are recommendations for the specified name and username.
Context: {context}
Current conversation:
Question: {question}
Assistant:
"""

# Ensure 'selected_context' is defined based on user input, possibly adjusted with a default or error handling
if not selected_context:
    selected_context = "General social media influencing strategies."

# The user question should be the one initially asked if it's meant to be responded to
user_question = "Provide me with a detailed 500-word report on recommendations on how I can increase my engagement?"  # The real user question

# Using str.format() to insert the context and question into the template
full_prompt = prompt_template.format(context=selected_context, question=user_question)



_template = """
Human: Given the following conversation and a follow-up question, rephrase the follow-up question into a standalone question without changing the content of the given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: Assistant:"
"""

_template = """
Human: Given the following conversation and a follow-up question, rephrase the follow-up question into a standalone question without changing the content of the given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: Assistant:"
"""




############################### Setting the AI model #########################################
API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b" #LLM model
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY")

#streamlit UI for model selection
company_logo = 'https://www.app.nl/wp-content/uploads/2019/01/Blendle.png'
st.set_page_config(page_title="CogniFusion Project", page_icon=company_logo)

embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002') #Ada OpenAI model for text embedding

# define the repository ID for the Gemma 2b model
repo_id = "google/gemma-7b"
huggingfacehub_api_token  = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

llm_model = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 800,
        "top_k": 5,
        "temperature": 0.7,
        "repetition_penalty": 1.03,
    },
    huggingfacehub_api_token = huggingfacehub_api_token,
)
########################### Process PDF Docs ##########################################

def process_pdf_documents(urls, download_directory, embedding_model):
    os.makedirs(download_directory, exist_ok=True)
    clean_data_instance = CleanData(data_dir=download_directory, urls=urls)
    clean_data_instance.download_files()
    # Load the documents and split them into smaller chunks
    loader = PyPDFDirectoryLoader(f"./{download_directory}/")
    documents = loader.load()
    # Character split setup
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Check documents are loaded and split correctly
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(docs)
    print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
    print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
    print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

    # Process embedding
    try:
        sample_embedding = np.array(embedding_model.embed_query(docs[0].page_content))
        print("Sample embedding of a document chunk: ", sample_embedding)
        print("Size of the embedding: ", sample_embedding.shape)

    except ValueError as error:
        if "AccessDeniedException" in str(error):
            print("\x1b[41mError: Access to embedding model is denied.\
              \nPlease check your access permissions or ensure that the embedding model is properly configured.\
              \nFor troubleshooting, refer to the documentation or contact your administrator.\x1b[0m\n")
        else:
            print("\x1b[41mError: Failed to process embedding.\
              \nPlease check your input data or ensure that the embedding model is properly configured.\
              \nFor troubleshooting, refer to the documentation or contact support.\x1b[0m\n")
    return docs

# List of URLs to download PDFs from
urls = [
    "data/influencer_a.pdf",  # influencer_a
    "data/influencer_b.pdf",  # influencer_b
    "data/influencer_c.pdf",  # influencer_c
    "data/influencer_d.pdf",  # influencer_d
    "data/influencer_e.pdf",  # influencer_e
]

download_directory = "test"
process_pdf_documents(urls, download_directory, embedding_model)
docs_to_be_processed = process_pdf_documents(urls, download_directory, embedding_model)



######################################   Conversation Chain           ###########################################
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)



def load_chain(llm_model, embedding_model):
    chain = ConversationalRetrievalChain.from_llm(
        llm= llm_model, 
        retriever=FAISS.from_documents(docs_to_be_processed, embedding_model).as_retriever(search_kwargs={'k': 4}),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=True,
        chain_type='stuff',
        condense_question_prompt= CONDENSE_QUESTION_PROMPT)
    
    chain.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template(prompt_template)
    
    return chain

#Sreamlit UI

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Define title and subheader
st.title("👩‍💻 Influencer Helper")
st.subheader(f"Powered by LLM: Google Gemma 🦜🔗")

 # Load the chain
chain = load_chain(llm_model, embedding_model)

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi human! I am MTU's smart AI. How can I help you today?"}]
# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = chain({"question": query})
        response = result['answer']
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

#runwith: python3 -m streamlit run app3.py 

#Test the app with the following questions:
#Use the Customer Discovery Interview for @adams_fitness_fanatic provide her with a detailed 500-word report on recommendations on how to increase engagement with her followers and suggest brands she could potentially work with.
#Provide a report for @adams_fitness_fanatic
#@adams_tech_talk
#@mommybitesandmore
#@wanderlustwarrior
#@glowbynaomi





