from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import re

import os
import streamlit as st

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

class ChatBot:

    def __init__(self, url, question):
        self.video_id = self.extract_video_id(url)
        self.user_question = question

    def extract_video_id(self, url: str) -> str:
        # Extracts video ID from different YouTube URL formats
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else:
            st.error("Invalid YouTube URL")
            return None

    def transcript_generator(self, vid_id):   
        try:
            yt_api = YouTubeTranscriptApi()
            transcript_list = yt_api.fetch(video_id= vid_id, languages = ['en', 'hi'])

            # flatten dict
            transcript = " ".join(chunk.text for chunk in transcript_list)

            #split transcript into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200
            )
            chunks = splitter.create_documents([transcript])

            return chunks

        except (TranscriptsDisabled, NoTranscriptFound):
            st.error(f"Error: No transcript available!")
            st.stop() 

    def RAG(self, vector_store, user_question):
        retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {"k":5})

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

        prompt = PromptTemplate(
            template= """
            You are a helpful assistant.
            Answer appropriately ONLY from the provided transcript context.
            If you dont know, answer "Sorry, I don't know about this topic".\n

            Context:
            {context}
            
            Question : {question}""",
            
            input_variables=['context', 'question']
        )
       
        def format_docs(retrieved_docs):
            # join page contents of retrieved doc
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        result = main_chain.invoke(user_question)
        return result
    
    def main(self):
        

        # Session state init
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        if "chunks" not in st.session_state:
            st.session_state.chunks = None
        if "last_video_id" not in st.session_state:
            st.session_state.last_video_id = None


        # Build vector store only if new video is entered
        if self.video_id and self.video_id != st.session_state.last_video_id:
            chunks = self.transcript_generator(self.video_id)
            st.session_state.chunks = chunks
            st.session_state.vector_store = FAISS.from_documents(
                chunks,
                GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            )
            st.session_state.last_video_id = self.video_id  # track last one

        if st.session_state.vector_store:
            result = self.RAG(st.session_state.vector_store, self.user_question)
            st.write(result)
    
if __name__ == "__main__":
    st.title("YouTube RAG QA")
    url = st.text_input("Enter youtube video url:")
    question = st.text_input("Ask a question:")
    
    if st.button("Submit"):
        app = ChatBot(url, question)
        app.main()