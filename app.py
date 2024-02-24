import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import tempfile
import base64
import pdfkit
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def generate_pdf(html_content, output_path):
    config = pdfkit.configuration(wkhtmltopdf='wkhtmltopdf')
    pdfkit.from_string(html_content, output_path, configuration=config)


# Function to read HTML content
def read_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # Extract text from HTML
    text = soup.get_text(separator=" ")
    return text    


# Load environment variables
load_dotenv()


def main():
    st.title("Data Analyzer Ai")

    # File uploader
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xls", "xlsx", "csv"])

    if uploaded_file is None:
        st.info("Please upload a file of type: " + ", ".join(["xls", "xlsx", "csv"]) + " to start analysing your data.")
        st.image("waiting.jpg", use_column_width=True)
        return

    if uploaded_file is not None:
        st.sidebar.caption("Uploaded file: " + uploaded_file.name)
        

        # Read uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return

        # Generate ProfileReport if not already generated
        if "profile_report_path" not in st.session_state:
            with st.spinner('Generating Profile Report...'):
                profile = ProfileReport(df, correlations={"auto": {"calculate": False}}, missing_diagrams={"Heatmap": False})
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                    profile.to_file(output_file=temp_file.name)
                    st.session_state.profile_report_path = temp_file.name
               

        # Get HTML content from the generated profile report
        with open(st.session_state.profile_report_path, 'r') as f:
            html_string = f.read()
            text = read_html(html_string)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )   

        # Process the text and create the documents list
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        # Load the Langchain chatbot
        llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())


        # Add an outline around the HTML content
        styled_html = f'<div style="border: 2px solid orange; padding: 10px;">{html_string}</div>'

        # Display the HTML content with outline
        st.components.v1.html(styled_html, height=800, scrolling=True)
        
        
        
        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Ask ai anything regarding your data?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})  # Adjust height as needed

        # Download button for PDF
        pdf_button_sidebar = st.sidebar.button("Download PDF Report")

        if pdf_button_sidebar:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
                generate_pdf(html_string, temp_pdf_file.name)
                with open(temp_pdf_file.name, 'rb') as f:
                    pdf_data = f.read()
                b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="report.pdf">Click here to download</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)
                


if __name__ == "__main__":
    main()
