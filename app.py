import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import tempfile
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from g4f.client import Client
from langchain.embeddings.base import Embeddings
from typing import List
from ydata_profiling import ProfileReport
from bs4 import BeautifulSoup

# Custom embedding class to match the expected interface
class CustomEmbedding(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0]

# Load Sentence-Transformers model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
custom_embedding = CustomEmbedding(embedding_model)

def read_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")
    return text    

def main():
    st.title("Data Analyzer Ai")

    tab1, tab2, tab3 = st.tabs(["Data Profile", "Excel Sheet", "Pair Plots"])

    # File uploader
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xls", "xlsx", "csv"])

    if uploaded_file is None:
        st.info("Please upload a file of type: " + ", ".join(["xls", "xlsx", "csv"]) + " to start analysing your data.")
        st.image("waiting.jpg", use_column_width=True)
        return

    elif uploaded_file is not None:
        st.sidebar.caption("Uploaded file: " + uploaded_file.name)
        
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

            # Extract text from ProfileReport HTML
            with open(st.session_state.profile_report_path, 'r') as f:
                html_string = f.read()
                profile_text = read_html(html_string)

            # Generate embeddings for the profile report text
            profile_text_list = [profile_text]
            profile_embeddings = custom_embedding.embed_documents(profile_text_list)

            # Store embeddings in FAISS
            vectorstore = FAISS.from_texts(texts=[profile_text], embedding=custom_embedding)
            st.session_state.vectorstore = vectorstore
        else:
            # Load the existing profile report HTML content
            with open(st.session_state.profile_report_path, 'r') as f:
                html_string = f.read()

        # Initialize G4F chatbot client if not already initialized
        if "client" not in st.session_state:
            try:
                st.session_state.client = Client()
            except Exception as e:
                st.error(f"Error initializing the AI chatbot: {e}")
                st.session_state.client = None

        with tab1:
            st.title("Data Profile")
            if 'html_string' in locals():
                styled_html = f'<div style="border: 2px solid orange; padding: 10px;">{html_string}</div>'
                st.components.v1.html(styled_html, height=800, scrolling=True)
            else:
                st.warning("Data Profile is not available.")

        with tab2:
            st.title("Excel")
            st.dataframe(df)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask AI anything regarding your data?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.client:
                try:
                    # Query FAISS vectorstore for the relevant data based on user prompt
                    retriever = st.session_state.vectorstore.as_retriever()
                    docs = retriever.get_relevant_documents(prompt)

                    # Generate a context from the retrieved documents
                    context = "\n".join([doc.page_content for doc in docs])
                    full_prompt = f"Context: {context}\n\nUser Prompt: {prompt}\n\nPlease respond in English with specific insights or answers based on the dataset provided."

                    response = st.session_state.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": full_prompt}
                        ]
                    )
                    full_response = response.choices[0].message.content

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown(full_response + "|")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error during chat interaction: {e}")
            else:
                st.error("G4F chatbot is not available due to initialization error.")

        html_button_sidebar = st.sidebar.button("Download Report")
        if html_button_sidebar:
            progress_bar = st.sidebar.progress(0)
            progress_message = st.sidebar.empty()
            
            progress_message.text("Preparing profile report for download...")
            progress_bar.progress(10)

            # Use the previously generated ProfileReport for download
            with open(st.session_state.profile_report_path, "rb") as f:
                report_data = f.read()

            progress_bar.progress(30)
            encoded_report = base64.b64encode(report_data).decode()
            download_link = f'<a href="data:text/html;base64,{encoded_report}" download="profile_report.html">Click here to download the profile report</a>'
            
            progress_bar.progress(60)
            st.sidebar.markdown(download_link, unsafe_allow_html=True)
            progress_bar.progress(80)
            st.sidebar.info("☝︎ Click the link to download the profile report, and open it in your browser.\n\n You can press Ctrl+P to save the report as a PDF.")
            progress_bar.progress(100)
            progress_message.text("Download link generated successfully!")

        numeric_columns = df.select_dtypes(include=['number'])
        non_blank_numeric_columns = numeric_columns.dropna(axis=1)
        if not non_blank_numeric_columns.empty:
            with tab3:
                st.title("Pair Plots")
                with st.spinner('Generating Pair Plot...'):
                    try:
                        fig = sns.pairplot(non_blank_numeric_columns)
                        st.pyplot(fig)
                        st.markdown("Pair Plot Generated on these columns: \n\n")
                        st.write(non_blank_numeric_columns)
                    except Exception as e:
                        st.error(f"An error occurred while generating the pair plot: {e}")
                        st.stop()
        else:
            with tab3:
                st.warning("No numeric value column found for pair plot generation.")

if __name__ == "__main__":
    main()

