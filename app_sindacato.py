"""
Assistente Sindacale Scuola - Versione con caricamento documenti automatico
Installa: pip install streamlit groq chromadb sentence-transformers PyPDF2 python-docx
"""

import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
import os
from datetime import datetime
import io

st.set_page_config(
    page_title="🎓 Assistente Sindacale Scuola",
    page_icon="🎓",
    layout="wide"
)

class DocumentProcessor:
    """Gestisce il caricamento di PDF, DOCX, TXT"""
    
    @staticmethod
    def extract_from_pdf(file):
        """Estrae testo da PDF"""
        try:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Errore lettura PDF: {e}")
            return None
    
    @staticmethod
    def extract_from_docx(file):
        """Estrae testo da DOCX"""
        try:
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            st.error(f"Errore lettura DOCX: {e}")
            return None
    
    @staticmethod
    def extract_from_txt(file):
        """Estrae testo da TXT"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Errore lettura TXT: {e}")
            return None
    
    @staticmethod
    def split_into_chunks(text, chunk_size=1000, overlap=200):
        """Divide il testo in chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Cerca di terminare a fine frase
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks

class SchoolAssistant:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        
        if 'embedding_model' not in st.session_state:
            with st.spinner('⚙️ Caricamento modello AI...'):
                st.session_state.embedding_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
                )
        self.embedding_model = st.session_state.embedding_model
        
        if 'school_chroma' not in st.session_state:
            st.session_state.school_chroma = chromadb.Client()
            try:
                st.session_state.school_collection = st.session_state.school_chroma.get_collection("school_docs")
            except:
                st.session_state.school_collection = st.session_state.school_chroma.create_collection(
                    name="school_docs",
                    metadata={"hnsw:space": "cosine"}
                )
        
        self.collection = st.session_state.school_collection
    
    def add_document(self, text: str, filename: str, categoria: str):
        """Aggiunge un documento al database"""
        chunks = DocumentProcessor.split_into_chunks(text)
        
        if not chunks:
            return 0
        
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        existing = self.collection.count()
        ids = [f"doc_{existing + i}" for i in range(len(chunks))]
        
        metadata = [{
            "filename": filename,
            "categoria": categoria,
            "chunk": i + 1,
            "total_chunks": len(chunks),
            "data": datetime.now().isoformat()
        } for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids,
            metadatas=metadata
        )
        
        return len(chunks)
    
    def search(self, query: str, n_results: int = 4):
        """Cerca documenti rilevanti"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def answer_question(self, question: str, model: str = "llama-3.3-70b-versatile"):
        """Risponde usando RAG"""
        results = self.search(question)
        
        if not results['documents'][0]:
            context = "Nessun documento trovato."
            sources = []
        else:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            context_parts = []
            sources = []
            
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                context_parts.append(f"[Fonte {i+1} - {meta.get('filename', 'N/A')}, {meta.get('categoria', 'N/A')}]\n{doc}")
                sources.append({
                    "filename": meta.get('filename', 'N/A'),
                    "categoria": meta.get('categoria', 'N/A')
                })
            
            context = "\n\n".join(context_parts)
        
        prompt = f"""Sei un esperto consulente sindacale per il personale della scuola italiana.

CONTESTO (Documenti disponibili):
{context}

DOMANDA: {question}

ISTRUZIONI:
- Rispondi in modo chiaro e professionale
- Cita sempre le fonti quando usi informazioni dal contesto
- Se il contesto non basta, usa la tua conoscenza delle normative scolastiche
- Fornisci informazioni pratiche e operative
- Se necessario, suggerisci di rivolgersi al sindacato territoriale

RISPOSTA:"""

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Sei un esperto del settore scuola: CCNL, graduatorie, diritti e doveri del personale."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=2048
        )
        
        return chat_completion.choices[0].message.content, sources


def main():
    st.title("🎓 Assistente Sindacale Scuola")
    st.markdown("*Con caricamento documenti PDF/DOCX*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        api_key = st.text_input(
            "🔑 API Key Groq",
            type="password",
            help="Ottieni su https://console.groq.com"
        )
        
        if not api_key:
            st.warning("⚠️ Inserisci API Key")
            st.info("👉 https://console.groq.com")
            st.stop()
        
        st.success("✅ Sistema attivo")
        
        model = st.selectbox(
            "🤖 Modello",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        )
        
        st.divider()
        
        try:
            assistant = SchoolAssistant(api_key)
            doc_count = assistant.collection.count()
            st.metric("📄 Documenti", doc_count)
        except Exception as e:
            st.error(f"Errore: {e}")
            st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Consulenza", "📤 Carica Documenti", "📖 Database"])
    
    # TAB 1: Chat
    with tab1:
        st.header("💬 Chiedi all'assistente")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📅 Ferie"):
                st.session_state.quick = "Quanti giorni di ferie ho come docente?"
        with col2:
            if st.button("💰 Stipendio"):
                st.session_state.quick = "Come funzionano gli scatti?"
        with col3:
            if st.button("📋 Supplenze"):
                st.session_state.quick = "Differenza tra 31/08 e 30/06?"
        
        st.divider()
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Fonti"):
                        for s in msg["sources"]:
                            st.write(f"• {s['filename']} ({s['categoria']})")
        
        default = st.session_state.get('quick', '')
        if default:
            prompt = default
            st.session_state.quick = None
        else:
            prompt = st.chat_input("Scrivi la tua domanda...")
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Ricerca..."):
                    try:
                        response, sources = assistant.answer_question(prompt, model)
                        st.markdown(response)
                        
                        if sources:
                            with st.expander("📚 Fonti"):
                                for s in sources:
                                    st.write(f"• {s['filename']} ({s['categoria']})")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Errore: {e}")
        
        if st.button("🗑️ Nuova chat"):
            st.session_state.messages = []
            st.rerun()
    
    # TAB 2: Upload documenti
    with tab2:
        st.header("📤 Carica Documenti")
        
        st.info("💡 Carica CCNL, circolari, contratti integrativi in PDF/DOCX/TXT")
        
        uploaded_files = st.file_uploader(
            "Scegli i file",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        categoria = st.selectbox(
            "📁 Categoria",
            ["CCNL Scuola", "Circolari MIUR", "Contratto Integrativo", 
             "Normativa", "Delibere", "Altro"]
        )
        
        if st.button("📥 Carica e Elabora", type="primary"):
            if not uploaded_files:
                st.warning("⚠️ Seleziona almeno un file")
            else:
                progress = st.progress(0)
                total_chunks = 0
                
                for i, file in enumerate(uploaded_files):
                    st.write(f"📄 Elaborazione: {file.name}")
                    
                    # Estrai testo
                    if file.name.endswith('.pdf'):
                        text = DocumentProcessor.extract_from_pdf(file)
                    elif file.name.endswith('.docx'):
                        text = DocumentProcessor.extract_from_docx(file)
                    elif file.name.endswith('.txt'):
                        text = DocumentProcessor.extract_from_txt(file)
                    else:
                        st.warning(f"Formato non supportato: {file.name}")
                        continue
                    
                    if text:
                        chunks = assistant.add_document(text, file.name, categoria)
                        total_chunks += chunks
                        st.success(f"  ✅ Caricato in {chunks} chunks")
                    
                    progress.progress((i + 1) / len(uploaded_files))
                
                st.success(f"🎉 Completato! {total_chunks} chunks aggiunti al database")
                st.balloons()
    
    # TAB 3: Esplora
    with tab3:
        st.header("📖 Esplora Database")
        
        search_query = st.text_input("🔍 Cerca", placeholder="es. ferie, stipendio...")
        
        if search_query:
            results = assistant.search(search_query, n_results=5)
            
            if results['documents'][0]:
                st.subheader(f"Trovati {len(results['documents'][0])} risultati")
                
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    with st.expander(f"📄 {meta.get('filename', 'N/A')} - Chunk {meta.get('chunk', '?')}/{meta.get('total_chunks', '?')}"):
                        st.markdown(doc)
                        st.caption(f"Categoria: {meta.get('categoria', 'N/A')}")
            else:
                st.info("Nessun risultato trovato")

if __name__ == "__main__":
    main()