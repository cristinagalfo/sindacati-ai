"""
Assistente Sindacale Scuola - Documenti precaricati da cartella
I documenti nella cartella 'documenti/' vengono caricati automaticamente all'avvio

Installa: pip install streamlit groq chromadb sentence-transformers PyPDF2 python-docx
Struttura cartelle:
  sindacati-ai/
  ├── app_scuola.py
  ├── documenti/
  │   ├── CCNL_Scuola_2016-2018.pdf
  │   ├── Circolare_Ferie.pdf
  │   └── ...altri documenti...
"""

import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from datetime import datetime
import hashlib

# Configurazione pagina
st.set_page_config(
    page_title="🎓 Assistente Sindacale Scuola",
    page_icon="🎓",
    layout="wide"
)

class DocumentLoader:
    """Carica documenti dalla cartella 'documenti/'"""
    
    def __init__(self, folder="documenti"):
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    @staticmethod
    def extract_text_from_pdf(filepath):
        """Estrae testo da PDF"""
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.warning(f"⚠️ Errore PDF {os.path.basename(filepath)}: {e}")
            return None
    
    @staticmethod
    def extract_text_from_docx(filepath):
        """Estrae testo da DOCX"""
        try:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            st.warning(f"⚠️ Errore DOCX {os.path.basename(filepath)}: {e}")
            return None
    
    @staticmethod
    def extract_text_from_txt(filepath):
        """Estrae testo da TXT"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            st.warning(f"⚠️ Errore TXT {os.path.basename(filepath)}: {e}")
            return None
    
    @staticmethod
    def split_into_chunks(text, chunk_size=1500, overlap=300):
        """Divide il testo in chunks sovrapposti"""
        if not text or len(text.strip()) < 100:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Cerca di terminare a fine frase
            if end < len(text):
                # Cerca punto, a capo o punto e virgola
                for sep in ['. ', '.\n', '; ', ';\n', '\n\n']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size * 0.6:
                        chunk = chunk[:last_sep + len(sep)]
                        end = start + last_sep + len(sep)
                        break
            
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def get_file_hash(self, filepath):
        """Calcola hash del file per tracking modifiche"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_all_documents(self):
        """Carica tutti i documenti dalla cartella"""
        if not os.path.exists(self.folder):
            return []
        
        documents = []
        files_found = []
        
        for filename in os.listdir(self.folder):
            filepath = os.path.join(self.folder, filename)
            
            if not os.path.isfile(filepath):
                continue
            
            # Filtra per estensione
            if not filename.lower().endswith(('.pdf', '.docx', '.txt')):
                continue
            
            files_found.append(filename)
            
            # Estrai testo in base al tipo
            if filename.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(filepath)
                doc_type = "PDF"
            elif filename.lower().endswith('.docx'):
                text = self.extract_text_from_docx(filepath)
                doc_type = "DOCX"
            elif filename.lower().endswith('.txt'):
                text = self.extract_text_from_txt(filepath)
                doc_type = "TXT"
            else:
                continue
            
            if not text or len(text.strip()) < 100:
                st.warning(f"⚠️ {filename}: documento vuoto o troppo corto")
                continue
            
            # Dividi in chunks
            chunks = self.split_into_chunks(text)
            
            if not chunks:
                continue
            
            # Determina categoria dal nome file
            categoria = self.detect_category(filename)
            file_hash = self.get_file_hash(filepath)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "tipo": doc_type,
                        "categoria": categoria,
                        "chunk_index": i + 1,
                        "total_chunks": len(chunks),
                        "file_hash": file_hash,
                        "data_caricamento": datetime.now().isoformat()
                    }
                })
        
        return documents, files_found
    
    @staticmethod
    def detect_category(filename):
        """Rileva categoria dal nome file"""
        filename_lower = filename.lower()
        
        if 'ccnl' in filename_lower:
            return "CCNL Scuola"
        elif 'circolare' in filename_lower or 'miur' in filename_lower:
            return "Circolari MIUR"
        elif 'contratto' in filename_lower or 'integrativo' in filename_lower:
            return "Contratto Integrativo"
        elif 'delibera' in filename_lower:
            return "Delibere"
        elif 'ferie' in filename_lower or 'permessi' in filename_lower:
            return "Permessi e Ferie"
        elif 'supplenz' in filename_lower or 'gps' in filename_lower:
            return "Supplenze e Graduatorie"
        else:
            return "Documenti Generali"


class SchoolAssistant:
    def __init__(self, groq_api_key: str):
        """Inizializza l'assistente sindacale scuola"""
        self.client = Groq(api_key=groq_api_key)
        
        if 'embedding_model' not in st.session_state:
            st.session_state.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
            )
        self.embedding_model = st.session_state.embedding_model
        
        # Inizializza ChromaDB
        if 'chroma_client' not in st.session_state:
            st.session_state.chroma_client = chromadb.Client()
            try:
                st.session_state.collection = st.session_state.chroma_client.get_collection("school_docs")
            except:
                st.session_state.collection = st.session_state.chroma_client.create_collection(
                    name="school_docs",
                    metadata={"hnsw:space": "cosine"}
                )
        
        self.collection = st.session_state.school_collection
    
    def load_documents_into_db(self, documents):
        """Carica documenti nel database ChromaDB"""
        if not documents:
            return 0
        
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Genera embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
        
        # Genera IDs univoci
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
        
        # Aggiungi al database
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        return len(documents)
    
    def search(self, query: str, n_results: int = 4):
        """Cerca documenti rilevanti"""
        if self.collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]]}
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def answer_question(self, question: str, model: str = "llama-3.3-70b-versatile"):
        """Risponde usando RAG"""
        results = self.search(question, n_results=5)
        
        if not results['documents'][0]:
            context = "Nessun documento disponibile nel database."
            sources = []
        else:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            context_parts = []
            sources = []
            
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                fonte = f"{meta.get('filename', 'N/A')} - {meta.get('categoria', 'N/A')}"
                context_parts.append(f"[Fonte {i+1}: {fonte}]\n{doc}")
                sources.append({
                    "filename": meta.get('filename', 'N/A'),
                    "categoria": meta.get('categoria', 'N/A'),
                    "chunk": f"{meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}"
                })
            
            context = "\n\n".join(context_parts)
        
        prompt = f"""Sei un esperto consulente sindacale specializzato nel personale della scuola italiana (docenti, ATA, dirigenti). Conosci perfettamente CCNL Scuola, normative, contratti, graduatorie, concorsi.

DOCUMENTI DISPONIBILI:
{context}

DOMANDA: {question}

ISTRUZIONI:
- Rispondi in modo chiaro, pratico e professionale
- Cita SEMPRE le fonti specifiche quando usi informazioni dai documenti
- Se i documenti non contengono info sufficienti, usa la tua conoscenza delle normative scolastiche
- Fornisci informazioni operative (scadenze, procedure, riferimenti normativi)
- Se la questione è complessa, suggerisci di contattare il sindacato territoriale

RISPOSTA:"""

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Sei un esperto di normative scolastiche: CCNL, graduatorie, concorsi, diritti del personale."
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
    # Header
    st.title("🎓 Assistente Sindacale Scuola")
    st.markdown("*Documenti precaricati automaticamente*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurazione")
        
        api_key = st.text_input(
            "🔑 API Key Groq",
            type="password",
            help="Ottieni la tua chiave gratis su https://console.groq.com"
        )
        
        if not api_key:
            st.warning("⚠️ Inserisci l'API Key per iniziare")
            st.info("👉 Registrati gratis su https://console.groq.com")
            st.stop()
        
        st.success("✅ Sistema attivo")
        
        model = st.selectbox(
            "🤖 Modello",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            help="llama-3.3-70b è il più accurato"
        )
        
        st.divider()
        
        # Inizializza sistema
        assistant = initialize_system(api_key)
        
        # Info database
        st.header("📚 Database")
        doc_count = assistant.collection.count()
        st.metric("📄 Chunks totali", doc_count)
        
        # Mostra file caricati
        if 'files_loaded' in st.session_state and st.session_state.files_loaded:
            with st.expander(f"📁 File caricati ({len(st.session_state.files_loaded)})"):
                for filename in st.session_state.files_loaded:
                    st.write(f"✓ {filename}")
        
        if st.button("🔄 Ricarica documenti"):
            st.session_state.documents_loaded = False
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Consulenza", "📖 Esplora Database", "ℹ️ Info"])
    
    # TAB 1: Chat
    with tab1:
        st.header("💬 Chiedi all'assistente")
        
        # Domande frequenti per categoria
        st.markdown("**🔍 Domande frequenti:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📅 Ferie e permessi"):
                st.session_state.quick = "Quanti giorni di ferie ho come docente?"
        with col2:
            if st.button("💰 Stipendio"):
                st.session_state.quick = "Come funzionano gli scatti di anzianità?"
        with col3:
            if st.button("📋 Supplenze"):
                st.session_state.quick = "Differenza tra supplenza 31/08 e 30/06?"
        
        st.divider()
        
        # Cronologia chat
        if 'school_messages' not in st.session_state:
            st.session_state.school_messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Fonti utilizzate"):
                        for s in msg["sources"]:
                            st.write(f"• **{s['filename']}** ({s['categoria']}) - Chunk {s['chunk']}")
        
        # Input
        default_q = st.session_state.get('quick_q', '')
        if default_q:
            prompt = default_q
            st.session_state.quick_q = None
        else:
            prompt = st.chat_input("Scrivi la tua domanda... (es. 'Posso chiedere un'aspettativa?')")
        
        if prompt:
            st.session_state.school_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Ricerca nei documenti..."):
                    try:
                        response, sources = assistant.answer_question(prompt, model=model)
                        st.markdown(response)
                        
                        if sources:
                            with st.expander("📚 Fonti utilizzate"):
                                for s in sources:
                                    st.write(f"• **{s['filename']}** ({s['categoria']}) - Chunk {s['chunk']}")
                        
                        st.session_state.school_messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Errore: {e}")
        
        if st.button("🗑️ Nuova conversazione"):
            st.session_state.messages = []
            st.rerun()
    
    # TAB 2: Esplora
    with tab2:
        st.header("📖 Esplora Database")
        
        search_query = st.text_input(
            "🔍 Cerca nei documenti",
            placeholder="es. ferie, stipendio, GPS, maternità..."
        )
        
        if search_query:
            results = assistant.search(search_query, n_results=8)
            
            if results['documents'][0]:
                st.subheader(f"Trovati {len(results['documents'][0])} risultati")
                
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    with st.expander(
                        f"📄 {meta.get('filename', 'N/A')} - "
                        f"Chunk {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')} "
                        f"({meta.get('categoria', 'N/A')})"
                    ):
                        st.markdown(doc)
                        st.caption(f"Tipo: {meta.get('tipo', 'N/A')}")
            else:
                st.info("Nessun risultato trovato")
    
    # TAB 3: Info
    with tab3:
        st.header("ℹ️ Come Funziona")
        
        st.markdown("""
        ### 🎯 Sistema di Caricamento Automatico
        
        Questa app carica automaticamente tutti i documenti dalla cartella `documenti/` all'avvio.
        
        ### 📁 Struttura Cartelle
        
        ```
        sindacati-ai/
        ├── app_scuola.py
        ├── requirements.txt
        └── documenti/           ← Metti qui i tuoi file!
            ├── CCNL_Scuola_2016-2018.pdf
            ├── Circolare_Ferie.pdf
            ├── Contratto_Integrativo.docx
            └── Altri_documenti.txt
        ```
        
        ### 📤 Formati Supportati
        
        - ✅ **PDF** (.pdf)
        - ✅ **Word** (.docx)
        - ✅ **Testo** (.txt)
        
        ### 🔄 Aggiungere Nuovi Documenti
        
        1. Metti il file nella cartella `documenti/`
        2. Clicca sul pulsante **"🔄 Ricarica documenti"** nella sidebar
        3. Oppure riavvia l'app
        
        ### 🏷️ Categorie Automatiche
        
        L'app rileva automaticamente la categoria dal nome del file:
        - `ccnl` → CCNL Scuola
        - `circolare` → Circolari MIUR
        - `contratto`, `integrativo` → Contratto Integrativo
        - `ferie`, `permessi` → Permessi e Ferie
        - `supplenz`, `gps` → Supplenze e Graduatorie
        
        ### ⚠️ Note Importanti
        
        - I documenti vengono **chunked** (divisi in pezzi) per migliorare la ricerca
        - Ogni chunk è circa 1500 caratteri con overlap di 300
        - I documenti rimangono **in memoria** durante la sessione
        - **Per Streamlit Cloud**: metti i documenti nella repository Git!
        
        ### 💡 Consigli
        
        - Usa nomi file descrittivi (es. `CCNL_Scuola_2018.pdf`)
        - Documenti troppo corti (<100 caratteri) vengono ignorati
        - PDF scansionati potrebbero non funzionare (serve OCR)
        """)

if __name__ == "__main__":
    main()