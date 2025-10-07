"""
Script per precaricare documenti PDF/DOCX nel sistema scuola
Installa: pip install PyPDF2 python-docx
"""

import os
from PyPDF2 import PdfReader
from docx import Document
import json

class DocumentLoader:
    def __init__(self, documents_folder="documenti"):
        """Inizializza il loader con la cartella dei documenti"""
        self.documents_folder = documents_folder
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
    
    def load_pdf(self, filepath):
        """Carica un PDF e restituisce il testo"""
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Errore caricamento PDF {filepath}: {e}")
            return None
    
    def load_docx(self, filepath):
        """Carica un documento Word e restituisce il testo"""
        try:
            doc = Document(filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Errore caricamento DOCX {filepath}: {e}")
            return None
    
    def load_txt(self, filepath):
        """Carica un file di testo"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Errore caricamento TXT {filepath}: {e}")
            return None
    
    def split_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Divide il testo in chunks per migliorare la ricerca"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Cerca di terminare a fine frase
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def load_all_documents(self):
        """Carica tutti i documenti dalla cartella"""
        all_documents = []
        
        for filename in os.listdir(self.documents_folder):
            filepath = os.path.join(self.documents_folder, filename)
            
            if not os.path.isfile(filepath):
                continue
            
            print(f"📄 Caricamento: {filename}")
            
            # Determina il tipo di file
            if filename.endswith('.pdf'):
                text = self.load_pdf(filepath)
                doc_type = "PDF"
            elif filename.endswith('.docx'):
                text = self.load_docx(filepath)
                doc_type = "DOCX"
            elif filename.endswith('.txt'):
                text = self.load_txt(filepath)
                doc_type = "TXT"
            else:
                print(f"⚠️ Formato non supportato: {filename}")
                continue
            
            if text:
                # Dividi in chunks
                chunks = self.split_into_chunks(text)
                
                for i, chunk in enumerate(chunks):
                    all_documents.append({
                        "text": chunk,
                        "metadata": {
                            "filename": filename,
                            "tipo": doc_type,
                            "chunk": i + 1,
                            "total_chunks": len(chunks)
                        }
                    })
                
                print(f"  ✅ Caricato in {len(chunks)} chunks")
        
        return all_documents
    
    def save_to_json(self, documents, output_file="documenti_caricati.json"):
        """Salva i documenti in JSON per backup"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"💾 Salvato backup in {output_file}")


# ============= SCRIPT DI ESEMPIO =============

def main():
    print("🎓 CARICATORE DOCUMENTI SCUOLA")
    print("=" * 50)
    
    loader = DocumentLoader("documenti")
    
    print(f"\n📁 Cartella documenti: {os.path.abspath('documenti')}")
    print("📝 Metti i tuoi PDF/DOCX/TXT nella cartella 'documenti'")
    print("\nFormati supportati: .pdf .docx .txt")
    print("=" * 50 + "\n")
    
    # Carica tutti i documenti
    documents = loader.load_all_documents()
    
    if not documents:
        print("\n⚠️ Nessun documento trovato!")
        print("Crea una cartella 'documenti' e mettici dentro:")
        print("  - CCNL Scuola (PDF)")
        print("  - Circolari ministeriali")
        print("  - Contratti integrativi")
        print("  - Qualsiasi altro documento")
        return
    
    print(f"\n✅ Totale documenti caricati: {len(documents)} chunks")
    
    # Salva backup
    loader.save_to_json(documents)
    
    # Mostra statistiche
    print("\n📊 STATISTICHE:")
    files = {}
    for doc in documents:
        filename = doc['metadata']['filename']
        files[filename] = files.get(filename, 0) + 1
    
    for filename, count in files.items():
        print(f"  • {filename}: {count} chunks")
    
    print("\n" + "=" * 50)
    print("🎯 PROSSIMO PASSO:")
    print("Integra questi documenti nell'app Streamlit")
    print("usando la funzione add_documents_to_chromadb()")


def add_documents_to_chromadb(documents, assistant):
    """
    Funzione da integrare nell'app Streamlit per caricare i documenti
    
    Uso nell'app:
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    add_documents_to_chromadb(docs, assistant)
    """
    texts = [doc['text'] for doc in documents]
    metadata = [doc['metadata'] for doc in documents]
    
    embeddings = assistant.embedding_model.encode(texts).tolist()
    
    existing_count = assistant.collection.count()
    ids = [f"loaded_doc_{existing_count + i}" for i in range(len(texts))]
    
    assistant.collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=ids,
        metadatas=metadata
    )
    
    print(f"✅ {len(documents)} documenti aggiunti al database ChromaDB")


if __name__ == "__main__":
    main()