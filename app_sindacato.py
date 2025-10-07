"""
Assistente Sindacale per il Personale della Scuola con Groq + RAG
Sistema per docenti, ATA, dirigenti scolastici

Installa: pip install groq chromadb sentence-transformers streamlit
Esegui: streamlit run app_scuola.py
"""

import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from datetime import datetime

# Configurazione pagina
st.set_page_config(
    page_title="🎓 Assistente Sindacale Scuola",
    page_icon="🎓",
    layout="wide"
)

# Database normative scolastiche precaricate
NORMATIVE_SCUOLA = {
    "CCNL Scuola 2016-2018": [
        {
            "argomento": "Orario di lavoro docenti",
            "contenuto": "L'orario di insegnamento è di 18 ore settimanali nella scuola secondaria, 22 ore nella primaria, 25 ore nella scuola dell'infanzia. Le attività funzionali all'insegnamento (40 ore collegiali + 40 ore consigli) sono obbligatorie. Le ore eccedenti l'orario sono retribuite come ore aggiuntive."
        },
        {
            "argomento": "Ferie docenti",
            "contenuto": "I docenti hanno diritto a 32 giorni di ferie durante i periodi di sospensione delle attività didattiche (Natale, Pasqua, estate). Le ferie devono essere fruite prioritariamente nei periodi di sospensione. Il dirigente può richiamare in servizio solo in casi eccezionali documentati."
        },
        {
            "argomento": "Permessi retribuiti",
            "contenuto": "Spettano 3 giorni di permesso retribuito per anno scolastico per motivi personali o familiari. I permessi per lutto o grave infermità del coniuge o parente entro il 2° grado sono 3 giorni per evento. Permessi per matrimonio: 15 giorni consecutivi. I permessi brevi (max 2 ore) devono essere recuperati entro 2 mesi."
        },
        {
            "argomento": "Malattia",
            "contenuto": "Periodo di comporto: 18 mesi (9 mesi con intera retribuzione + 9 mesi con 90% della retribuzione). I primi 10 giorni di malattia nell'anno sono soggetti a decurtazione (50% il 1° evento, 100% dal 2° in poi, salvo ricovero ospedaliero o patologie gravi). Obbligo di reperibilità dalle 10-12 e 17-19."
        },
        {
            "argomento": "Mobilità",
            "contenuto": "La mobilità avviene tramite domanda volontaria entro i termini stabiliti dall'ordinanza ministeriale annuale (solitamente febbraio-marzo). Le operazioni si articolano in 3 fasi: trasferimenti, passaggi di cattedra, passaggi di ruolo. Il punteggio è determinato da anzianità, esigenze familiari, titoli."
        }
    ],
    "Personale ATA": [
        {
            "argomento": "Orario di lavoro ATA",
            "contenuto": "L'orario di lavoro è di 36 ore settimanali distribuite su 6 giorni (o 5 giorni per accordo). Collaboratori scolastici: 36 ore. Assistenti amministrativi: 36 ore. Assistenti tecnici: 36 ore. DSGA: orario flessibile funzionale alle esigenze dell'istituzione scolastica."
        },
        {
            "argomento": "Ferie ATA",
            "contenuto": "32 giorni lavorativi di ferie, da fruire prioritariamente nei periodi di sospensione delle attività didattiche. Le ferie non godute entro il 31 agosto devono essere fruite entro l'anno scolastico successivo. Il dirigente deve garantire la fruizione di almeno 15 giorni continuativi nel periodo estivo."
        },
        {
            "argomento": "Incarichi specifici ATA",
            "contenuto": "Gli incarichi specifici comportano compensi aggiuntivi finanziati dal FIS (Fondo dell'Istituzione Scolastica). Esempi: gestione laboratori, supporto informatico, primo soccorso, coordinamento biblioteca. Gli incarichi sono assegnati dal DSGA su proposta del dirigente e contrattazione RSU."
        },
        {
            "argomento": "Straordinario ATA",
            "contenuto": "Le ore eccedenti le 36 settimanali sono retribuite come lavoro straordinario. Limite massimo: 200 ore annuali recuperabili o retribuite. Compenso orario: quota oraria della retribuzione + maggiorazione. Lo straordinario deve essere preventivamente autorizzato dal DSGA."
        }
    ],
    "Supplenze e Precariato": [
        {
            "argomento": "Supplenza annuale (31/08)",
            "contenuto": "Contratto fino al 31 agosto per cattedre vacanti. Diritto a: intera retribuzione, scatti di anzianità, ferie estive pagate, TFS/TFR. Valutabile per ricostruzione di carriera. Conferimento tramite graduatorie GPS (I fascia laureati con abilitazione, II fascia laureati senza abilitazione)."
        },
        {
            "argomento": "Supplenza termine attività (30/06)",
            "contenuto": "Contratto fino al 30 giugno per cattedre di fatto disponibili. Diritti analoghi alla supplenza annuale ma senza retribuzione luglio/agosto (salvo proroga). Valutabile ai fini della ricostruzione di carriera. Conferimento da GPS."
        },
        {
            "argomento": "Supplenza breve e saltuaria",
            "contenuto": "Supplenze temporanee per assenze di titolari (malattia, maternità, ecc.). Retribuzione calcolata per giorni effettivi. Non maturano scatti né ferie. Conferimento dalle graduatorie d'istituto. Le supplenze superiori a 30 giorni danno diritto all'indennità di disoccupazione (NASpI)."
        },
        {
            "argomento": "GPS - Graduatorie Provinciali Supplenze",
            "contenuto": "Aggiornate biennalmente. Prima fascia: abilitati. Seconda fascia: non abilitati con 24 CFU. Validità: 2 anni. Le GPS hanno priorità sulle graduatorie d'istituto. È possibile iscriversi in una sola provincia per ciascuna classe di concorso."
        },
        {
            "argomento": "Graduatorie d'Istituto",
            "contenuto": "Derivano dalle GPS. Ogni aspirante può scegliere fino a 20 scuole della provincia. Utilizzate per supplenze brevi dopo esaurimento GPS. Aggiornate contestualmente alle GPS (ogni 2 anni). Possibile aggiornamento annuale per nuovi titoli."
        }
    ],
    "Congedi e Permessi Speciali": [
        {
            "argomento": "Legge 104 - Permessi",
            "contenuto": "Lavoratore disabile o per assistenza familiare disabile: 3 giorni mensili retribuiti o 2 ore giornaliere. Condizioni: handicap grave art.3 comma 3. Referente unico per l'assistenza. Non frazionabili in ore (salvo richiesta del lavoratore). Retribuiti al 100%, figurativi ai fini pensionistici."
        },
        {
            "argomento": "Congedo parentale",
            "contenuto": "10 mesi complessivi tra i genitori, fruibili fino ai 12 anni del bambino. Retribuzione: 80% fino a 6 anni del bambino, 30% dai 6 agli 8 anni, non retribuito dagli 8 ai 12 anni. Nella scuola: preferibile fruizione nei periodi di sospensione attività didattica."
        },
        {
            "argomento": "Maternità obbligatoria",
            "contenuto": "5 mesi: 2 mesi prima del parto + 3 dopo (o 1+4 con certificato medico). Retribuzione: 100% a carico dell'istituzione scolastica (anticipa per INPS). Interdizione anticipata possibile per gravidanza a rischio. Divieto di licenziamento dall'inizio gravidanza fino al 1° anno del bambino."
        },
        {
            "argomento": "Aspettativa non retribuita",
            "contenuto": "Aspettativa per motivi personali: fino a 12 mesi continuativi o frazionati (max 2 anni nell'arco della carriera). Non retribuita, non utile ai fini pensionistici e anzianità. Aspettativa per dottorato/ricerca: retribuita. Aspettativa per cariche pubbliche elettive: retribuita secondo normativa."
        }
    ],
    "Retribuzione e Carriera": [
        {
            "argomento": "Stipendio docenti",
            "contenuto": "Tabellare base + anzianità + scatti. Scatti: ogni 3 anni fino a 35 anni di servizio. Aumenti progressivi: da €1.350 iniziali a €2.200 finali (lordi mensili circa). Tredicesima mensilità a dicembre. Elemento perequativo: circa €80 mensili. RPD (Retribuzione Professionale Docenti) per chi non ha beneficiato degli scatti 2011-2014."
        },
        {
            "argomento": "Bonus merito/valorizzazione",
            "contenuto": "Abolito il bonus merito individuale (€500 card docente rimane). Introdotti compensi per attività aggiuntive dal FIS: funzioni strumentali, coordinatori, referenti progetti, ore eccedenti. Importi definiti in contrattazione integrativa d'istituto con RSU."
        },
        {
            "argomento": "Ricostruzione di carriera",
            "contenuto": "Domanda entro 1 anno dall'assunzione in ruolo. Riconoscimento servizi pre-ruolo: 100% servizio ruolo (anche altre amministrazioni), 66% supplenze annuali e TOI, 50% altre supplenze. Domanda telematica tramite Istanze Online. Decorrenza giuridica dalla domanda, economica dal 1° settembre successivo."
        },
        {
            "argomento": "Passaggio da tempo parziale a tempo pieno",
            "contenuto": "Possibile presentare domanda entro i termini della mobilità annuale. Priorità: motivi di salute documentati, esigenze familiari, anzianità di servizio. Il passaggio avviene dal 1° settembre. In part-time: stipendio e anzianità proporzionali (es. 50% = metà stipendio, 6 mesi anzianità)."
        },
        {
            "argomento": "TFS/TFR Scuola",
            "contenuto": "Personale assunto prima del 31/12/2010: TFS (Trattamento Fine Servizio) erogato da INPS con tempistiche variabili (fino a 24 mesi per età pensionabile). Personale dal 01/01/2011: possibilità di scegliere TFR presso fondo pensione. Calcolo TFS: 80% ultima retribuzione x anni servizio / 12."
        }
    ],
    "Graduatorie e Concorsi": [
        {
            "argomento": "Concorsi ordinari",
            "contenuto": "Prove: scritta + orale. Requisiti: laurea + abilitazione (o 24 CFU fino a dicembre 2024, poi 60 CFU). Titoli valutabili: servizio, titoli culturali, abilitazioni. Graduatorie di merito valide per assunzioni. Percorso annuale di formazione e prova per immissione in ruolo definitiva."
        },
        {
            "argomento": "Immissioni in ruolo",
            "contenuto": "50% da Graduatorie a Esaurimento (GAE), 50% da Graduatorie di Merito concorsi. Fase informatizzata tramite portale MIUR. Vincolo triennale nella provincia di assunzione. Possibilità di partecipare a mobilità dopo anno di prova positivo (mobilità straordinaria) o dopo 5 anni (ordinaria)."
        },
        {
            "argomento": "Anno di prova",
            "contenuto": "Obbligatorio per neoassunti. Durata: 180 giorni di servizio di cui 120 di attività didattica. Attività: 50 ore formazione (online + laboratori + peer to peer + bilancio competenze). Tutor assegnato. Valutazione finale da parte del dirigente scolastico e comitato valutazione."
        },
        {
            "argomento": "GPS e aggiornamenti",
            "contenuto": "Aggiornamento GPS ogni 2 anni (prossimo previsto 2024). Possibile inserimento nuovi titoli, spostamento provincia, cambio ordine scuola. Valutabili: servizio specifico, titoli culturali, certificazioni informatiche/linguistiche, master/perfezionamenti. Punteggio diverso per I e II fascia."
        }
    ],
    "Diritti e Doveri": [
        {
            "argomento": "Libertà di insegnamento",
            "contenuto": "Art. 33 Costituzione e art. 1 DPR 275/99: la libertà di insegnamento è garantita nel rispetto delle norme costituzionali e delle indicazioni nazionali. Il docente ha autonomia didattica e metodologica. Limite: rispetto curricolo nazionale, programmazione collegiale, PTOF."
        },
        {
            "argomento": "Codice disciplinare",
            "contenuto": "Sanzioni: richiamo verbale, richiamo scritto, multa, sospensione, licenziamento. Procedure: contestazione scritta, diritto di difesa (5 giorni), decisione con comunicazione. Sanzioni gravi: Ufficio Procedimenti Disciplinari (UPD) provinciale. Prescrizione: 5 anni. Reiterazione: aggrava la sanzione."
        },
        {
            "argomento": "Assenze per sciopero",
            "contenuto": "Diritto costituzionale di sciopero (art.40). Preavviso: 10 giorni. Trattenuta stipendio: 1/30 per giorno di sciopero. Obbligo comunicazione adesione: per garantire servizi minimi essenziali (vigilanza alunni). Possibile astensione da attività non obbligatorie senza trattenuta."
        },
        {
            "argomento": "Responsabilità docente",
            "contenuto": "Vigilanza alunni: obbligo di sorveglianza durante attività scolastiche, intervalli, entrata/uscita. Responsabilità civile: per danni causati da alunni durante orario di servizio (culpa in vigilando). Copertura assicurativa: RC professionale consigliata. Responsabilità penale: per reati commessi nell'esercizio della funzione."
        }
    ]
}

class SchoolUnionAssistant:
    def __init__(self, groq_api_key: str):
        """Inizializza l'assistente sindacale scuola"""
        self.client = Groq(api_key=groq_api_key)
        
        if 'embedding_model' not in st.session_state:
            with st.spinner('⚙️ Inizializzazione sistema...'):
                st.session_state.embedding_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
                )
        self.embedding_model = st.session_state.embedding_model
        
        if 'school_chroma_client' not in st.session_state:
            st.session_state.school_chroma_client = chromadb.Client()
            try:
                st.session_state.school_collection = st.session_state.school_chroma_client.get_collection("school_docs")
            except:
                st.session_state.school_collection = st.session_state.school_chroma_client.create_collection(
                    name="school_docs",
                    metadata={"hnsw:space": "cosine"}
                )
        
        self.collection = st.session_state.school_collection
    
    def preload_contracts(self):
        """Precarica le normative scolastiche"""
        if self.collection.count() > 0:
            return False
        
        all_docs = []
        all_metadata = []
        
        for categoria, contenuti in NORMATIVE_SCUOLA.items():
            for item in contenuti:
                all_docs.append(f"{item['argomento']}: {item['contenuto']}")
                all_metadata.append({
                    "categoria": categoria,
                    "argomento": item['argomento'],
                    "data_caricamento": datetime.now().isoformat()
                })
        
        embeddings = self.embedding_model.encode(all_docs).tolist()
        ids = [f"doc_{i}" for i in range(len(all_docs))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=all_docs,
            ids=ids,
            metadatas=all_metadata
        )
        
        return True
    
    def add_custom_content(self, text: str, categoria: str, argomento: str):
        """Aggiungi contenuto personalizzato"""
        embeddings = self.embedding_model.encode([text]).tolist()
        doc_id = f"custom_{self.collection.count()}"
        
        self.collection.add(
            embeddings=embeddings,
            documents=[text],
            ids=[doc_id],
            metadatas=[{
                "categoria": categoria,
                "argomento": argomento,
                "tipo": "personalizzato",
                "data_caricamento": datetime.now().isoformat()
            }]
        )
    
    def search_content(self, query: str, n_results: int = 4):
        """Cerca contenuti rilevanti"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def answer_question(self, question: str, model: str = "llama-3.1-70b-versatile"):
        """Risponde alla domanda con RAG"""
        results = self.search_content(question, n_results=4)
        
        if not results['documents'][0]:
            context = "Nessun documento rilevante trovato."
            sources = []
        else:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            context_parts = []
            sources = []
            
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                context_parts.append(f"[Fonte {i+1} - {meta['categoria']}, {meta['argomento']}]\n{doc}")
                sources.append({
                    "categoria": meta['categoria'],
                    "argomento": meta['argomento']
                })
            
            context = "\n\n".join(context_parts)
        
        prompt = f"""Sei un esperto consulente sindacale specializzato nel personale della scuola italiana (docenti, ATA, dirigenti). Conosci perfettamente CCNL Scuola, normative, contratti, graduatorie, concorsi.

CONTESTO (Estratti da CCNL e normative scolastiche):
{context}

DOMANDA: {question}

ISTRUZIONI:
- Rispondi in modo chiaro, pratico e professionale
- Cita SEMPRE le fonti quando usi informazioni dal contesto (es. "Secondo il CCNL Scuola...")
- Se il contesto non è sufficiente, usa la tua conoscenza delle normative scolastiche italiane
- Fornisci informazioni operative e pratiche (scadenze, procedure, modulistica)
- Usa un tono professionale ma accessibile
- Se la questione è complessa, suggerisci di rivolgersi al sindacato scolastico territoriale
- Distingui chiaramente tra docenti e ATA quando necessario
- Indica riferimenti normativi specifici quando possibile

RISPOSTA:"""

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Sei un esperto consulente sindacale del comparto scuola, specializzato in CCNL, graduatorie, concorsi, diritti e doveri del personale scolastico."
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
    st.markdown("*Consulenza per docenti, ATA e personale scolastico*")
    
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
            ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
            help="70b è più accurato per risposte complesse"
        )
        
        st.divider()
        
        # Info database
        st.header("📚 Database Normative")
        
        try:
            assistant = SchoolUnionAssistant(api_key)
            
            if assistant.collection.count() == 0:
                with st.spinner("📥 Caricamento normative scuola..."):
                    assistant.preload_contracts()
                    st.success("✅ Database caricato!")
            
            doc_count = assistant.collection.count()
            st.metric("📄 Articoli caricati", doc_count)
            
            with st.expander("📋 Contenuti disponibili"):
                for categoria in NORMATIVE_SCUOLA.keys():
                    st.write(f"✓ {categoria}")
        
        except Exception as e:
            st.error(f"Errore: {e}")
            st.stop()
    
    # Tabs principali
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Consulenza", "📥 Aggiungi Documenti", "📖 Esplora Database", "ℹ️ Info"])
    
    # TAB 1: Chat
    with tab1:
        st.header("💬 Chiedi all'assistente")
        
        # Domande frequenti per categoria
        st.markdown("**🔍 Domande frequenti:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📅 Ferie e permessi"):
                st.session_state.quick_q = "Quanti giorni di ferie ho come docente?"
        with col2:
            if st.button("💰 Stipendio e scatti"):
                st.session_state.quick_q = "Come funzionano gli scatti di anzianità?"
        with col3:
            if st.button("📋 Supplenze"):
                st.session_state.quick_q = "Differenza tra supplenza al 31/08 e 30/06?"
        with col4:
            if st.button("🔄 Mobilità"):
                st.session_state.quick_q = "Come funziona la mobilità dei docenti?"
        
        st.divider()
        
        # Cronologia chat
        if 'school_messages' not in st.session_state:
            st.session_state.school_messages = []
        
        for message in st.session_state.school_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Fonti normative"):
                        for source in message["sources"]:
                            st.write(f"• {source['categoria']} - {source['argomento']}")
        
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
                with st.spinner("🔍 Ricerca nelle normative..."):
                    try:
                        response, sources = assistant.answer_question(prompt, model=model)
                        st.markdown(response)
                        
                        if sources:
                            with st.expander("📚 Fonti normative"):
                                for source in sources:
                                    st.write(f"• {source['categoria']} - {source['argomento']}")
                        
                        st.session_state.school_messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Errore: {e}")
        
        if st.button("🗑️ Nuova conversazione"):
            st.session_state.school_messages = []
            st.rerun()
    
    # TAB 2: Aggiungi documenti
    with tab2:
        st.header("📥 Aggiungi Nuovi Documenti")
        
        st.info("💡 Aggiungi circolari ministeriali, contratti integrativi d'istituto, delibere, o altre normative specifiche")
        
        col1, col2 = st.columns(2)
        
        with col1:
            categoria_custom = st.selectbox(
                "📁 Categoria",
                ["CCNL Scuola", "Circolari MIUR", "Contratto Integrativo", "Normativa Locale", "Delibere", "Altro"]
            )
        
        with col2:
            argomento_custom = st.text_input(
                "🏷️ Argomento",
                placeholder="es. Bonus 150€, Organico COVID, ecc."
            )
        
        contenuto_custom = st.text_area(
            "📝 Contenuto",
            height=250,
            placeholder="Inserisci il testo della circolare, delibera o normativa...\n\nPuoi includere: circolari ministeriali, note USR, delibere collegio docenti, contrattazione d'istituto, ecc."
        )
        
        if st.button("➕ Aggiungi al Database", type="primary"):
            if contenuto_custom.strip() and categoria_custom.strip() and argomento_custom.strip():
                with st.spinner("Elaborazione..."):
                    try:
                        assistant.add_custom_content(
                            contenuto_custom,
                            categoria_custom,
                            argomento_custom
                        )
                        st.success(f"✅ Documento aggiunto con successo!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Errore: {e}")
            else:
                st.warning("⚠️ Compila tutti i campi")
    
    # TAB 3: Esplora database
    with tab3:
        st.header("📖 Esplora il Database Normativo")
        
        search_query = st.text_input(
            "🔍 Cerca nel database", 
            placeholder="es. ferie, GPS, ore eccedenti, maternità..."
        )
        
        if search_query:
            results = assistant.search_content(search_query, n_results=6)
            
            st.subheader(f"Trovati {len(results['documents'][0])} risultati:")
            
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                with st.expander(f"📄 {meta['categoria']} - {meta['argomento']}"):
                    st.markdown(doc)
                    st.caption(f"Tipo: {meta.get('tipo', 'precaricato')}")
    
    # TAB 4: Info
    with tab4:
        st.header("ℹ️ Informazioni sul Sistema")
        
        st.markdown("""
        ### 🎯 Cos'è l'Assistente Sindacale Scuola?
        
        Un sistema AI specializzato per il personale della scuola che fornisce consulenza immediata su:
        - **CCNL Comparto Istruzione e Ricerca**
        - **Supplenze e graduatorie** (GPS, GI, GAE)
        - **Concorsi** e immissioni in ruolo
        - **Diritti e doveri** del personale
        - **Retribuzione** e carriera
        - **Permessi** e congedi speciali
        
        ### 📚 Database Completo Precaricato
        
        **CCNL Scuola 2016-2018:**
        - ✅ Orario di lavoro docenti e ATA
        - ✅ Ferie e permessi
        - ✅ Malattia e comporto
        - ✅ Mobilità territoriale
        
        **Personale ATA:**
        - ✅ Orario 36 ore settimanali
        - ✅ Incarichi specifici
        - ✅ Lavoro straordinario
        - ✅ Diritti e doveri
        
        **Supplenze e Precariato:**
        - ✅ Supplenza annuale (31/08)
        - ✅ Supplenza termine attività (30/06)
        - ✅ Supplenze brevi
        - ✅ GPS e Graduatorie d'Istituto
        
        **Congedi e Permessi Speciali:**
        - ✅ Legge 104/92
        - ✅ Congedo parentale
        - ✅ Maternità obbligatoria
        - ✅ Aspettativa
        
        **Retribuzione e Carriera:**
        - ✅ Stipendio e scatti
        - ✅ Bonus e compensi
        - ✅ Ricostruzione carriera
        - ✅ Part-time e TFS/TFR
        
        **Graduatorie e Concorsi:**
        - ✅ Concorsi ordinari
        - ✅ Immissioni in ruolo
        - ✅ Anno di prova
        - ✅ Aggiornamenti GPS
        
        **Diritti e Doveri:**
        - ✅ Libertà di insegnamento
        - ✅ Codice disciplinare
        - ✅ Sciopero
        - ✅ Responsabilità
        
        ### 🚀 Come Usarlo
        
        1. **Fai domande specifiche** nella tab Consulenza
        2. **Usa i pulsanti rapidi** per temi comuni
        3. **Aggiungi circolari** del tuo istituto nella tab dedicata
        4. **Esplora** il database per trovare normative specifiche
        
        ### 💡 Esempi di Domande
        
        **Per Docenti:**
        - "Quante ore di lezione devo fare alla settimana?"
        - "Come funziona la mobilità volontaria?"
        - "Posso rifiutare ore eccedenti?"
        - "Quando posso usare i permessi della Legge 104?"
        
        **Per Supplenti:**
        - "Differenza tra GPS prima e seconda fascia?"
        - "Le supplenze brevi danno punteggio?"
        - "Quando escono le convocazioni?"
        - "Ho diritto alla disoccupazione?"
        
        **Per ATA:**
        - "Quante ore di straordinario posso fare?"
        - "Come funzionano gli incarichi specifici?"
        - "Posso chiedere il part-time?"
        
        ### ⚠️ Note Importanti
        
        - Questo è uno strumento di **prima consulenza**
        - Per questioni legali complesse **rivolgiti al sindacato**
        - Le normative possono essere aggiornate: **verifica sempre**
        - **Non sostituisce** la consulenza legale professionale
        
        ### 🔗 Risorse Utili
        
        - [MIUR - Ministero Istruzione](https://www.miur.gov.it)
        - [Istanze Online](https://www.istruzione.it/polis/Istanzeonline.htm)
        - [NoiPA - Stipendi](https://noipa.mef.gov.it)
        - [FLC CGIL Scuola](https://www.flcgil.it)
        - [CISL Scuola](https://www.cislscuola.it)
        - [UIL Scuola](https://www.uilscuola.it)
        - [SNALS](https://www.snals.it)
        
        ### 📞 Contatti Sindacati
        
        Per assistenza diretta, contatta il sindacato della tua provincia:
        - **FLC CGIL**, **CISL Scuola**, **UIL Scuola**, **SNALS**, **GILDA**
        
        ---
        
        <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💙 Sviluppato per il personale della scuola italiana</p>
        <p>Powered by Groq + RAG Technology</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()