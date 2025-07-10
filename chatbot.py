from flask import *
from dotenv import load_dotenv
from flask_cors import CORS
import chromadb
from openai import OpenAI

# Lade Umgebungsvariablen aus .env (z. B. API-Keys)
load_dotenv()

# Flask-App erstellen
app = Flask(__name__)
CORS(app)  

# ChromaDB initialisieren
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="Abi-Vorgaben")

# OpenAI-Client vorbereiten
client = OpenAI()

# API-Endpunkt für den Chat
@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_query = request.form.get("question", "")
    if not user_query:
        return jsonify({"error": "Keine Frage erhalten."}), 400

    # Suche in ChromaDB
    results = collection.query(
        query_texts=[user_query],
        n_results=2
    )

    # System-Prompt mit den gefundenen Dokumenten
    system_prompt = f"""
    Du bist ein Assistent, der bei Fragen zu Abi-Vorgaben hilft –
    in Mathe, Informatik (Leistungskurs) und Englisch (Grundkurs).
    Du antwortest nur auf Basis von PDF-Daten.
    --------------------
    The data:
    {results['documents']}
    """

    # Anfrage an OpenAI senden
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route zum Laden der HTML-Seite

@app.route("/")
def index():
    return render_template("chatbot.html")

if __name__ == "__main__":
    app.run(debug=True, port=21648)