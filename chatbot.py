from flask import *
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from flask_cors import CORS  # Wichtig für Zugriff aus PyScript/Browser

load_dotenv()

app = Flask(__name__)
CORS(app)  # Erlaubt Zugriff vom Browser (z. B. PyScript)

# ChromaDB Setup
CHROMA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="Abi-vorgaben")

# OpenAI Client
client = OpenAI()
@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_query = request.form.get("question", "")
    if not user_query:
        return jsonify({"error": "Keine Frage erhalten."}), 400

    # ChromaDB-Suche
    results = collection.query(
        query_texts=[user_query],
        n_results=1
    )

    system_prompt = """
    Du bist ein Assistent, der bei Fragen zu Abi-Vorgaben hilft – 
    in Mathe, Informatik (Leistungskurs) und Englisch (Grundkurs). 
    Du antwortest nur auf Basis von PDF-Daten.
    --------------------
    The data:
    """ + str(results["documents"])

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

if __name__ == "__main__":
    app.run(debug=True, port=8080)
