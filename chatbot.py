from flask import Flask, render_template, request
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

# ChromaDB setup
CHROMA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="Abi-vorgaben")

# OpenAI client
client = OpenAI()

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        user_query = request.form["question"]

        # Search in ChromaDB
        results = collection.query(
            query_texts=[user_query],
            n_results=1
        )

        # Prepare prompt
        system_prompt = """
        You are a helpful assistant. You answer questions about growing vegetables in Florida. 
        But you only answer based on knowledge I'm providing you. You don't use your internal 
        knowledge and you don't make things up.
        If you don't know the answer, just say: I don't know
        --------------------
        The data:
        """ + str(results['documents'])

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        answer = response.choices[0].message.content

    return render_template("chatbot.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
