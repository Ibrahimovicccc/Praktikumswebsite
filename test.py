from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chatbot.html")

if __name__ == "__main__":
    app.run(debug=True, port=21648)