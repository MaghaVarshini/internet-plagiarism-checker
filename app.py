# Internet Plagiarism Checker – Ready Web App (Stylish UI)

from flask import Flask, render_template, request
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import PyPDF2

app = Flask(__name__)

BING_API_KEY = "PASTE_YOUR_BING_KEY"
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def extract_text(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")

    if file.filename.endswith(".docx"):
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)

    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    return ""


def similarity(a, b):
    vec = TfidfVectorizer().fit_transform([a, b])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0] * 100


def search_bing(sentence):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": sentence}

    r = requests.get(BING_ENDPOINT, headers=headers, params=params)
    data = r.json()

    snippets = []
    if "webPages" in data:
        for page in data["webPages"]["value"]:
            snippets.append(page.get("snippet", ""))

    return snippets


@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        file = request.files["file"]
        text = extract_text(file)

        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 25]

        for s in sentences:
            snippets = search_bing(s)
            max_sim = 0

            for snip in snippets:
                max_sim = max(max_sim, similarity(s, snip))

            results.append((s, round(max_sim, 2)))

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run()
