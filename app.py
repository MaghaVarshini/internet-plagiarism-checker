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


def similarity(text, snippets):
    """
    Compares a text chunk with multiple snippets and returns the highest similarity %
    """
    if not snippets:
        return 0
    vec = TfidfVectorizer().fit([text] + snippets)
    text_vec = vec.transform([text])
    snippets_vec = vec.transform(snippets)
    sims = cosine_similarity(text_vec, snippets_vec)
    return round(max(sims[0]) * 100, 2)



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
        file = request.files.get("file")
        if not file:
            return render_template("index.html", results=[], error="No file uploaded")

        # Extract text from the uploaded file
        text = extract_text(file)

        # Split text into chunks of 3 sentences for better similarity
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
        chunks = []
        for i in range(0, len(sentences), 3):
            chunk = ". ".join(sentences[i:i+3])
            if chunk:
                chunks.append(chunk)

        for chunk in chunks:
            # Search Bing for this chunk
            snippets = search_bing(chunk)

            # Calculate max similarity
            max_sim = 0
            if snippets:
                # Use TF-IDF vectorizer for chunk vs all snippets
                vec = TfidfVectorizer(stop_words='english').fit([chunk] + snippets)
                chunk_vec = vec.transform([chunk])
                snippets_vec = vec.transform(snippets)
                sims = cosine_similarity(chunk_vec, snippets_vec)
                max_sim = round(max(sims[0]) * 100, 2)

            results.append((chunk, max_sim))

    return render_template("index.html", results=results)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
