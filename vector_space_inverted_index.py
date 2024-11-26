import re
import pymongo
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["search_engine"]
terms_collection = db["terms"]
documents_collection = db["documents"]

# Document content from Question 3
documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The patient reported nausea and dizziness caused by the medication.",
    "Headache and dizziness are common effects of this medication.",
    "The medication caused a headache and nausea, but no dizziness was reported."
]

# Preprocess documents
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

preprocessed_documents = [preprocess(doc) for doc in documents]

# Insert documents into MongoDB
documents_collection.delete_many({})  # Clear existing documents
document_ids = []
for i, doc in enumerate(preprocessed_documents):
    document = {"_id": i + 1, "content": doc}
    documents_collection.insert_one(document)
    document_ids.append(i + 1)

# Generate unigrams, bigrams, and trigrams
def generate_ngrams(text, n):
    words = text.split()
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

terms_collection.delete_many({})  # Clear existing terms
vocabulary = set()

# Generate n-grams and build vocabulary
for doc_id, doc in enumerate(preprocessed_documents, 1):
    words = doc.split()
    unigrams = generate_ngrams(doc, 1)
    bigrams = generate_ngrams(doc, 2)
    trigrams = generate_ngrams(doc, 3)
    ngrams = unigrams + bigrams + trigrams
    
    for pos, term in enumerate(ngrams):
        vocabulary.add(term)
        terms_collection.update_one(
            {"term": term},
            {"$addToSet": {"docs": {"doc_id": doc_id, "pos": pos}}},
            upsert=True
        )

# Calculate TF-IDF values
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Update terms with TF-IDF values
for idx, term in enumerate(feature_names):
    tfidf_values = tfidf_matrix[:, idx].toarray().flatten()
    docs_with_tfidf = []
    for doc_id, tfidf_value in enumerate(tfidf_values, 1):
        if tfidf_value > 0:
            docs_with_tfidf.append({"doc_id": doc_id, "tfidf": tfidf_value})
    terms_collection.update_one(
        {"term": term},
        {"$set": {"tfidf_values": docs_with_tfidf}},
        upsert=True
    )

# Function to search and rank documents
def search(query):
    query = preprocess(query)
    query_terms = query.split()
    matching_docs = {}

    for term in query_terms:
        term_data = terms_collection.find_one({"term": term})
        if term_data and "tfidf_values" in term_data:
            for entry in term_data["tfidf_values"]:
                doc_id = entry["doc_id"]
                tfidf_value = entry["tfidf"]
                if doc_id not in matching_docs:
                    matching_docs[doc_id] = 0
                matching_docs[doc_id] += tfidf_value

    # Rank documents by score
    ranked_docs = sorted(matching_docs.items(), key=lambda x: x[1], reverse=True)
    for doc_id, score in ranked_docs:
        doc_content = documents_collection.find_one({"_id": doc_id})["content"]
        print(f"Document: {doc_content}, Score: {score}")

# Example queries
queries = [
    "nausea and dizziness",
    "effects",
    "nausea was reported",
    "dizziness",
    "the medication"
]

for q in queries:
    print(f"\nResults for query: '{q}'")
    search(q)
