import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Collect FAQs related to a topic or product (questions and their answers). ---
faqs = {
    "What are your operating hours?": "Our operating hours are Monday to Friday, 9 AM to 5 PM IST.",
    "How can I contact customer support?": "You can reach customer support by calling 1-800-123-4567 or emailing support@example.com.",
    "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, and PayPal.",
    "Do you offer international shipping?": "Yes, we offer international shipping to most countries. Shipping fees and delivery times may vary.",
    "How do I reset my password?": "You can reset your password by clicking on 'Forgot Password' on the login page and following the instructions.",
    "Where is your main office located?": "Our main office is located in Bangalore, India.",
    "What is your refund policy?": "Our refund policy allows returns within 30 days of purchase for a full refund. Please see our website for more details.",
    "How are you doing today?": "I'm a bot, so I don't have feelings, but I'm ready to assist you!",
    "Hello there": "Hi! How can I help you today?",
    "What is your purpose?": "I'm here to answer your frequently asked questions.",
    "What are the features of your premium plan?": "Our premium plan includes unlimited storage, priority support, and advanced analytics.",
    "Tell me about the software features.": "The software offers real-time data processing, a user-friendly interface, and secure cloud integration.",
    "Does your product have [specific feature]?": "Yes, our product includes [specific feature].",
    "Can I get a demo of the product?": "Yes, you can request a demo by filling out the form on our website or contacting our sales team.",
    "What is the warranty period for your products?": "All our products come with a one-year warranty covering manufacturing defects.",
    "How do I track my order?": "You can track your order using the tracking link sent to your email after purchase.",
}

# --- Step 2: Preprocess the text using NLP libraries like NLTK. ---

print("Checking and downloading NLTK data (if necessary)...")

# ADD 'punkt_tab' to this list
required_nltk_data = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'punkt_tab']

for data_item in required_nltk_data:
    try:
        # For 'punkt_tab', it might be directly under 'tokenizers' or 'corpora'.
        # The `find` function is smart enough to search standard locations.
        nltk.data.find(f'tokenizers/{data_item}') # Try tokenizers first for punkt_tab
        print(f"'{data_item}' data already present.")
    except LookupError:
        try: # If not in tokenizers, try corpora (for stopwords, wordnet, omw-1.4)
            nltk.data.find(f'corpora/{data_item}')
            print(f"'{data_item}' data already present.")
        except LookupError: # If still not found, download it
            print(f"Downloading '{data_item}'...")
            nltk.download(data_item)
            print(f"Downloaded '{data_item}'.")
        except AttributeError:
             print(f"An AttributeError occurred while checking/downloading '{data_item}'. Attempting download anyway.")
             nltk.download(data_item)
             print(f"Downloaded '{data_item}'.")
    except AttributeError: # Catch the AttributeError for older NLTK versions or specific issues
        print(f"An AttributeError occurred while checking/downloading '{data_item}'. Attempting download anyway.")
        nltk.download(data_item)
        print(f"Downloaded '{data_item}'.")


print("NLTK data check complete.\n")


# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return " ".join(processed_tokens)

faq_questions_list_preprocessed = [preprocess_text(q) for q in faqs.keys()]

# --- Step 3: Match user questions with the most similar FAQ using cosine similarity. ---

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions_list_preprocessed)

def get_best_match(user_question, similarity_threshold=0.6):
    preprocessed_user_question = preprocess_text(user_question)
    user_question_vector = vectorizer.transform([preprocessed_user_question])

    similarities = cosine_similarity(user_question_vector, faq_vectors)

    best_match_index = similarities.argmax()
    best_match_similarity = similarities[0, best_match_index]

    if best_match_similarity >= similarity_threshold:
        original_faq_keys = list(faqs.keys())
        matched_original_question = original_faq_keys[best_match_index]
        return matched_original_question, best_match_similarity
    else:
        return None, best_match_similarity

# --- Step 4: Display the best matching answer as a chatbot response. ---

def chatbot_response(user_question):
    matched_original_question, similarity_score = get_best_match(user_question)

    if matched_original_question:
        answer = faqs[matched_original_question]
        print(f"Bot (Match Confidence: {similarity_score:.2f}): {answer}")
    else:
        print(f"Bot (No good match, highest confidence: {similarity_score:.2f}): I'm sorry, I don't have an answer to that specific question. Please try rephrasing or ask a different question.")

# --- Step 5 (Optional): Create a simple chat UI for user interaction (console-based). ---

if __name__ == "__main__":
    print("\n--- Simple FAQ Chatbot Started ---")
    print("Type your questions and press Enter.")
    print("Type 'quit' or 'exit' to end the chat.")

    while True:
        user_input = input("You: ")
        user_input_lower = user_input.lower().strip()

        if user_input_lower in ['quit', 'exit']:
            print("Bot: Goodbye! Have a great day.")
            break
        elif not user_input_lower:
            print("Bot: Please type a question.")
            continue
        
        chatbot_response(user_input)
