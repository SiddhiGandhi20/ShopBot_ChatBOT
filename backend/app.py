from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatbot import responses

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Load and preprocess chatbot responses
try:
    with open('chatbot.py', 'r', errors='ignore') as f:  # File for predefined responses
        exec(f.read())  # Executes chatbot.py, which should define the `responses` dictionary
except FileNotFoundError:
    print("Error: 'chatbot.py' file not found. Please ensure the file exists in the directory.")
    exit()

# Make sure the responses dictionary exists
if 'responses' not in globals():
    print("Error: 'responses' dictionary not found in 'chatbot.py'. Please ensure it's defined.")
    exit()

raw_doc = "Some predefined responses here."  # Placeholder, replace with actual content

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Tokenization
sent_tokens = nltk.sent_tokenize(raw_doc)

# Lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def greet(sentence):
    for word in sentence.split():
        word = word.lower()  # Convert to lowercase for case-insensitive comparison
        if word in responses:  # Check if the word exists in the responses dictionary
            return responses[word]  # Return the matching response
    return None


# Predefined responses for specific queries
PRODUCT_RESPONSES = {
     "can you help me": "Of course! I'm here to help. What do you need assistance with?",
    "show me the latest products": "Here are the latest products we have on offer: [Product A, Product B, Product C].",
    "what are your bestsellers": "Our bestsellers are: [Product X, Product Y, Product Z].",
    "what is your return policy": "Our return policy allows you to return products within 30 days of purchase. For more details, visit our returns page.",
    "do you offer discounts": "Yes, we have ongoing discounts! Check out our 'Deals' section for the latest offers.",
    "do you provide free shipping": "We offer free shipping on orders over $50. Standard shipping charges apply otherwise.",
    "can i track my order": "Yes, you can track your order using the tracking link sent to your email. Need help? Provide your order ID, and I'll assist you.",
    "are your products available internationally": "Yes, we ship internationally! Shipping charges and delivery times vary based on your location.",
    "do you have eco-friendly products": "Yes, we offer a range of eco-friendly products. Browse our 'Green Collection' for sustainable options.",
    "how can i cancel my order": "You can cancel your order within 24 hours of placing it. Contact customer support for assistance.",
    "can i pre-order a product": "Yes, you can pre-order select products. Check the product page for availability and delivery timelines.",
    "do you sell gift cards": "Yes, we offer gift cards ranging from $10 to $500. They're perfect for any occasion!",
    "how can i redeem a coupon code": "You can redeem your coupon code at checkout. Enter the code in the 'Promo Code' section and click 'Apply'.",
    "do you offer bulk discounts": "Yes, we provide bulk discounts for large orders. Contact our sales team for more details.",
    "how long does delivery take": "Standard delivery takes 3-7 business days. Express delivery options are available at checkout.",
    "do you provide installation services": "Yes, for select products, we provide installation services. Details are mentioned on the product page.",
    "how do i know if a product is in stock": "You can check product availability on the product page. If it's out of stock, you can sign up for restock notifications.",
    "can i request a product demo": "Yes, we offer demos for select products. Contact our support team to schedule one.",
    "what payment methods do you accept": "We accept credit/debit cards, PayPal, Apple Pay, Google Pay, and bank transfers.",
    "do you have a loyalty program": "Yes, join our loyalty program to earn points on every purchase and redeem them for discounts!",
    "do you offer a warranty": "Yes, we offer a warranty on select products. Details can be found on the product page.",
    "can i modify my order after placing it": "Once your order is placed, we cannot modify it. However, you can cancel it within 24 hours and place a new order.",
    "how do i contact customer support": "You can contact our customer support team via email at support@ourstore.com or call us at 1-800-123-4567.",
    "what is your privacy policy": "You can view our privacy policy on our website. We take your privacy seriously and ensure the safety of your data.",
    "how do i apply for a refund": "To request a refund, please visit the 'Returns and Refunds' section on our website and follow the instructions.",
    "do you offer a subscription service": "Yes, we offer a subscription service for select products. Check the product page for subscription options.",
    "how do i reset my account password": "You can reset your password by clicking on the 'Forgot Password' link on the login page and following the instructions.",
    "do you offer personalized products": "Yes, we offer personalized products. You can add custom text or designs to select items. Visit the product page for more information.",
    "can i get a discount for my first order": "Yes, we offer a special discount for first-time customers! Check your email for the discount code.",
    "how can I leave a review": "You can leave a review by visiting the product page and scrolling down to the 'Customer Reviews' section.",
    "is my payment secure": "Yes, all payments are processed securely through SSL encryption. Your payment details are never stored on our servers.",
    "how do I know if my order has been shipped": "You will receive an email with tracking information once your order has been shipped.",
    "do you offer gift wrapping": "Yes, we offer gift wrapping for an additional charge. You can select this option at checkout.",
    "can I exchange a product": "Yes, exchanges are possible within 30 days of purchase. Please refer to our exchange policy for more details.",
    "goodbye": "Goodbye! Have a great day ahead.",
    "bye": "Bye! Looking forward to assisting you again.",
    "see you later": "See you later! Take care.",
    "take care": "Take care! I'm here whenever you need help.",
    "have a great day": "Have a great day! Let me know if you need anything else.",
    "catch you later": "Catch you later! Feel free to return for assistance.",
    "I'm leaving now": "Alright! Have a good day!",
    "thank you": "You're welcome! Happy to help.",
    "thanks": "No problem! Let me know if you need anything else.",
    "I appreciate it": "Glad I could assist you.",
    "many thanks": "You're most welcome!",
    "I'm grateful": "It's my pleasure to help.",
    "thanks a ton": "You're welcome! Always here to assist.",
    "thank you so much": "You're very welcome! Happy to help.",
    "can you help me?": "Of course! Please tell me how I can assist you.",
    "I need assistance": "Sure! What do you need help with?",
    "can you assist me?": "I'm here to help! What can I do for you?",
    "I have a question": "Feel free to ask your question. I'm here for you.",
    "can you guide me?": "Absolutely! Tell me what you need guidance with.",
    "I need help with my account": "Sure! Let me know the issue you're facing with your account.",
    "how do I get started?": "I can help you get started! What would you like to know?",
    "how are you?": "I'm just a bot, but I'm here to make your day better!",
    "what's your name?": "My name is ShopBot! Let me help you with your shopping needs.",
    "what can you do?": "I can assist you with shopping queries, product info, order tracking, and more!",
    "tell me a joke": "Why did the shopper bring a ladder? To reach new heights of savings!",
    "make me laugh": "Why don't skeletons fight? They don't have the guts!",
    "are you human?": "Nope, I'm a chatbot here to help you 24/7.",
    "what are your skills?": "I can assist with shopping queries, product suggestions, and order support!",
    "can you tell me something interesting?": "Did you know online shopping is over 40 years old? It began in 1979!",
    "show me the latest products": "Here are our latest arrivals! [link or description]",
    "what are your bestsellers?": "Our bestsellers are listed here. Let me show you! [link or description]",
    "do you have discounts?": "Yes, we have discounts available! Check out these deals: [link].",
    "what are your current offers?": "Here are our current offers: [link].",
    "is there a sale going on?": "Yes, we have an ongoing sale! Check it out here: [link].",
    "what's trending right now?": "Here's what's trending: [Product 1, Product 2, Product 3].",
    "can you suggest some products?": "Sure! Let me suggest some great options for you.",
    "show me products under $50": "Here are some products under $50: [link or description].",
    "how can I track my order?": "You can track your order using the tracking number sent to your email.",
    "can I return a product?": "Yes, you can return products within 30 days. Check our return policy here: [link].",
    "what is your return policy?": "Our return policy allows returns within 30 days. More details here: [link].",
    "what's your name?": "You can track your order with the tracking details sent to your email.",
    "I want to check my order status": "You can check your order status on your account dashboard.",
    "can I get a refund?": "Refunds are processed within 7 business days after the return is approved.",
    "how can I cancel my order?": "You can cancel your order from your account dashboard or contact support.",
    "do you ship internationally?": "Yes, we ship internationally. Charges depend on your location.",
    "what are your shipping charges?": "Shipping charges vary based on your order and location. Check details at checkout.",
    "can I change my shipping address?": "Yes, you can change your shipping address from your account before dispatch.",
    "is express shipping available?": "Express shipping is available for select locations. Check options at checkout.",
    "how do I reschedule delivery?": "You can reschedule delivery by contacting support or using the delivery app link.",
    "how long does delivery take?": "Delivery times depend on your location. Typically, orders arrive in 3-7 days.",
    "how can I contact customer service?": "You can contact our customer service via chat, email, or phone.",
    "do you have a customer care number?": "Yes, our customer care number is available on our contact page.",
    "can I get live chat support?": "Yes, live chat support is available from 9 AM to 9 PM.",
    "is your support available 24/7?": "Our support is available 24/7 via email and chat.",
    "do you have gift cards?": "Yes, we offer gift cards! You can purchase them here: [link].",
    "what payment methods do you accept?": "We accept credit cards, PayPal, and cash on delivery.",
    "can I pay using cash on delivery?": "Yes, cash on delivery is available for select locations.",
    "is EMI available?": "Yes, EMI options are available for select products. Check details at checkout.",
    "how do I redeem a gift card?": "You can redeem a gift card at checkout by entering the code.",
    "can I use multiple payment methods?": "Currently, we allow one payment method per transaction.",
    "is online payment secure?": "Yes, our online payments are secure with encrypted transactions.",
    "do you have a loyalty program?": "Yes, join our loyalty program to earn points and get exclusive rewards!",
    "how can I redeem my reward points?": "You can redeem your reward points during checkout for discounts.",
    "is my personal information safe?": "Your personal information is protected by industry-standard encryption.",
    "how do I sign up for the loyalty program?": "Sign up for our loyalty program on our rewards page.",
    "can I pre-order a product?": "Yes, pre-orders are open for select products. Check the product page for details.",
    "do you offer installation services?": "We offer installation services for certain products. Contact support for assistance.",
    "how do I subscribe to your newsletter?": "Subscribe to our newsletter to stay updated on new arrivals and offers!",
    "do you have eco-friendly products?": "Yes, we have eco-friendly options. Check our eco-friendly category!",
    "what's your company mission?": "Our mission is to provide quality products with excellent customer service.",
    "where are your stores located?": "Our stores are located in multiple cities. Check our store locator for details."
}


def product_related_query(user_response):
    return PRODUCT_RESPONSES.get(user_response, None)

# Generate chatbot response
def response(user_response, previous_responses):
    print(f"User input: {user_response}")
    greeting_response = greet(user_response)
    if greeting_response:
        print(f"Found response in responses dictionary: {greeting_response}")
        return greeting_response
    
    print("Using cosine similarity...")
    product_response = product_related_query(user_response)
    if product_response:
        return product_response.strip()

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    user_tfidf = TfidfVec.transform([user_response])
    vals = cosine_similarity(user_tfidf, tfidf)
    print(f"Cosine similarity values: {vals}")
    
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]

    if req_tfidf < 0.1:
        return "I'm sorry, I didn't quite understand that. Can you rephrase?"
    else:
        response_text = sent_tokens[idx].strip()
        return response_text.strip()


# API endpoint for chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json.get('message', '').strip().lower()
    if not user_input:
        return jsonify({'reply': "I'm sorry, I didn't quite catch that."})
    
    # Call the response function
    reply = response(user_input, previous_responses=[])
    
    return jsonify({'reply': reply.strip()})  # Strip any trailing spaces or newline characters

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
