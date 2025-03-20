import os
import pandas as pd
import torch
import pickle
from django.http import JsonResponse
from sentence_transformers import SentenceTransformer, util


predefined_responses = {
    # Greetings & Small Talk
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I do for you?",
    "hey": "Hey! How’s your day going?",
    "good morning": "Good morning! Hope you have a great day!",
    "good afternoon": "Good afternoon! How can I assist you?",
    "good evening": "Good evening! What’s on your mind?",
    
    # Emotions & Feelings  
    "how are you": "I'm just a chatbot, but I'm here to assist you!",
    "how’s it going": "I’m doing great! Thanks for asking. How about you?",
    "i'm feeling down": "I’m here for you. Want to share what’s on your mind?",
    "i feel lonely": "You’re not alone! I'm here to listen if you want to talk.",
    "i'm sad": "I'm sorry you're feeling this way. Want to talk about it?",
    "i feel empty": "Feeling empty can be tough. Do you want to explore what might be causing it?",
    "i’m stressed": "Stress is tough. Have you tried deep breathing or talking to someone about it?",
    "i have anxiety": "Anxiety can be overwhelming. Grounding techniques like the 5-4-3-2-1 method can help.",
    "how to calm anxiety": "Try deep breathing, mindfulness, or progressive muscle relaxation. Would you like some guided exercises?",
    "i feel overwhelmed": "Taking small breaks and prioritizing tasks can help. Do you want to talk about what's overwhelming you?",
    
    # Depression & Mental Health Support  
    "i feel depressed": "I'm really sorry you're feeling this way. Talking to a friend or therapist may help.",
    "what is depression": "Depression is a mental health condition that affects mood, thoughts, and energy levels. You're not alone.",
    "how to deal with depression": "Engaging in self-care, social support, therapy, and mindfulness can help.",
    "why do i feel sad for no reason": "Sometimes emotions arise without a clear reason. It might be helpful to explore your thoughts and feelings.",
    "does therapy help": "Yes! Therapy can help you understand and manage your emotions better.",
    "how do i know if i need therapy": "If you’re struggling with emotions, thoughts, or daily life, talking to a therapist could be helpful.",
    "i feel hopeless": "I'm really sorry you feel this way. You deserve support. Talking to someone may help.",
    "does talking help mental health": "Yes! Expressing your feelings can lighten emotional burdens.",
    
    # Self-Care & Coping Strategies  
    "how to practice self-care": "Self-care can be anything that makes you feel good—exercise, reading, rest, or even a hobby!",
    "how to deal with stress": "Try deep breathing, exercise, and talking to someone you trust.",
    "why is journaling helpful": "Journaling helps process emotions, clear thoughts, and track personal growth.",
    "how to improve self-esteem": "Focus on your strengths, practice self-compassion, and challenge negative thoughts.",
    "how to stop negative thoughts": "Try cognitive reframing—challenge negative thoughts and replace them with balanced perspectives.",
    "how to manage emotions": "Recognizing emotions, taking deep breaths, and practicing mindfulness can help.",
    "what is mindfulness": "Mindfulness is being fully present in the moment without judgment.",
    "how can i be more positive": "Practicing gratitude, self-care, and surrounding yourself with positive influences can help.",
    
    # Therapy & Psychology  
    "what is therapy": "Therapy is a space where you can talk to a professional about your thoughts and emotions to find better ways to cope.",
    "how does therapy work": "Therapists help you explore emotions, behaviors, and coping strategies in a supportive environment.",
    "what is cognitive behavioral therapy": "CBT is a therapy that helps identify and change negative thought patterns.",
    "does meditation help mental health": "Yes! Meditation can reduce stress, increase focus, and improve emotional well-being.",
    "how to control anger": "Try deep breathing, taking a short walk, or identifying triggers to help manage anger.",
    
    # Personal Development  
    "how to build confidence": "Confidence grows when you challenge yourself and acknowledge your strengths.",
    "why do i overthink": "Overthinking often comes from anxiety or fear of uncertainty. Practicing mindfulness can help.",
    "how to be happy": "Happiness comes from self-acceptance, gratitude, and meaningful connections.",
    "how to make friends": "Being open, kind, and engaging in social activities can help build friendships.",
    
    # Sleep & Health  
    "why can't i sleep": "Insomnia can be caused by stress or screen time before bed. A bedtime routine might help.",
    "how to sleep better": "Avoid screens before bed, keep a regular schedule, and practice relaxation techniques.",
    "why am i tired all the time": "Fatigue can come from stress, poor sleep, or underlying health conditions.",
    
    # Existential Questions  
    "what is the meaning of life": "The meaning of life is personal—some find meaning in love, learning, or helping others.",
    "do i have a purpose": "Everyone has a purpose, even if it takes time to discover it.",
    
    # Handling Unclear or Off-Topic Questions  
    "i don't know what to do": "That’s okay! It’s okay to feel uncertain. Talking it through might help.",
    "what should i do now": "That depends! What are your interests or goals?",
    "do you understand me": "I try my best! If I ever get confused, feel free to clarify.",
    "i need help": "I'm here for you! If you need professional help, talking to a therapist might be beneficial.",
    
    # Default Response for Unrecognized Questions  
    "default": "I'm sorry, but I can't help with that. Maybe try asking something else?",
}


# ✅ Get the absolute path of the chatbot directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'chatbot', 'chatbot_embeddings.pkl')

# ✅ Check if chatbot_embeddings.pkl exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train and save the model.")

# ✅ Load dataset with embeddings
with open(MODEL_PATH, "rb") as file:
    df = pickle.load(file)

# ✅ Load pre-trained transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Function to find the best match
def find_best_match(user_question, threshold=0.5):
    # ✅ Check predefined responses first
    user_question_lower = user_question.lower().strip()
    if user_question_lower in predefined_responses:
        return predefined_responses[user_question_lower]

    # ✅ Encode user question
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    # ✅ Compute cosine similarity with dataset questions
    similarity_scores = df["question_embedding"].apply(lambda x: util.pytorch_cos_sim(user_embedding, x).item())

    # ✅ Get the best match index
    best_idx = similarity_scores.idxmax()
    best_score = similarity_scores[best_idx]

    # ✅ Return best match if above threshold, else fallback response
    if best_score >= threshold:
        return df.loc[best_idx, "cleaned_answer"]
    else:
        return "I'm sorry, I don't understand. Can you rephrase?"


from django.shortcuts import render

def chatbot_page(request):
    return render(request, "chatbot.html") 


def chatbot_response(request):
    if request.method == "GET":
        user_question = request.GET.get("question", "").strip()

        if not user_question:  # Handle empty input
            return JsonResponse({"response": "Please ask a question!"})

        # ✅ Find best response
        response = find_best_match(user_question)
        
        return JsonResponse({"response": response})

# ✅ Simple chatbot logic (Extra Function)
def generate_chatbot_reply(message):
    """ A simple chatbot logic (Replace with AI logic) """
    responses = {
        "hello": "Hi there! How can I assist you?",
        "how are you": "I'm just a bot, but I'm doing great! How about you?",
        "bye": "Goodbye! Have a great day!",
    }
    return responses.get(message.lower(), "I'm not sure how to respond to that.")
