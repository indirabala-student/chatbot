from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import re
import pandas as pd

app = Flask(__name__)

client = OpenAI(
    api_key
)

# Load your specialist CSV file
specialist_df = pd.read_csv('C:/Users/nmurthy/OneDrive - GalaxE. Solutions, Inc/Desktop/Disease Predictor/Disease_Prediction_System/specialists.csv')


# Extract clean location from GPT response
def extract_location(text):
    match = re.search(r"Location:\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


# Match if GPT explicitly says "consult a Cardiologist", etc.
def extract_specialist_from_text(text):
    possible_specialists = [s.lower() for s in specialist_df["specialist"].unique()]
    for specialist in possible_specialists:
        if specialist in text.lower():
            return specialist.title()
    return None


# Match based on diseases listed in CSV
def map_disease_to_specialist(text):
    for _, row in specialist_df.iterrows():
        diseases = row["example_diseases"].split(",")
        for disease in diseases:
            if disease.strip().lower() in text.lower():
                return row["specialist"]
    return None


# Match based on body system phrases (e.g., "visual system")
def map_body_system_to_specialist(text):
    for _, row in specialist_df.iterrows():
        if row["body_system"].lower() in text.lower():
            return row["specialist"]
    return None


# Tiered matching logic: explicit → disease → body system
def resolve_specialist(text):
    specialist = extract_specialist_from_text(text)
    if specialist:
        return specialist
    specialist = map_disease_to_specialist(text)
    if specialist:
        return specialist
    specialist = map_body_system_to_specialist(text)
    if specialist:
        return specialist
    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.get_json("message")
    chat_history = user_input.get("history", [])
    user_message = user_input.get("message", "")
    user_city = user_input.get("city")

    # Check if GPT previously asked for city
    last_ai_reply = next((msg for msg in reversed(chat_history) if msg["role"] == "assistant"), None)

    if last_ai_reply:
        last_reply_lower = last_ai_reply["content"].lower()
        if not user_city:
            user_city = extract_location(user_message)

        if user_city and any(
            phrase in last_reply_lower
            for phrase in ["which city", "your city", "where are you located", "please tell me"]
        ):
            found_specialty = resolve_specialist(last_reply_lower)
            if found_specialty:
                reply_text = f"Thanks! Noted your location as **{user_city.strip()}** for a **{found_specialty}** referral. Please wait while I fetch doctor details..."
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": reply_text})
                return jsonify({"reply": reply_text, "history": chat_history})

    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_message})

    # Prepare the system prompt for GPT
    outmessages = [{
        "role": "system",
        "content": """
You are a helpful AI doctor assistant that diagnoses symptoms, assesses severity, and refers users to local doctors.

Instructions:
1. First assess the severity of the condition: mild, moderate, or severe.
2. If mild: suggest only home remedies , OTC medications, Suggest Food Diets and follow-up questions.
3. If moderate or severe:
    - Recommend lab tests (in a separate paragraph).
    - Suggest the type of medical specialist to consult.
    - Generate a short summary script the user can take to the doctor.
4.Ask for Current City, return a list of **5 realistic doctor names** mpathetic message:in that city who match the required specialty (invent names + clinic + address + phone if needed).
5.Format the location as:
Location: [place]

Format for doctors:
Doctors in [City] for [Specialty]:
- Dr. Name, Clinic Name, Address, Phone Number.
- Dr. Name, Clinic Name, Address, Phone Number.

5. Ask for booking an appointment with the Doctor by asking Doctor’s name. If user provides a name, validate it against your list. 
    If its not there in the List Provide deny Message politely and inform to select from provided list
6. After successful validation, provide appointment details in the following format:

Doctor: Dr. Name (Specialty)  
Clinic: Clinic Name  
Location: Address  
Date: For the next day of appointment booking (e.g., Random Days Friday, 17 May 2025) 
Time: Preferably morning or afternoon (e.g., Random timings 11:30 AM)  
Contact: Phone Number

Please arrive 10 minutes early and bring any past prescriptions or health reports.

7. End with a warm, e
We wish you a smooth and speedy recovery.  
Get well soon — your health matters most to us!

Return severity level as the first line: (Severity: Mild/Moderate/Severe)

Always be clear and concise. If the topic is unrelated to health, say so politely.
"""
    }] + chat_history

    # Call GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=outmessages
    )

    reply = response.choices[0].message
    reply_text = reply.content
    chat_history.append(dict(reply))

    # Severity check
    severity_level = None
    if "severity: mild" in reply_text.lower():
        severity_level = "mild"
    elif "severity: moderate" in reply_text.lower():
        severity_level = "moderate"
    elif "severity: severe" in reply_text.lower():
        severity_level = "severe"

    # Ensure city and specialty are stored (for continuity)
    if not user_city:
        user_city = extract_location(reply_text)
    found_specialty = resolve_specialist(reply_text)

    # Ask for city if not found and severity is serious
    if severity_level in ["moderate", "severe"] and not user_city:
        reply_text += "\n\n To help you find a nearby specialist, please tell me which **city or area** you're located in."

    return jsonify({"reply": reply_text, "history": chat_history})


if __name__ == "__main__":
    app.run(debug=True)
