import spacy
import json

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def extract_info(text):
    # Process the input text using spaCy
    doc = nlp(text)

    feature = "flight booking"
    from_place = None
    to_place = None

    # Initialize a list to keep track of potential places
    places = []

    # Check for presence of keywords 'from' and 'to' in the text
    is_first_occurrence_from = False
    if "from" in text and "to" in text:
        is_first_occurrence_from = text.index("from") < text.index("to")
    elif "from" in text:
        is_first_occurrence_from = True

    # Loop through each entity in the text
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE (Geopolitical Entity) label is used for place names
            places.append(ent.text)

    # Assign from and to places based on the identified keywords
    if len(places) >= 2:
        if is_first_occurrence_from:
            from_place = places[0]
            to_place = places[1]
        else:
            from_place = places[1]
            to_place = places[0]
    elif len(places) == 1:
        if "from" in text:
            from_place = places[0]
        elif "to" in text:
            to_place = places[0]

    # Create the JSON object with conditional fields
    json_object = {"feature": feature}

    if from_place:
        json_object["from"] = from_place
    if to_place:
        json_object["to"] = to_place

    return json_object

# Example text inputs from Speech-to-Text
input_texts = [
    "I would like to book a flight from New York to Los Angeles next week.",
    "Book a flight from London.",
    "Can you find a flight to Tokyo?",
    "I need to travel to Paris.",
    "Book me a flight from Mumbai.",
    "What about flights from Delhi?",
]

# Extract information and create JSON objects for each example
for text in input_texts:
    result = extract_info(text)
    json_result = json.dumps(result, indent=4)
    print(json_result)
