import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Did you see a school bus last week?")
print([chunk.text for chunk in doc.noun_chunks])
