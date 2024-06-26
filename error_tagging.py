from collections import Counter
import spacy
from pathlib import Path
import jsonlines

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    data_dir = Path("data/conll_logs/")

    for error_class_path in data_dir.iterdir():
        with jsonlines.jsonlines.open(error_class_path, "r") as io:
            docs = list(io.iter())
        for doc in docs:
            tags = nlp(doc["text"])
            spans = {(token.idx, token.idx + len(token)): token.tag_ for token in tags}
        
            for entity in doc["clusters"]:
                tags = tuple(spans.get(x, "None") for x in entity)
                counts = Counter(tags)
