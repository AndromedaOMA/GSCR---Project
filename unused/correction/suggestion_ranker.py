from seq2seq_corrector import generate_corrections
from sentence_transformers import SentenceTransformer
import numpy as np


def score_clarity(options):
    """ Atribuie scor fiecărei opțiuni în funcție de claritate """
    # Încarcă modelul SentenceTransformer pentru claritate
    clarity_model = SentenceTransformer("BlackKakapo/cupidon-mini-ro")

    reference_sentence = "Aceasta este o propoziție clară și corectă din punct de vedere gramatical."
    ref_embedding = clarity_model.encode(reference_sentence)

    scores = []
    for option in options:
        option_embedding = clarity_model.encode(option)
        similarity = np.dot(ref_embedding, option_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(option_embedding))
        scores.append(similarity)

    return scores


def select_best_option(options, scores):
    """ Selectează cea mai clară variantă bazată pe scoruri """
    best_index = scores.index(max(scores))
    return options[best_index]


if __name__ == "__main__":
    # Testăm cu un text greșit gramatical
    text_with_errors = "As dori ca sa corecteze toate greșelile."

    # 1️⃣ Generăm opțiuni multiple
    generated_options = generate_corrections(text_with_errors)
    print("Opțiuni generate:")
    for i, option in enumerate(generated_options):
        print(f"{i+1}. {option}")

    # 2️⃣ Calculăm scorurile de claritate
    clarity_scores = score_clarity(generated_options)
    print("\nScoruri de claritate:")
    for i, (option, score) in enumerate(zip(generated_options, clarity_scores)):
        print(f"{i+1}. {option} - Scor: {score:.4f}")

    # 3️⃣ Selectăm varianta optimă
    best_option = select_best_option(generated_options, clarity_scores)
    print("\n✅ Varianta finală selectată:", best_option)