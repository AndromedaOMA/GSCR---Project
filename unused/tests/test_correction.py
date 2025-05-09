from src.correction.seq2seq_corrector import generate_corrections
from src.correction.suggestion_ranker import score_clarity, select_best_option

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