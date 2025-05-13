from src.correction.seq2seq_corrector import generate_corrections
from src.correction.suggestion_ranker import score_clarity, select_best_option

if __name__ == "__main__":
    # testing with a text example with errors
    text_with_errors = "As dori ca sa corecteze toate greșelile."

    # generation of correction options
    generated_options = generate_corrections(text_with_errors)
    print("Opțiuni generate:")
    for i, option in enumerate(generated_options):
        print(f"{i+1}. {option}")

    # clarity scores
    clarity_scores = score_clarity(generated_options)
    print("\nScoruri de claritate:")
    for i, (option, score) in enumerate(zip(generated_options, clarity_scores)):
        print(f"{i+1}. {option} - Scor: {score:.4f}")

    # chosing the best option
    best_option = select_best_option(generated_options, clarity_scores)
    print("\nVarianta finala selectata:", best_option)