# CIKM 2026 paper preparation and methodology
_Exported on 4/21/2026 at 15:20:41 GMT+3 from Cursor (3.1.17)_

---

**User**

Verifica proiectul, documentatia etc.. Vreau as vedem cum putem scoate o lucrare pentru conferinta: https://cikm2026.diag.uniroma1.it

Pentru a transforma acest proiect într-un Full Paper de succes la CIKM 2026, trebuie să rezolvăm marea verigă lipsă: evaluarea credibilă și noutatea metodologică.Arhiva ta arată un sistem robust (BERT/ULMFiT pentru detecție + T5 pentru corecție), dar pentru o conferință de Rank A, "merge bine" nu este o dovadă științifică. Iată planul de atac pentru benchmark-ul intern și noutatea necesară:1. Crearea Benchmark-ului Intern (Protocol Experimental)Nu putem folosi doar test.txt (care este probabil sintetic sau limitat). Propun un "GSCR-Benchmark-RO" structurat pe trei niveluri de dificultate:A. Level 1: In-Distribution (Cele 1.519 perechi din test)Sursă: Fișierul test.txt pe care îl ai deja.Scop: Demonstrează că modelul a învățat distribuția datelor de antrenament.Metrici: Accuracy, BLEU, chrF.B. Level 2: Stress Test (Diacritics & Agreement)Sursă: Generăm automat din corpus.txt și test.txt prin eliminarea sistematică a diacriticelor și alterarea terminațiilor verbale.Scop: Verificăm dacă modelul este robust la cele mai comune greșeli în limba română.C. Level 3: Authentic Gold Standard (Contribuția Manuală - "The Real Deal")Sursă: Colectăm 250 de propoziții "din sălbăticie" (comentarii Facebook/Reddit, articole de presă cu greșeli, texte studențești).Efort: Trebuie corectate manual de cel puțin 2 persoane (inter-annotator agreement).Valoare: Aceasta este secțiunea care "vinde" lucrarea la CIKM. Arată că sistemul funcționează pe date reale, nu doar pe scripturi.2. Noutatea Metodologică: Reranker-ul ConservatorDacă trimiți lucrarea doar cu "Detecție + T5", recenzorii vor spune că e un pipeline standard. Noutatea ta trebuie să fie "Minimal Edit & Uncertainty-Aware Correction".Ce modificăm în cod (suggestion_ranker.py):În loc să alegi prima sugestie a modelului T5 (care uneori rescrie propoziția inutil), implementăm un scor de ranking care penalizează distanța de editare mare dacă încrederea este mică.$$Score(s) = \text{Confidence}_{det} \times P(s|T5) - \lambda \cdot \text{Levenshtein}(original, s)$$Dacă modelul vrea să schimbe 5 cuvinte pentru a repara o virgulă, $\lambda$ (penalizarea) va trage scorul în jos, preferând o variantă mai conservatoare.De ce e asta o noutate? Pentru că rezolvă problema "over-correction" (supra-corectare), o temă majoră în NLP în 2025-2026.

---

**Cursor**

Ținta mea: să înțeleg rapid ce ai în repo (pipeline, date, evaluare), apoi să-ți propun un plan de „CIKM-ready” cu (a) benchmark intern reproductibil și (b) noutatea „conservative reranker” implementată în cod + evaluare pentru over-correction. Încep prin a scana structura proiectului și fișierele cheie (`test.txt`, `corpus.txt`, `suggestion_ranker.py`, scripturi de training/eval).

Am găsit deja două versiuni de `suggestion_ranker.py` (în `src/` și în `unused/`). Următorul pas: fac inventarul repo-ului (fișiere de date, scripturi de evaluare/training) și citesc `src/suggestion_ranker.py` + `GEC/test.txt` ca să văd formatul perechilor și cum e ales „prima sugestie” acum.

Repo-ul e curat și are documentație (mai multe update-uri în `README.md`). Urmează să citesc `README.md`, `utils/flow.md`, fișierele `GEC/{train,dev,test}.txt` și `src/suggestion_ranker.py` ca să extrag exact formatul datelor, pipeline-ul curent și unde introducem reranker-ul + protocolul de benchmark.

`GEC/test.txt` e formatat ca perechi pe linii consecutive: linia impară pare „gold/correct”, linia pară e varianta greșită (input). Următorul pas: găsesc exact unde e apelat `rank_suggestions` și cum se combină detectorul cu T5, apoi adaptez `src/suggestion_ranker.py` la scorul tău (confidence-aware + penalizare Levenshtein) și pregătesc scripturile de benchmark Level 1/2/3.

Acum e clar unde intră reranker-ul: endpointul `/correct` folosește `generate_corrections(..., num_suggestions=5)` și apoi `src/suggestion_ranker.py` (acum doar BLEU față de original, deci favorizează „cât mai aproape”, dar fără să țină cont de probabilitatea T5 sau de detector). Urmează să: (1) fac `generate_corrections` să returneze și scoruri (log-prob), (2) implementez reranker-ul „confidence-aware + penalizare Levenshtein”, (3) conectez asta în `app.py` și adaug scripturile de benchmark/eval reproductibile.

Am implementat partea „metodologică” în cod: T5 returnează acum scoruri (avg log-prob), iar `rank_suggestions` suportă modul `conservative` cu \( \text{det\_conf} \cdot \log P - \lambda \cdot \text{Lev} \). Următorul pas: adaug un mini-framework de benchmark/evaluare (Level 1/2) ca să producem rapid tabele/figuri pentru paper, și rulez linter-ul pe fișierele atinse (`app.py`, `src/models.py`, `src/suggestion_ranker.py`).

## Ce am găsit în proiect (important pentru paper)

- **Dataset-ul existent (`GEC/train.txt`, `dev.txt`, `test.txt`)** e în format perechi pe 2 linii:  
  **linia 1 = target (corect)**, **linia 2 = input (greșit)**. Asta e perfect pentru un benchmark Level 1 reproducibil.
- **Reranking-ul curent** (`src/suggestion_ranker.py`) folosea practic **BLEU față de propoziția originală**, fără:
  - probabilitatea modelului T5 pentru fiecare candidat
  - „confidence” din detector
  - o penalizare explicită pentru supra-corectare

## Ce am implementat deja (noutatea metodologică „Conservative Reranker”)

Am introdus exact ideea ta, dar într-o formă robustă numeric (cu log-probabilități):

\[
Score(s)=\text{det\_conf}\cdot \log P_{T5}(s\mid x)\;-\;\lambda\cdot Lev(x,s)
\]

### Modificări făcute în cod
- **`src/models.py`**
  - am adăugat `generate_corrections_with_scores(...)` care întoarce sugestii de forma `{"text": ..., "avg_logprob": ...}` (aprox. logP per token din `generate(output_scores=True)`).
- **`src/suggestion_ranker.py`**
  - `rank_suggestions(..., metric="conservative", det_confidence=..., lambda_edit=...)`
  - `Lev` e calculat cu funcția ta existentă `src/correct_word/levenshtein.py` (include cost redus pentru diferențe doar de diacritice).
- **`app.py`**
  - endpointul **`/correct`**: generează 5 candidați cu scoruri și aplică reranker-ul conservator (în loc de BLEU).
  - endpointul **`/check`**: calculează `prob_incorrect` din softmax-ul detectorului și îl folosește ca `det_confidence` în reranker.

Fișiere schimbate/adiționate acum în workspace:
- `app.py`, `src/models.py`, `src/suggestion_ranker.py`
- `bench/README.md`, `bench/eval_gec.py`, `bench/build_level2_stress.py`

## Benchmark intern (GSCR-Benchmark-RO) — am pus fundația Level 1 & 2

Am adăugat un mini-pachet `bench/`:

- **Level 1 (In-distribution)**: rulezi direct pe `GEC/test.txt`
- **Level 2 (Stress: diacritice + agreement-like noise)**:
  - script generator: `bench/build_level2_stress.py`
  - produce un fișier în același format (target/input pe 2 linii)

- **Evaluare**: `bench/eval_gec.py` raportează:
  - `exact_match`
  - `BLEU`, `chrF` (sacrebleu)
  - **`overcorrection_rate`**: cât de des modifică sistemul exemple deja corecte (indicator cheie pentru „over-correction”)
  - `unchanged_rate_on_correct`

Instrucțiunile sunt în `bench/README.md`.

## Ce lipsește pentru „Full Paper CIKM 2026” (partea care vinde)
- **Level 3 (Authentic Gold Standard)**: încă nu există în repo (doar am documentat formatul recomandat).
  - Pentru paper: e ideal să ai **2 annotatori + adjudicare** și să raportezi **inter-annotator agreement** (ex. Cohen’s kappa pe etichete de „needs edit” și/sau agreement pe target).
- **Baselines și ablații** (ca să treacă de „pipeline standard”):
  - T5 top-1 (fără reranker)
  - reranker doar cu `-λ*Lev` (fără det_conf, fără logP)
  - reranker doar cu `logP` (fără Lev)
  - (opțional) detector off vs on
- **Metrica de supra-corectare** e acum prinsă în `overcorrection_rate`; asta e exact povestea „Minimal Edit & Uncertainty-Aware Correction”.

Dacă vrei, următorul pas concret e să facem un „runner” complet (un script care rulează detector+T5+reranker pe Level 1/2/3 și scrie automat un JSON cu rezultate + tabele pentru secțiunea de Experiments).

---

**User**

da, realizează următorul pas

---

**Cursor**

Perfect — trec acum la „runner”-ul complet de experimente: un script unic care rulează automat pe Level 1/2/3, compară ablațiile (top-1, conservative etc.) și exportă rezultate în JSON gata de pus în lucrare. După implementare, îți las și comanda exactă de rulare.

Am realizat următorul pas: ai acum un **runner complet de benchmark** pentru paper.

## Ce am adăugat

- `bench/run_benchmark.py`  
  Rulează end-to-end pe Level 1/2/3 și exportă un JSON cu rezultate pe toate strategiile.

- Update la `bench/README.md`  
  Include comanda unică de rulare + explicația strategiilor/ablațiilor.

## Ce face runner-ul

Pentru fiecare nivel de dataset:
- încarcă perechi `(input_greșit, target_corect)` din formatul vostru pe 2 linii
- rulează strategiile:
  - `t5_top1`
  - `bleu_ranker` (baseline vechi)
  - `conservative` (metoda propusă)
  - `conservative_no_detconf` (ablație)
  - `conservative_no_edit` (ablație)
- calculează metrici:
  - `exact_match`, `BLEU`, `chrF`
  - `overcorrection_rate`
  - `unchanged_rate_on_correct`
  - `changed_rate`
- scrie raport JSON (default `bench/results.json`)

Dacă `Level 3` lipsește, nu crapă: marchează `missing_dataset`.

## Comanda de rulat

```bash
python bench/run_benchmark.py \
  --model t5-grammar-finetuned \
  --level1 GEC/test.txt \
  --level2 bench/level2_stress.txt \
  --level3 bench/level3_authentic.txt \
  --detector-path src/detection/content/trained_model_V2_2 \
  --k 5 \
  --lambda-edit 0.35 \
  --out bench/results.json
```

## Notă

Nu am putut executa local scripturile (în sesiunea curentă `python` nu pornește din mediul shell), dar codul e integrat și fără erori de linter.  
Dacă vrei, în pasul următor îți fac și un script `bench/render_tables.py` care transformă `results.json` direct în tabele LaTeX/Markdown pentru secțiunea de Experiments.

---

**User**

da, realizează script-ul render_tables.py

---

**Cursor**

Fac acum `bench/render_tables.py` ca să transforme `bench/results.json` în tabele gata de paper (Markdown + LaTeX), inclusiv evidențierea best-score per metric pe fiecare nivel.

Am adăugat scriptul cerut: `bench/render_tables.py`.

## Ce face

- Citește raportul din `bench/results.json`
- Generează:
  - `bench/tables.md` (tabele Markdown)
  - `bench/tables.tex` (tabele LaTeX)
- Face highlight automat pentru **cel mai bun scor** pe fiecare metrică (per nivel).
- Gestionează și cazul în care un nivel lipseste (`missing_dataset`).

## Cum îl rulezi

```bash
python bench/render_tables.py --in bench/results.json --out-md bench/tables.md --out-tex bench/tables.tex
```

Am actualizat și `bench/README.md` cu această comandă.

Dacă vrei, în pasul următor pot să-l extind cu:
- ordonare custom a strategiilor (ex. `conservative` primul),
- medii macro peste niveluri,
- export CSV pentru figuri în Excel/Sheets.

---

**User**

Odată ce fac rost de level3_authentic.txt care ar trebui să fie rezultatele așteptate pentru publicarea articolului și cum ai realiza acest articol mai precis?

---

**Cursor**

Foarte bună întrebare — `level3_authentic.txt` este practic piesa care decide dacă lucrarea arată „academic solid” sau doar „engineering demo”.

## Ce rezultate ar trebui să urmărești pe `level3_authentic.txt`

Nu există praguri oficiale CIKM, dar realist pentru acceptare ai nevoie de **poveste convingătoare + câștig clar pe metrici relevante**:

- `conservative` să fie **best** pe Level 3 la:
  - `overcorrection_rate` (mai mic e mai bine, la tine e raportat ca rate de schimbări pe propoziții deja corecte)
  - `unchanged_rate_on_correct` (mai mare)
- să păstrezi performanță competitivă la:
  - `BLEU`, `chrF`, `exact_match` (să nu sacrifici masiv corectitudinea pentru conservatorism)
- ablațiile să confirme noutatea:
  - `conservative` > `conservative_no_detconf` (beneficiul incertitudinii detectorului)
  - `conservative` > `conservative_no_edit` (beneficiul penalizării edit distance)
  - `conservative` > `t5_top1` pe over-correction, cu scădere mică sau zero la BLEU/chrF

Țintă practică (orientativ):
- reducere relativă de **15-30%** la over-correction vs baseline puternic (`t5_top1`/`bleu_ranker`)
- menținere BLEU/chrF în interval **±1-2 puncte** față de cel mai bun baseline
- efect consistent pe Level 2 și mai ales Level 3 (nu doar in-distribution).

## Cum aș construi articolul (ca să „vândă” la CIKM)

## 1) Mesajul central (1 frază)
„Propunem un reranker uncertainty-aware, minimal-edit, care reduce supra-corectarea în GEC română pe date autentice.”

## 2) Structura paper-ului

- **Introduction**
  - problema: over-correction în GEC real-world
  - gap: pipeline-urile standard optimizează corectitudinea, dar nu conservatorismul
  - contribuții (3 bullets): benchmark 3-level, reranker nou, analiză ablații

- **Related Work**
  - GEC, reranking, confidence calibration, minimal-edit correction
  - pune accent pe ce lipsește: robust eval pe română autentică + control over-correction

- **Method**
  - pipeline detector + generator
  - formula de scor:
    \[
    Score(s)=\text{Conf}_{det}\cdot \log P(s|x)-\lambda\cdot Lev(x,s)
    \]
  - explici intuitiv fiecare termen și de ce combate over-editing

- **Benchmark (GSCR-Benchmark-RO)**
  - L1 in-distribution
  - L2 stress (diacritice/agreement)
  - L3 authentic (250, 2 annotatori, adjudicare)
  - protocol de annotation + quality control + agreement

- **Experimental Setup**
  - modele, hyperparametri, hardware, seeds
  - strategii comparate (baselines + ablații)
  - metrici (inclusiv overcorrection)

- **Results**
  - tabel principal pe L1/L2/L3
  - tabel ablații
  - 5-10 exemple calitative (în special cazuri de supra-corectare evitate)

- **Analysis**
  - trade-off conservatorism vs agresivitate
  - analiza pe tipuri de erori (diacritice, acord, punctuație)
  - sensitivity pe `lambda_edit`

- **Limitations & Ethics**
  - bias surse L3, domenii limitate, limba română only
  - riscuri de normalizare excesivă a stilului

- **Conclusion**
  - 2-3 idei clare + plan de extensie multilingvă

## 3) Ce mai trebuie să faci concret după ce ai L3

- rulezi:
  1) `run_benchmark.py` pentru toate nivelurile
  2) `render_tables.py` pentru tabele finale
- adaugi:
  - test de semnificație (bootstrap resampling pe BLEU/chrF și overcorrection)
  - grafic simplu `lambda_edit` vs (`BLEU`, `overcorrection_rate`)
- pregătești appendix:
  - guideline de annotare L3
  - distribuția surselor și tipurilor de erori

---

Dacă vrei, în următorul pas îți scriu direct un **draft de 1 pagină** pentru secțiunile `Introduction + Contributions + Method`, în stil apropiat de template CIKM (gata de pus în LaTeX).

---

**User**

da, poți detalia și implementările aduse, precum și comentarea întregului framework făcând apel și la aceste documentații.

<h1 align="center">Hi 👋, here we have the GSCR (Grammar and stylistic correction for Romanian) project</h1>
<h3 align="center">Developed this project via NLP optional!</h3>


## Table Of Content
* [Brief Description of the Assignment](#brief)
* [Main Objectives](#obj)
* [Technologies and Programming Languages](#tech)
* [Implementation Stages](#stages)

--------------------------------------------------------------------------------
<h3 align="left">Keywords:</h3>

NLP; Romanian language; grammar correction; spell checking; punctuation correction; stylistic improvement; preprocessing; RoBERT; MarianMT; PyTorch; Active Learning; pattern matching; tokenization; POS tagging; diacritic normalization; text editors integration; continuous optimization; user feedback; real-time correction; Apache Spark; Teprolin; spaCy; generative AI; linguistic analysis

<h1 id="brief" align="left">Brief Description of the Assignment:</h1>

This assignment is about creating a helpful NLP tool specifically designed for Romanian texts. The main purpose of this project is to automatically spot and correct common mistakes in grammar, spelling, punctuation, and even improve the style to make texts clearer and easier to read.
The solution is structured around several essential modules:

<h3 align="left">Preprocessing Module:</h3>

○ This first step cleans up the text by breaking it into meaningful units (tokenization), identifying parts of speech (like nouns, verbs, adjectives), fixing special characters, and checking diacritics.
    
<h3 align="left">Error Detection Module:</h3>

○ In this step, advanced NLP models such as RoBERT, specially trained on Romanian language datasets, are used to find grammatical and spelling errors.

○ Additionally, common mistakes are identified through pattern recognition (e.g. Levenstein) and a built-in dictionary-based spell checker.

<h3 align="left">Automatic Correction Module:</h3>

○ This module uses powerful language models such as MarianMT to automatically suggest corrections. It offers several correction options and ranks them to suggest the best one.

<h3 align="left">Continuous Optimization Through Active Learning and Integrating with Writing Tools:</h3>

○ The system continually learns and improves through user interaction. Feedback from users helps fine-tune the model and enhance its accuracy over time.The final application should provide integration with standard writing tools.


Ultimately, the project aims to help Romanian speakers write better, clearer texts with fewer mistakes, continuously optimizing through active learning


* [Table Of Content](#table-of-content)

<h1 id="obj" align="left">Brief Description of the Assignment:</h1>

The main objectives to achieve through this assignment include:

  ● Developing a robust NLP model capable of accurately detecting grammatical, orthographic, punctuation, and stylistic errors in Romanian texts. </br>
  ● Implementing a preprocessing pipeline for Romanian text. </br>
  ● Creating an error detection module using advanced Romanian language NLP models, enhanced by pattern matching and dictionary-based spell checking. </br>
  ● Developing an automated correction mechanism that generates and ranks multiple correction suggestions, ensuring clarity and preservation of the original meaning. </br>
  ● Establishing a user-driven continuous optimization process, through active learning and user feedback. </br>
  ● Achieving integration of this NLP tool into commonly used text-editing platforms to enhance accessibility and usability for diverse use cases. </br>

* [Table Of Content](#table-of-content)

<h1 id="tech" align="left">Technologies and Programming Languages:</h1>

● Programming Language: Python </br>
● Frameworks and Libraries: PyTorch, Teprolin, spaCy, MarianMT, RoBERT </br>
● Additional Tools: pattern-matching algorithms, dictionary-based spell-checking, Active Learning frameworks </br>

<h1 id="stages" align="left">Implementation Stages:</h1>

Phase 1 (Research and Setup): Literature review, dataset collection, setting up development environment. </br>
Phase 2 (Preprocessing Module Implementation): Implementing tokenization, POS tagging, diacritic normalization using Teprolin, spaCy. </br>
Phase 3 (Error Detection Module Development): Fine-tuning RoBERT and integrating dictionary-based and pattern-matching error detection. </br>
Phase 4 (Automatic Correction Module): Implementing and fine-tuning MarianMT for generating and ranking correction suggestions. </br>
Phase 5 (Continuous Optimization and Testing): User feedback integration, active learning framework implementation, iterative optimization. </br>
Phase 6 (Deployment): Integration with text editors and final system evaluation. </br>
  
<h3 align="left">The logic behind the code:</h3>

  Soon

* [Table Of Content](#table-of-content)

---

<h3 id="installation" align="left">Installation:</h3>

  # GSCR — Grammar & Stylistic Correction for Romanian
    _A Chrome Extension + Python NLP backend_
    
    ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
    [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)
    
    GSCR brings real-time Romanian grammar and style checking to any web page.  
    It combines a lightweight Chrome extension with a Flask-based NLP service
    powered by **ULMFiT** (error detection) and a fine-tuned **T5 Transformer**
    (error correction).
    
    ---
    
    ## Features
    - Two-stage GEC pipeline → fast + accurate
    - On-device privacy: text never leaves your machine
    - Optional GPU acceleration (`utils/cuda.py`)
    - Dockerised **Teprolin** for tokenisation & morpho-syntactic tags
    - SQLite logging of accepted fixes for active learning
    
    ---
    
    ## Quick Start
    
    ### 1· Clone + Install Python deps
    ```bash
    git clone https://github.com/AndromedaOMA/GSCR---Project.git
    cd GSCR---Project
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download ro_core_news_sm

    ### Install safetensors:
    1. Download the .zip files from here: https://we.tl/t-AxtK3ket8f (link available for 3 days - contact one of the members if expired)
    2. Unzip the files into the parent folder of the project

    ### 2· (optional) Enable CUDA
    See utils/cuda.py for manual install steps matching your GPU/CUDA version.

    ### 3. Run Teprolin Docker
    docker pull raduion/teprolin:1.1
    docker run -d -p 5000:5000 --name teprolin raduion/teprolin:1.1

    ### 4. Start backend
    python app.py

    ### 5. Load Extension (example for Google Chrome)
    1. Go to `chrome://extensions/`
    2. Enable **Developer mode**
    3. Load unpacked → select the extension/ folder
    4. After loading, open **DevTools**:
    5. Right-click on the page and choose **Inspect** or press `Ctrl+Shift+I`.
    6. In the **Console**, you may see the error: ```Not allowed to load local resource```
    7. In the browser address bar, go to: `https://localhost:<PORT>`
    8. On the **Your connection is not private** warning page, click **Advanced**, then **Proceed to localhost (unsafe)**.
    9. Reload the page; the extension should now function correctly.

    ### 6. Extension in Google Docs: Google Apps Script and ngrok Configuration for Backend APIs
    1. Open Google Docs and create a new document. In the menu bar, click Extensions and then Apps Script. 
    2. An editor will be opened in a new tab. In the Code.gs file, insert the code from the Code.gs file provided in the Github repository. Additionally, create the SuggestionsDialog.html and SynonymsDialog.html files, and copy-paste the content provided in those files in the Github repository.
    3. Run the backend server that will receive the requests for processing text and offering suggestions.
    4. Install ngrok, authenticate with the token provided in your ngrok user profile, and type ngrok http <server_port> (Replace <server_port> with the port your local server is running on).
    5. Update the server url from Code.gs with the one provided by ngrok in the terminal ( something like: var urlDeBaza="https://d9ed-2a02-2f00-c307-1800-55f1-5e48-d5b1-3781.ngrok-free.app").
    6. When you run the Apps Script for the first time, you will be prompted to grant the necessary permissions. Accept all required scopes. Then, go to Deployments -> Test deployments, set the project type to Editor Add-on, and select a Google Docs document to test the extension. Save and install the deployment.
    7. Open the Google Docs file where the extension was installed. You will now see a new menu added by the extension. While testing, ensure that the ngrok terminal session is active—this is required for the Google Apps Script to reach your local backend. When a menu item is triggered, the backend will receive the request, process the text, and return a response. This response will be used to either modify the document or display suggestions in a custom dialog box.

#Model Tensors
*.safetensors files (T5 corrector + ULMFiT detector) are not part of the repo.
Request them from the maintainers, place under models/, and restart app.py.

#Inspect the Database

Download DB Browser for SQLite → https://sqlitebrowser.org/dl/
Open gscr.db to explore logged feedback.

#Contributing
Fork → git checkout -b feat/awesome

Run pre-commit install (lint hooks)

PR against main with a clear description.



* [Table Of Content](#table-of-content)

---

**NOTE**: This project represents the final project supported and realized within the NLP laboratories of the Faculty of Computer Science of the Alexandru Ioan Cuza University in Iași, Romania.

* [Table Of Content](#table-of-content)

---
- ⚡ Fun fact: **Through this project I developed better the subtle concepts of NLP concepts!**


---

**Cursor**

Da — fac acum o documentație tehnică completă (în română) care explică arhitectura actuală, modificările implementate pentru CIKM (reranker + benchmark + ablații), cum se leagă de README-ul existent și cum se transformă direct în secțiuni de articol. O pun într-un fișier nou în repo ca să o poți folosi direct.

Am făcut exact ce ai cerut: am documentat tehnic și comentat framework-ul end-to-end, legând explicit noile implementări de documentația existentă.

## Ce am adăugat

- Document nou: `docs/CIKM_framework_ro.md`
  - arhitectura completă GSCR (preprocess, detect, correct, rank, feedback)
  - modificările implementate pentru CIKM (reranker conservator + confidence-aware)
  - benchmark-ul pe 3 niveluri (L1/L2/L3)
  - ablațiile și interpretarea rezultatelor
  - comenzi reproducibile pentru run + tabele
  - ghid concret de mapare în secțiuni de articol

- Update în `README.md`
  - la secțiunea “The logic behind the code” am înlocuit `Soon` cu trimitere la `docs/CIKM_framework_ro.md`.

## Ce obții practic

Acum ai un document „de punte” între:
- descrierea generală din README,
- implementările reale din cod,
- protocolul experimental,
- și structura de paper CIKM.

Dacă vrei, în pasul următor îți fac și varianta **în engleză academică** (`docs/CIKM_framework_en.md`) direct în stil de secțiuni pentru manuscris (aproape copy-paste în LaTeX).

---

**User**

python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 bench/level2_stress.txt --level3 bench/level3_authentic.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
Traceback (most recent call last):
  File "D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project\bench\run_benchmark.py", line 11, in <module>
    from src.detection.detect import HFWrapperULMFiT
ModuleNotFoundError: No module named 'src'


---

**Cursor**

Eroarea e clară: scriptul pornește din `bench/` și Python nu vede automat rădăcina repo-ului pentru importul `src.*`. Fac acum un fix robust în scripturile din `bench` ca să-și adauge singure root-ul proiectului în `sys.path`.

Corect — am reparat problema de import.

## Ce am modificat

- `bench/run_benchmark.py`
- `bench/eval_gec.py`

Am adăugat bootstrap de path la începutul fișierelor, ca să poți rula direct:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Asta rezolvă `ModuleNotFoundError: No module named 'src'` când rulezi `python bench/...`.

## Rulează din nou comanda

```bash
python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 bench/level2_stress.txt --level3 bench/level3_authentic.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
```

Dacă apare următoarea eroare, trimite traceback-ul și o fixăm imediat.

---

**User**

python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 bench/level2_stress.txt --level3 bench/level3_authentic.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 257/257 [00:00<00:00, 1865.15it/s]
Traceback (most recent call last):
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 422, in cached_files
    hf_hub_download(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 84, in _inner_fn
    validate_repo_id(arg_value)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 132, in validate_repo_id
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 684, in from_pretrained
    config = AutoConfig.from_pretrained(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\models\auto\configuration_auto.py", line 1483, in from_pretrained
    config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 670, in get_config_dict     
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 725, in _get_config_dict    
    resolved_config_file = cached_file(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 473, in cached_files
    raise OSError(f"{e}") from e
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 422, in cached_files
    hf_hub_download(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 84, in _inner_fn
    validate_repo_id(arg_value)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 132, in validate_repo_id   
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project\bench\run_benchmark.py", line 243, in <module>
    main()
  File "D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project\bench\run_benchmark.py", line 196, in main
    detector = DetectorConfidence(args.detector_path, device=device)
  File "D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project\bench\run_benchmark.py", line 43, in __init__
    self.tokenizer = AutoTokenizer.from_pretrained(detector_path)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 688, in from_pretrained
    config = PreTrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 629, in from_pretrained     
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 670, in get_config_dict     
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 725, in _get_config_dict    
    resolved_config_file = cached_file(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 473, in cached_files
    raise OSError(f"{e}") from e
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.


---

**Cursor**

Perfect, acum problema e la `--detector-path`: Transformers îl interpretează ca repo HF în loc de director local. Fac un fix în `run_benchmark.py` ca să rezolve automat path-ul relativ la rădăcina proiectului și să folosească calea absolută locală dacă există.

Fix aplicat. Cauza era exact asta: `src/detection/content/trained_model_V2_2` era tratat ca nume de repo Hugging Face, nu ca folder local.

## Ce am schimbat

În `bench/run_benchmark.py`, clasa `DetectorConfidence`:
- rezolvă `--detector-path` relativ la rădăcina proiectului (`PROJECT_ROOT`)
- dacă path-ul există local, folosește **cale absolută** (`resolve()`), astfel `from_pretrained(...)` îl încarcă local, nu din Hub
- dacă nu există local, păstrează fallback-ul (poate fi repo HF valid)

## Reîncearcă aceeași comandă

```bash
python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 bench/level2_stress.txt --level3 bench/level3_authentic.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
```

Dacă mai apare ceva, dă-mi traceback-ul și îl rezolvăm imediat.

---

**User**

(gscr_env) PS D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project> python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 bench/level2_stress.txt --level3 bench/level3_authentic.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 257/257 [00:00<00:00, 3418.60it/s]
Traceback (most recent call last):
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 422, in cached_files
    hf_hub_download(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 84, in _inner_fn
    validate_repo_id(arg_value)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 132, in validate_repo_id
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 684, in from_pretrained
    config = AutoConfig.from_pretrained(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\models\auto\configuration_auto.py", line 1483, in from_pretrained
    config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 670, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 725, in _get_config_dict    
    resolved_config_file = cached_file(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 473, in cached_files
    raise OSError(f"{e}") from e
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 422, in cached_files
    hf_hub_download(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 84, in _inner_fn
    validate_repo_id(arg_value)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\huggingface_hub\utils\_validators.py", line 132, in validate_repo_id   
    raise HFValidationError(
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project\bench\run_benchmark.py", line 251, in <module>
    main()
  File "D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project\bench\run_benchmark.py", line 204, in main
    detector = DetectorConfidence(args.detector_path, device=device)
  File "D:\Windows_Folders\MASTER\An I\Sem II\GSCR---Project\bench\run_benchmark.py", line 51, in __init__
    self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 688, in from_pretrained
    config = PreTrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 629, in from_pretrained     
    config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 670, in get_config_dict     
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\configuration_utils.py", line 725, in _get_config_dict    
    resolved_config_file = cached_file(
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 278, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
  File "D:\Windows_Folders\FACULTATE\Anaconda\envs\gscr_env\lib\site-packages\transformers\utils\hub.py", line 473, in cached_files
    raise OSError(f"{e}") from e
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'src/detection/content/trained_model_V2_2'. Use `repo_type` argument if needed.


---

**Cursor**

Mersi pentru traceback, e util: înseamnă că scriptul încă nu găsește local checkpoint-ul detectorului (deci cade pe fallback-ul HF repo string). Verific exact cum e calea pe disk și fac patch ca să accepte și structura reală din proiect (inclusiv fallback automat la calea implicită din `app.py`).

Perfect, am identificat cauza reală din logul tău: **folderul detectorului nu există în repo-ul curent** (am verificat `src/detection/content/trained_model_V2_2` și e `False`).

Am făcut `run_benchmark.py` mai robust:

- dacă `--detector-path` nu există local, scriptul:
  - nu mai crapă
  - afișează warning clar
  - continuă cu `det_confidence=1.0` (fallback)
- caută detectorul în mai multe locații candidate (inclusiv calea hardcodata din `app.py`).

## Ce înseamnă pentru tine acum

Comanda ta ar trebui să ruleze până la capăt chiar fără checkpoint-ul detectorului local.

```bash
python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 bench/level2_stress.txt --level3 bench/level3_authentic.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
```

## Important pentru paper

- dacă rulezi fără detector, ablația `conservative` vs `conservative_no_detconf` va fi aproape identică (normal).
- pentru contribuția „uncertainty-aware”, ai nevoie de checkpoint-ul detectorului disponibil local.

Dacă vrei, următorul pas îl fac eu: adaug în `results.json` un câmp explicit `detector_enabled: true/false`, ca să nu existe ambiguitate când citești tabelele.

---

**User**

nu am checkpoint-ul detectorului... ajută-mă să antrenez modelul pentru a realiza un alt checkpoint

---

**Cursor**

Perfect — atunci facem un flux complet ca să obții rapid un nou checkpoint pentru detector (de la date până la `from_pretrained`). Încep prin a verifica scriptul vostru de training pentru detector și îl ajustez să salveze checkpoint-ul exact în formatul așteptat de `app.py`/`run_benchmark.py`.

Am identificat cauza principală: scriptul actual de train nu salvează explicit checkpoint-ul în formatul/locația folosită de runtime (`src/detection/content/trained_model_V2_2`) și are căi relative fragile. Îți fac acum un script dedicat de antrenare detector + salvare `save_pretrained` compatibil direct cu `app.py` și benchmark.

Super, am pregătit exact ce-ți trebuie ca să refaci checkpoint-ul detectorului.

## Ce am implementat

- Script nou: `src/detection/train_detector.py`
  - antrenează detectorul pe `GEC/train.txt`, validează pe `GEC/dev.txt`, testează pe `GEC/test.txt`
  - salvează checkpoint-ul în format HuggingFace (`save_pretrained`) + tokenizer
  - output default: `src/detection/content/trained_model_V2_2` (compatibil direct cu `app.py` și benchmark)
  - salvează și metricile test în `metrics_test.json`

- Am actualizat și documentația: `docs/CIKM_framework_ro.md`
  - am adăugat pasul explicit de antrenare detector înainte de benchmark.

## Comanda pe care să o rulezi acum

```bash
python src/detection/train_detector.py --train GEC/train.txt --dev GEC/dev.txt --test GEC/test.txt --out src/detection/content/trained_model_V2_2
```

După ce termină, rulezi din nou benchmark-ul:

```bash
python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 bench/level2_stress.txt --level3 bench/level3_authentic.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
```

Dacă vrei, după ce rulezi training-ul, îmi trimiți output-ul cu metrici și îți spun dacă checkpoint-ul e „suficient de bun” pentru experimentele de paper sau trebuie încă un fine-tune (epoch/lr/batch).

