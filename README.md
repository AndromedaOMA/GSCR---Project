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
