Installation:
```bash
pip install -r requirments.txt
python -m spacy download ro_core_news_sm
```

In order to use cuda, see `utils/cuda.py`, a manual installation is required depending on the GPU version.

Install Docker Desktop, run:
```
docker pull raduion/teprolin:1.1
docker run -p 5000:5000 raduion/teprolin:1.1
```

Optionally, to view the database:
https://sqlitebrowser.org/dl/ - default installation

In the end, run using ```python app.py```

Testing example in Postman: https://imgur.com/a/kfazGcN