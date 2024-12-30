# Universal Document Linking for Enhancing the Zero-Shot Retrieval 

   ```
   conda create -n udl python=3.10
   conda activate udl
   pip install -r requirements.txt
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_scibert-0.5.1.tar.gz
   python -m spacy download en_core_web_trf
   python main.py
   ```