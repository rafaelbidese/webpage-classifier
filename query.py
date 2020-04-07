from bs4 import BeautifulSoup  # utilizado para extrair o texto de um html
import time
import re
import requests
from sklearn.externals import joblib
import pandas as pd
import numpy as np

def extract_text(html_text):
    """
    Extracts set of unique words on document
    :return: set of words(string)
    """
    remove = "%#&,!:.@()-{}/"  # caracteres especiais para serem removidos

    soup = BeautifulSoup(html_text, "html.parser")
    html_content = soup.get_text()  # extrai o texto do arquivo .html

    spaces = " " * remove.__len__()  # utilizado para remover os caracteres
    html_content = html_content.translate(str.maketrans(remove, spaces))  # remove os caracteres desejados
    html_line = " ".join(html_content.splitlines()[6:])  # remove um cabeçalho que tem nos arquivos antes do <html>
    html_line = re.sub("\d+", "", html_line)


    return html_line

vectorizer = joblib.load('vectorizer.pkl')
model =  joblib.load('RandomForestClassifier.pkl')
target_names = joblib.load('target_names.pkl')

address = "https://www.ece.cornell.edu/ece/people/profile.cfm?netid=ja679"
result = requests.get(address, headers={'User-Agent': 'Mozilla/5.0'})

if result.status_code == 200:
	query_html = extract_text(result.text)
	query_tokens = vectorizer.transform([query_html])
	proba = model.predict_proba(query_tokens)[0]
	query_result = np.column_stack((target_names, proba))
	df = pd.DataFrame(query_result, columns=['label','proba']).sort_values(by='proba', ascending=False)
	print(df)