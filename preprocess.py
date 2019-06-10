import numpy as np
import string
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer


def pre_process(doc, c_sinonimo, metodo):
################################################
# FUNÇÃO PARA PRE-PROCESSAMENTO
# - word_tokenize = segmenta as sentenças em palavras
# - POS-TAG = anotar palavras de acordo com sua classificação gramatical ou de acordo com sua função no discurso
# - non_stopwords = remove as palavras mais comuns
# - lower = minimiza todas as palavras
# - non_punctuation = remove a pontuação
# - são padronizadas as classificações gramaticais devido a possuir grande variabilidade e por necessidar utiliza-las na "lemmatização"
# - "lemmatização" = deixa na palavra a parte que não varia conforme senteças, o radical. (e.g: bigger -> big)
# - synset.hypernyms (hyponyms, member_holonyms, lemma_names) = encontra as palavras sinonimas com base no wordnet
# - por ultimo é estruturado o dado conforme o método escolhido:
# metodo 0 -> Retorna dois vetores, um de palavras e outro de postag
# [NAME1 NAME2 NAME3][POS1 POS2 POS3] - PARA UTILIZAR APENAS COM C-BOW N-GRAM >= 1
# metodo 1 -> retorna 1 vetor com nomes e pos tag juntos
# [NAME1_POS1 NAME2_POS2 NAME3_POS3] - PARA UTILIZAR APENAS COM C-BOW N-GRAM >= 1
# c_sinonimo - com sinonimo é igual a 1 e sem é igual a 0
################################################

    dado_preprocessado = []
    sinonimos2_temp = []
    stopwords_list = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for d in doc:
        words = word_tokenize(d) # separa sentenças em lista de tuplas onde cada tupla é uma palavra
        pos_tags = pos_tag(words) #add pos tag
        non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list] #remove stopwords
        non_stopwords = [w for w in non_stopwords if len(w[0])>1]
        non_punctuation = [] 
        # remover pontuação e espaços
        for w in non_stopwords:
            replace_punctuation = str.maketrans(string.punctuation,' '*len(string.punctuation))
            wtemp = w[0].lower().translate(replace_punctuation)
            w2 = (wtemp.replace(" ",""), w[1])
            if not w[0][0] in string.punctuation:
                non_punctuation.append(w2) 
        non_punctuation = [w for w in non_punctuation if len(w[0])>1]
        lemmas = []
        pos2=[]
        lemmas2 = []
        doc_trans = []
        # Arruma pos tags para utilizar lemmatização e encontrar palavras similares dado postag
        for w in non_punctuation: 
            if w[1].startswith('J'):
                pos = wordnet.ADJ
            elif w[1].startswith('V'):
                pos = wordnet.VERB
            elif w[1].startswith('N'):
                pos = wordnet.NOUN
            elif w[1].startswith('R'):
                pos = wordnet.ADV
            else:
                pos = wordnet.NOUN
            lemma_temp = lemmatizer.lemmatize(w[0], pos) #aplica a lemmatização antes de procurar sinonimos
            pos2.append(pos) # usado no metodo 0
            lemmas2.append(lemma_temp) # usado no metodo 0
            lemmas.append((lemma_temp,pos)) # usado no metodo 1
        sinonimos2_temp = list(dict.fromkeys(lemmas))
        sinonimos2 = []
        if c_sinonimo == 1: #inclui os sinonimos 
            for p in range(len(sinonimos2_temp)):
                sinonimos = wordnet.synsets(sinonimos2_temp[p][0], pos=sinonimos2_temp[p][1])
                temp_list = []
                temp_list2 = []
                for synset in sinonimos:
                    temp_list.append(synset.hypernyms())
                    temp_list.append(synset.hyponyms())
                    temp_list.append(synset.member_holonyms())
                    temp_list = list(np.hstack(temp_list))
                    temp_list = list(dict.fromkeys(temp_list))
                if len(temp_list)>1:
                    for synset2 in temp_list:
                        temp_list2.append(synset2.lemma_names())
                        temp_list2 = list(np.hstack(temp_list2))
                        temp_list2 = list(dict.fromkeys(temp_list2))
                if (len(temp_list2)>0):
                    for w2 in (temp_list2):
                        if metodo == 0:
                            w3 = (w2.replace("_"," "))
                        else:
                            w3 = (w2, sinonimos2_temp[p][1])
                        sinonimos2.append(w3)
            sinonimos2 = list(dict.fromkeys(sinonimos2))
            sinonimos2.extend(lemmas)
        else:
            if metodo == 0:
                sinonimos2.extend(lemmas2)
            else:
                sinonimos2.extend(lemmas)
        #metodos possuem diferentes tipos de retorno de função
        if metodo == 0:
            doc_trans = [[" ".join(["".join(w) for w in sinonimos2])]]
            doc_trans.append(pos2)
        else:
            doc_trans = [" ".join(["_".join(w) for w in sinonimos2])]
        dado_preprocessado.append(doc_trans)
    return dado_preprocessado

def bow_transform(sentences, max_words):
    tokenizer = Tokenizer(num_words = max_words)
    #filters=None #não pode filtrar "_"
    tokenizer.fit_on_texts(sentences)
    dataset = tokenizer.texts_to_matrix(sentences)
    return tokenizer, dataset

# Proximidade semântica com janela (window=3) igual a 3 e tamanho de neurônios(size) igual a 100 (size=100)
# utilizando a técnica de cbow (sg=2)
# (min_count=2, sg=2, window=3, size=100) 
# window e size podem ser passados como hiperparametros
#sg - algoritmo de treino: 1 para skip-gram; 2 ou qualquer outro valor é CBOW.
class Word2VecTransformer(object):
    ALGO_SKIP_GRAM=1
    ALGO_CBOW=2    
    
    def __init__(self, algo=1):
        self.algo = algo
    
    def fit(self, X, y=None):     
        X = [nltk.word_tokenize(x) for x in X]
        self.word2vec = Word2Vec(X, min_count=2, sg=self.algo, window=5, size=200)
        # Pegamos a dimensão da primeira palavra, para saber quantas dimensões estamos trabalhando,
        # assim podemos ajustar nos casos em que aparecerem palavras que não existirem no vocabulário.
        first_word = next(iter(self.word2vec.wv.vocab.keys()))
        self.num_dim = len(self.word2vec[first_word])       
        return self
    
    def transform(self, X, Y=None):        
        X = [nltk.word_tokenize(x) for x in X]
        
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.num_dim)], axis=0) 
                         for words in X])
    def get_params(self, deep=True):
        return {}
    

