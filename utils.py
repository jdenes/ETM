import torch 
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2dense
from gensim.models import LdaModel, TfidfModel, word2vec

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1: 
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1: 
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, vocab):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))


def get_classif_perf(theta, tokens, labels, embeds, methods=['theta', 'lda', 's-bert', 'tfidf']):
    # print('Checking inputs dim for classif:', len(theta), len(labels))
    import pandas as pd
    perf = []
    
    if 'theta' in methods:
        X = theta
        perf.append(train_predict(X, labels))
        
    if 'lda' in methods:
        corpus = tokens.tolist()
        corpus = [[str(w) for w in d[0]] for d in corpus]
        dictionary = Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(x) for x in corpus]
        mod = LdaModel(bow_corpus, num_topics=theta.shape[1])
        transcorp = mod[bow_corpus]
        X = transcorp2matrix(transcorp, bow_corpus, theta.shape[1])
        perf.append(train_predict(X, labels))      
        
    if 's-bert' in methods:
        from sklearn.decomposition import PCA
        X = PCA(n_components=theta.shape[1]).fit_transform(embeds)
        perf.append(train_predict(X, labels))

    if 'tfidf' in methods:
        corpus = tokens.tolist()
        corpus = [[str(w) for w in d[0]] for d in corpus]
        dictionary = Dictionary(corpus)
        dictionary.filter_extremes(keep_n=theta.shape[1])
        bow_corpus = [dictionary.doc2bow(x) for x in corpus]
        mod = TfidfModel(bow_corpus, dictionary=dictionary)
        corpus_tfidf = mod[bow_corpus]
        X = corpus2dense(corpus_tfidf, num_terms=theta.shape[1]).T
        perf.append(train_predict(X, labels))
    
    perf = pd.DataFrame(perf, index=methods)
    print('Model performances on classification is:\n{}'.format(perf))


def train_predict(X, labels):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    import tensorflow as tf
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=123)
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache().shuffle(1000).batch(1000).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1000).repeat()
    model = Sequential([
                    Dense(248, name='first', activation='relu', input_shape=(X_test.shape[1],)),
                    Dense(248, name='hidden', activation='relu'),
                    Dense(len(np.unique(y_train)), name='output', activation='softmax') ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint.hdf5', monitor='val_loss', save_best_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit(train_data, steps_per_epoch=1000, epochs=100, validation_steps=100, validation_data=val_data, callbacks=[checkpoint, earlystop])
    perf = evaluate_model(model, X_train, X_test, y_train, y_test)
    return perf


def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred, normalize=True)
    pr = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return {'Accuracy': acc, 'Precision': pr, 'Recall': rec, 'F1-score':f1}
    

def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.index(word)
    # print('vectors: ', vectors.shape)
    query = vectors[index]
    # print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:12]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors

# From a sparse transformed corpus of gensim, i.e. [(0, 12), (1, 15)], return matrix format: [12, 15].
def transcorp2matrix(transcorp, bow_corpus, vector_size):
    x = np.zeros((len(bow_corpus), vector_size))
    for i, doc in enumerate(transcorp):
        for wpair in doc:
            x[i][wpair[0]] = wpair[1]
    return x