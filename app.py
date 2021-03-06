import re
from collections import defaultdict, Counter
from functools import reduce
from operator import add

import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn import cluster
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

import datasets
import interactive
import preprocessing

app = Flask(__name__)

# configs
# algo
k = 5  # start with assuming there are this many clusters
max_features = 1000
docs_limit = 5000
default_percentage = 0.1

# cosmetic
color_alpha = 0.5
colors_base = [
    'rgba(47,126,216,{alpha})',
    'rgba(13,35,58,{alpha})',
    'rgba(139,188,33,{alpha})',
    'rgb(145,0,0,{alpha})',
    'rgba(26,173,206,{alpha})',
    'rgba(73,41,112,{alpha})',
    'rgba(242,143,67,{alpha})',
    'rgba(119,161,229,{alpha})',
    'rgba(196,37,37,{alpha})',
    'rgba(166,201,106,{alpha})'
]
text_top_n_chars = 500


# util
def _get_number_of_docs(rows: int, pct: float) -> int:
    n = int(rows * pct)
    return max(1, min(rows, n))


# algo
corpus = datasets.get_bbc()
pre_processor = preprocessing.NLPProcessor(max_features=max_features)
bbc_vectorized_features_bound = pre_processor.fit_transform(corpus)
slice_value = _get_number_of_docs(bbc_vectorized_features_bound.shape[0], default_percentage)
data = bbc_vectorized_features_bound[:slice_value].todense()
terms = np.array(pre_processor.vec.get_feature_names()).reshape((1, max_features))

# cosmetic
colors = [x.format(alpha=color_alpha) for x in colors_base]

# text processing
regex = re.compile('[^a-zA-Z\s]')
stop_words_ad_hoc = {'said'}
stop_words = set(ENGLISH_STOP_WORDS).union(stop_words_ad_hoc)


def get_pca_for_highcharts(clusters_docs, pca):
    cluster_data = defaultdict(list)
    for i, docs in enumerate(clusters_docs):
        for doc in docs:
            pca_doc = pca[doc - 1]
            cluster_data[i].append({'x': pca_doc[0], 'y': pca_doc[1], 'name': corpus[doc - 1][:text_top_n_chars]})

    return [
        {
            'name': f'Cluster {i+1}',
            'color': colors[i % len(colors)],
            'data': cluster_data[i],
        }
        for i in range(len(cluster_data))
    ]


def set_data_set(dataset_name: str, percentage: float) -> None:
    global data, terms, corpus
    corpus_local = None
    if dataset_name == "BBC":
        corpus_local = datasets.get_bbc()
    elif dataset_name == "20 News Groups":
        dataset = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'))
        corpus_local = dataset.data
    elif dataset_name == "All the news":
        corpus_local = datasets.get_all_the_news()

    vectorized_features_bound = pre_processor.fit_transform(corpus_local)

    # calculate percentage to include
    n = _get_number_of_docs(vectorized_features_bound.shape[0], percentage)
    data = vectorized_features_bound[:n].todense()
    terms = np.array(pre_processor.vec.get_feature_names()).reshape((1, max_features))
    corpus = corpus_local


def get_algorithm(algorithm_name: str, clusters: int) -> cluster:
    if algorithm_name == "Birch":
        return cluster.Birch(n_clusters=clusters)
    elif algorithm_name == "Spectral Clustering":
        return cluster.SpectralClustering(n_clusters=clusters)
    elif algorithm_name == 'Affinity Propagation':
        return cluster.AffinityPropagation()
    else:
        raise NotImplementedError(f'algorithm: {algorithm_name} not implemented')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load', methods=['POST'])
def load():
    user_input_json = request.json
    algorithms = ["iKMeans", "DBSCAN", "birch", "means", "spectral", "affinity"]
    dataset_names = ["BBC", "20 News Groups", "All the news"]

    percentage = user_input_json["percentage"]
    current_dataset = user_input_json["dataset"]
    if current_dataset is None:
        current_dataset = dataset_names[0]
    set_data_set(current_dataset, percentage)

    clusters = user_input_json["numOfClusters"]
    if clusters == "" or clusters is None:
        clusters = 5
    else:
        clusters = int(clusters)

    current_algorithm = user_input_json["algorithm"]
    if current_algorithm is None:
        current_algorithm = algorithms[0]
    if current_algorithm == "iKMeans":
        return get_clusters(clusters, [])

    pca = IncrementalPCA(n_components=2).fit_transform(data)
    pca = StandardScaler().fit_transform(pca)
    # key_terms = [list(x) for x in key_terms]

    algorithm = get_algorithm(current_algorithm, clusters)
    algorithm.fit(data)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(data)
    silhouette_avg = (50 * silhouette_score(data, y_pred, 'cosine')) + 50
    print(silhouette_avg)
    sample_silhouette_values = silhouette_samples(data, y_pred, 'cosine')

    clusters_dict = defaultdict(list)
    scores = dict()
    for i, label in enumerate(y_pred):
        clusters_dict[label].append(i + 1)
        ith_cluster_silhouette_values = sample_silhouette_values[y_pred == label]
        avg = np.mean(ith_cluster_silhouette_values)
        scores[str(label)] = interactive.scale_score(avg)
    clusters_docs = [clusters_dict[i] for i in range(len(clusters_dict))]

    # print(f'# of docs in each cluster: {list(map(len, clusters_docs))}')
    # print(f'sum of docs in each cluster: {sum(map(len, clusters_docs))}')
    # print(f'Cluster doc: {len(clusters_docs)}')
    #
    # print(clusters_docs)
    cluster_key_terms = [
        [x[0] for x in reduce(add, (Counter((w for w in regex.sub('', corpus[i - 1]).split()
                                             if w.lower() not in stop_words and len(w) > 3))
                                    for i in c)).most_common(50)]
        for c in clusters_docs
    ]

    return send_data(cluster_key_terms, silhouette_avg, scores, clusters_docs, pca)


def send_data(cluster_key_terms, silhouette_avg, scores, clusters_docs, pca):
    return jsonify({
        'cluster_key_terms': cluster_key_terms,
        # 'key_terms': key_terms,
        'silhouette': silhouette_avg,
        'pca': get_pca_for_highcharts(clusters_docs, pca),
        'scores': scores
    })


def get_clusters(no_clusters, user_input):
    user_feedback = -1 if len(user_input) == 0 else +1
    clusters_docs, cluster_key_terms, key_terms, silhouette_avg, scores = interactive.icluster(data, terms, user_input,
                                                                                               no_clusters,
                                                                                               user_feedback)
    pca = PCA(n_components=2).fit_transform(data)
    return send_data(cluster_key_terms, silhouette_avg, scores, clusters_docs, pca)


@app.route('/start')
def start():
    return get_clusters(k, [])


@app.route('/update', methods=['POST'])
def update():
    user_input_json = request.json

    user_input = [
        cluster_text.strip().split(' ')
        for cluster_text in user_input_json['terms'].split('\n')
        if len(cluster_text.strip()) > 0
    ]
    print(user_input)

    no_clusters = len(user_input)
    return get_clusters(no_clusters, user_input)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
