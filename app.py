from collections import defaultdict
import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn.datasets import fetch_20newsgroups
import datasets
import interactive
import preprocessing

app = Flask(__name__)

# configs
# algo
k = 6  # start with assuming there are this many clusters
# options = (1.1, 25, 0.01, 0)
max_features = 1000
docs_limit = 500
n_samples = 2000

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

# algo
corpus = datasets.get_bbc()
pre_processor = preprocessing.NLPProcessor(max_features=max_features)
bbc_vectorized_features_bound = pre_processor.fit_transform(corpus)
data = bbc_vectorized_features_bound[:docs_limit].todense()
terms = np.array(pre_processor.vec.get_feature_names()).reshape((1, max_features))

# cosmetic
colors = [x.format(alpha=color_alpha) for x in colors_base]


# def get_assignments(cluster_docs):
#     assignments = np.zeros(docs)
#
#     for c in range(len(cluster_docs)):
#         for d in range(len(cluster_docs[c])):
#             i = cluster_docs[c][d]
#             if i < docs:
#                 assignments[i] = c
#
#     return list(assignments)


def get_pca_for_highcharts(clusters_docs, pca):
    cluster_data = defaultdict(list)
    for i, docs in enumerate(clusters_docs):
        for doc in docs:
            pca_doc = pca[doc - 1]
            cluster_data[i].append({'x': pca_doc[0], 'y': pca_doc[1], 'name': corpus[doc - 1][:text_top_n_chars]})

    return [
        {
            'name': f'Cluster {i+1}',
            'color': colors[i],
            'data': cluster_data[i],
        }
        for i in range(k)
    ]

def set_data_set(dataset_name):
    if dataset_name == "BBC":
        corpus = datasets.get_bbc()
    elif dataset_name == "20 News Groups":
        dataset = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'))
        corpus = dataset.data
    elif dataset_name == "All the news":
        corpus = datasets.get_all_the_news()

    vectorized_features_bound = pre_processor.fit_transform(corpus)
    global data, terms
    data = vectorized_features_bound[:docs_limit].todense()
    terms = np.array(pre_processor.vec.get_feature_names()).reshape((1, max_features))



def get_algorithm(algorithm_name, clusters):
    if algorithm_name == "dbscan":
        algorithm = cluster.DBSCAN(eps=.2)
    elif algorithm_name == "birch":
        algorithm = cluster.Birch(n_clusters=clusters)
    elif algorithm_name == "means":
        algorithm = cluster.MiniBatchKMeans(n_clusters=clusters)
    elif algorithm_name == "spectral":
        algorithm = cluster.SpectralClustering(n_clusters=clusters, eigen_solver='arpack', affinity="nearest_neighbors")
    else:
        algorithm = cluster.AffinityPropagation(damping=.9, preference=-200)
    return algorithm

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load', methods=['POST'])
def load():
    user_input_json = request.json
    algorithms = ["iKMeans", "DBSCAN", "birch", "means", "spectral", "affinity"]
    dataset_names = ["BBC", "20 News Groups", "All the news"]
    
    current_dataset = user_input_json["dataset"]
    if current_dataset == None:
        current_dataset = dataset_names[0]
    set_data_set(current_dataset)
    
    current_algorithm = user_input_json["algorithm"]
    if current_algorithm == None:
        current_algorithm = algorithms[0]
    if current_algorithm == "iKMeans":
        return get_clusters(k, [])

    clusters = user_input_json["numOfClusters"]
    if clusters == "" or clusters == None:
        clusters = 5
    else:
        clusters = int(clusters)

    pca = IncrementalPCA(n_components=2).fit_transform(data)
    pca = StandardScaler().fit_transform(pca)
    # key_terms = [list(x) for x in key_terms]

    algorithm = get_algorithm(current_algorithm, clusters)
    algorithm.fit(pca)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(pca)
    cluster_key_terms = None
    silhouette_avg = None
    clusters_dict = dict()
    for i in range(len(y_pred)):
        cluster_id = int(y_pred[i])
        if not cluster_id in clusters_dict.keys():
            clusters_dict[cluster_id] = list()
        clusters_dict[cluster_id].append(i)
    clusters_docs = [None]*len(clusters_dict.keys())
    for key in clusters_dict.keys():
        clusters_docs[key] = clusters_dict[key]
    return send_data(cluster_key_terms, silhouette_avg, clusters_docs, pca)

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


def send_data(cluster_key_terms, silhouette_avg, clusters_docs, pca):
    return jsonify({
        'cluster_key_terms': cluster_key_terms,
        # 'key_terms': key_terms,
        'silhouette': silhouette_avg,
        'pca': get_pca_for_highcharts(clusters_docs, pca),
    })


def get_clusters(no_clusters, user_input):
    user_feedback = -1 if len(user_input) == 0 else +1
    clusters_docs, cluster_key_terms, key_terms, silhouette_avg = interactive.icluster(data, terms, user_input,
                                                                                       no_clusters, user_feedback)
    pca = PCA(n_components=2).fit_transform(data)
    # key_terms = [list(x) for x in key_terms]
    return send_data(cluster_key_terms, silhouette_avg, clusters_docs, pca)


@app.route('/start')
def start():
    return get_clusters(k, [])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
