from collections import defaultdict
import numpy as np
from flask import Flask, render_template, jsonify, request
from sklearn.decomposition import PCA

import datasets
import interactive
import preprocessing

app = Flask(__name__)

# configs
# algo
k = 6  # start with assuming there are this many clusters
# options = (1.1, 25, 0.01, 0)
max_features = 1000
docs_limit = 1000

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


@app.route('/')
def index():
    return render_template('index.html')


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
