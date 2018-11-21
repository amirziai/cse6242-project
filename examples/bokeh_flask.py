from sklearn.decomposition import IncrementalPCA
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from bokeh.plotting import figure
import numpy as np
import datasets
from flask import Flask, request, jsonify, render_template
from bokeh.embed import components
import preprocessing


app = Flask(__name__)

datasets.DATASET_PATH = "../" + datasets.DATASET_PATH
all_documents = datasets.get_bbc()

document_set = [all_documents[0:500], all_documents[500:1000], all_documents[1000:1500], all_documents[1500:2000]]


PLOT_SIZE = 400
colors = np.array([x for x in ('#00f', '#0f0', '#f00', '#0ff', '#f0f', '#ff0')])
colors = np.hstack([colors] * 20)

algorithms = ["dbscan", "birch", "means", "spectral", "affinity"]
dataset_names = ["First", "Second", "Third", "Fourth"]

def vectorize_documents(documents):
    processor = preprocessing.NLPProcessor('tf-idf')
    X = processor.fit_transform(documents).todense()
    return X

def reduce_dimensions(vectors):
    pca = IncrementalPCA(n_components=2).fit(vectors)
    data2D = pca.transform(vectors)
    data2D = StandardScaler().fit_transform(data2D)
    return data2D


def get_data_set(dataset_name):
    if dataset_name == "First":
        documents = document_set[0]
    elif dataset_name == "Second":
        documents = document_set[1]
    elif dataset_name == "Third":
        documents = document_set[2]
    else:
        documents = document_set[3]

    document_vectors = vectorize_documents(documents)
    data2D = reduce_dimensions(document_vectors)
    return data2D

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


# Create the main plot
def create_figure(algorithm_name, clusters, dataset_name):

    data2D = get_data_set(dataset_name)
    algorithm = get_algorithm(algorithm_name, clusters)
    algorithm.fit(data2D)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(data2D)

    p = figure(output_backend="webgl", title=algorithm.__class__.__name__,
               plot_width=PLOT_SIZE, plot_height=PLOT_SIZE)

    p.scatter(data2D[:, 0], data2D[:, 1], color=colors[y_pred].tolist(), alpha=0.5, )

    return p


# Index page
@app.route('/')
def index():
    # Determine the selected feature
    current_algorithm = request.args.get("algorithm")
    if current_algorithm == None:
        current_algorithm = algorithms[0]

    current_dataset = request.args.get("dataset")
    if current_dataset == None:
        current_dataset = dataset_names[0]

    clusters = request.args.get("clusters")
    if clusters == "" or clusters == None:
        clusters = 5
    else:
        clusters = int(clusters)

    plot = create_figure(current_algorithm, clusters, current_dataset)

    # Embed plot into HTML via Flask Render
    script, div = components(plot)
    return render_template("graph.html", script=script, div=div,
                           clusters=clusters, datasets=dataset_names, algorithms=algorithms,
                           current_dataset=current_dataset, current_algorithm=current_algorithm)


if __name__ == '__main__':
    app.run(port=5000, debug=False)