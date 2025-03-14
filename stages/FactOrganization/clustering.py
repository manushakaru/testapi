import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from common.config import CLUSTER_CONFIGS
from common.gpt_helper import GPTHelper
from common.utils import console
from common.utils.timing_logger import LOGGER, log_execution_time
from dotenv import load_dotenv
from kneed import KneeLocator
from nltk.corpus import stopwords
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

load_dotenv()

gpt_helper = GPTHelper()

STOPWORDS = set(stopwords.words("english"))


def get_all_embeddings(texts):
    embeddings = []
    for text in texts:
        try:
            embedding = gpt_helper.get_embeddings(text)
            embeddings.append(embedding)
        except Exception as e:
            console.print(f"Error getting embedding for text: {e}")
            embeddings.append(np.zeros(1536))
    return np.array(embeddings)


def reduce_dimensions(X, method="pca"):
    """Use PCA, t-SNE or UMAP for dimensionality reduction."""
    if method == "pca":
        pca = PCA(n_components=min(len(X), 10))
        X = pca.fit_transform(X)
    elif method == "tsne":
        tsne = TSNE(n_components=2, random_state=42)
        X = tsne.fit_transform(X)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        X = reducer.fit_transform(X)
    return X


def plot_umap(embeddings_2d, labels, facts, file_path):
    filename = f"{file_path}/umap_visualization.html"
    # Prepare hover text
    hover_texts = [
        f"Cluster: {label}<br>{fact['fact_content'][:150]}..."
        for label, fact in zip(labels, facts)
    ]

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": labels.astype(str),
            "hover_text": hover_texts,
        }
    )

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster",
        hover_name="hover_text",
        title="Fact Clusters UMAP Visualization",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        labels={"color": "Cluster ID"},
    )

    # Customize layout
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12),
        plot_bgcolor="rgba(240,240,240,0.9)",
        width=1200,
        height=800,
        title_x=0.5,
    )

    # Save and show
    fig.write_html(filename)
    console.print(f"[bold green]UMAP visualization saved to {filename}[/bold green]")
    return fig


def get_k_value(num_facts):
    max_clusters = CLUSTER_CONFIGS["MAX_CLUSTER_SIZE"]
    assumed_clusters = math.ceil(num_facts / 6) + 1
    if max_clusters < assumed_clusters:
        return max_clusters
    return assumed_clusters


@log_execution_time
def cluster_facts(facts, file_path, threshold=0.2):
    texts = [d["fact_content"] for d in facts]

    console.print("[bold yellow]Getting Embeddings...[/bold yellow]")
    X = get_all_embeddings(texts)
    console.print("[bold green]Getting Embeddings Completed![/bold green]")

    # Normalize & Reduce Dimensions using RobustScaler and UMAP
    X = RobustScaler().fit_transform(X)  # Use RobustScaler instead of StandardScaler
    X = reduce_dimensions(X, method="umap")  # Change method to 'pca', 'tsne', or 'umap'

    console.print("[bold green]Getting Embeddings Completed![/bold green]")

    # Finding Optimal k
    # k_values = range(
    #     2, max(CLUSTER_CONFIGS["MAX_CLUSTER_SIZE"], math.ceil(len(X) / 6) + 1)
    # )

    k_values = range(2, get_k_value(len(X)))

    neg_log_likelihoods, models = [], {}

    for k in k_values:
        console.print(f"k = {k}")
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=42
        )  # Try 'full' covariance type
        gmm.fit(X)
        total_log_likelihood = gmm.score(X) * X.shape[0]
        neg_log_likelihoods.append(-total_log_likelihood)
        models[k] = gmm

    kneedle = KneeLocator(
        k_values, neg_log_likelihoods, curve="convex", direction="increasing"
    )
    best_k = kneedle.elbow or min(k_values, key=lambda k: models[k].bic(X))

    console.print(
        f"[bold green]Best number of clusters (k) selected: {best_k}[/bold green]"
    )

    final_gmm = models[best_k]
    probabilities = final_gmm.predict_proba(X)
    labels = final_gmm.predict(X)

    # Fact-wise Assignments
    fact_wise_clusters = []
    cluster_dict = defaultdict(list)

    for i, fact in enumerate(facts):
        assigned_clusters = [
            {"cluster_id": cluster_id + 1, "probability": float(prob)}
            for cluster_id, prob in enumerate(probabilities[i])
            if prob >= threshold
        ]
        fact_wise_clusters.append(
            {
                "fact_id": fact["fact_id"],
                "fact_content": fact["fact_content"],
                "assigned_clusters": assigned_clusters,
            }
        )

        for cluster in assigned_clusters:
            cluster_dict[cluster["cluster_id"]].append(
                {"fact_id": fact["fact_id"], "fact_content": fact["fact_content"]}
            )

    cluster_wise_facts = [
        {"cluster_id": cluster_id, "cluster_size": len(facts), "facts": facts}
        for cluster_id, facts in cluster_dict.items()
    ]

    # Clustering Evaluation Metrics
    eval_metrics = evaluate_clustering(X, labels, probabilities)

    # Cluster Summarization
    cluster_summary = summarize_clusters(X, labels, cluster_dict)

    plot_umap(X, labels, facts, file_path)
    # Combine original clusters and sub-clusters into a final result
    return {
        "cluster_wise_facts": cluster_wise_facts,
        "fact_wise_clusters": fact_wise_clusters,
        "evaluation_metrics": eval_metrics,
        "cluster_summary": cluster_summary,
    }


def evaluate_clustering(X, labels, probabilities):
    silhouette = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    cluster_entropy = entropy(probabilities, axis=1).mean()

    # Convert numpy.float32 to regular Python float
    silhouette = float(silhouette)
    db_score = float(db_score)
    cluster_entropy = float(cluster_entropy)

    console.print(f"[bold cyan]Silhouette Score:[/bold cyan] {silhouette:.4f}")
    console.print(f"[bold cyan]Davies-Bouldin Index:[/bold cyan] {db_score:.4f}")
    console.print(f"[bold cyan]Cluster Entropy:[/bold cyan] {cluster_entropy:.4f}")

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_score": db_score,
        "cluster_entropy": cluster_entropy,
    }


def summarize_clusters(X, labels, cluster_dict):
    cluster_centers = []
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        cluster_centers.append(cluster_points.mean(axis=0))

    cluster_summaries = {}
    for cluster_id, center in enumerate(cluster_centers):
        cluster_facts = [
            fact["fact_content"] for fact in cluster_dict.get(cluster_id + 1, [])
        ]
        distances = np.linalg.norm(X[labels == cluster_id] - center, axis=1)
        closest_fact_idx = np.argmin(distances)
        representative_fact = (
            cluster_facts[closest_fact_idx] if cluster_facts else "N/A"
        )
        common_words = extract_keywords(cluster_facts)

        cluster_summaries[cluster_id + 1] = {
            "representative_fact": representative_fact,
            "top_keywords": common_words,
        }

    return cluster_summaries


def extract_keywords(facts, top_n=5):
    words = re.findall(r"\w+", " ".join(facts).lower()) if facts else []
    filtered_words = [word for word in words if word not in STOPWORDS]
    common_words = [word for word, _ in Counter(filtered_words).most_common(top_n)]
    return common_words
