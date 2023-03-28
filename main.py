import warnings

import spacy
from afinn import Afinn
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from spacy import Language
from spacy.tokens import Doc
import matplotlib.pyplot as plt


def build_tables(doc: Doc) -> (DataFrame, DataFrame):
    headings_table_1: list[str] = [
        'Token',
        'NE?',
        'NEtype',
        'Governor',
        'ListofDependants',
        'SentimentValueofToken',
        'SentimentValueofSentence'
    ]

    headings_table_2: list[str] = [
        'Token',
        'NEtype',
        'Governor',
        'ListofDependants',
        'SentimentValueofToken',
        'SentimentValueofSentence'
    ]

    df1: DataFrame = pd.DataFrame(columns=headings_table_1)
    df2: DataFrame = pd.DataFrame(columns=headings_table_2)

    afinn = Afinn()
    for sentence in doc.sents:
        for token in sentence:
            if not token.is_punct:
                ne: int = 1 if token.ent_type > 0 else 0

                token_score = afinn.score(str(token))
                sentence_score = afinn.score(str(sentence))

                row = {
                    'Token': token.text,
                    'NE?': ne,
                    'NEtype': token.ent_type_,
                    'Governor': token.head.text,
                    'ListofDependants': [child.text for child in token.children],
                    'SentimentValueofToken': token_score,
                    'SentimentValueofSentence': sentence_score
                }
                df1 = pd.concat([df1, pd.DataFrame([row], columns=headings_table_1)], ignore_index=True)

                if ne > 0:
                    row = {
                        'Token': token.text,
                        'NEtype': token.ent_type_,
                        'Governor': token.head.text,
                        'ListofDependants': [child.text for child in token.children],
                        'SentimentValueofToken': token_score,
                        'SentimentValueofSentence': sentence_score
                    }

                    df2 = pd.concat([df2, pd.DataFrame([row], columns=headings_table_2)], ignore_index=True)

    return df1, df2


def plot_clusters(kmeans, np_table, name="default cluster"):
    plt.title(name)
    plt.scatter(x=np_table[:, 0], y=np_table[:, 1], c=kmeans.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.5,
                label='centroid')
    plt.grid()
    plt.show()


def plot_elbow_method(np_table, name="default cluster"):
    sse = []
    list_k = list(range(1, 10))
    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(np_table)
        sse.append(km.inertia_)

    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.title(name)
    plt.grid()
    plt.show()


def build_cluster_1(table1: DataFrame, num_clusters: int) -> ndarray:
    temp: DataFrame = table1.drop(columns=['NEtype', 'Governor', 'ListofDependants'])
    np_table: ndarray = temp.to_numpy()
    label_encoder: LabelEncoder = LabelEncoder()

    np_table[:, 0] = label_encoder.fit_transform(np_table[:, 0])
    np_table[:, 1] = label_encoder.fit_transform(np_table[:, 1])
    np_table[:, 2] = label_encoder.fit_transform(np_table[:, 2])
    np_table[:, 3] = label_encoder.fit_transform(np_table[:, 3])

    kmeans: KMeans = KMeans(init='random', n_clusters=num_clusters, max_iter=1000, n_init=10)
    kmeans.fit(np_table)

    plot_clusters(kmeans, np_table, "T1 Clusters")
    plot_elbow_method(np_table, "T1 Elbow Method")

    return np_table


def build_cluster_2(table2: DataFrame, num_clusters: int) -> ndarray:
    temp: DataFrame = table2.drop(columns=['NEtype', 'Governor', 'ListofDependants'])
    np_table = temp.to_numpy()
    label_encoder = LabelEncoder()

    np_table[:, 0] = label_encoder.fit_transform(np_table[:, 0])
    np_table[:, 1] = label_encoder.fit_transform(np_table[:, 1])
    np_table[:, 2] = label_encoder.fit_transform(np_table[:, 2])

    kmeans: KMeans = KMeans(init='random', n_clusters=num_clusters, max_iter=1000, n_init=10)
    kmeans.fit(np_table)

    plot_clusters(kmeans, np_table, "T2 Clusters")
    plot_elbow_method(np_table, "T2 Elbow Method")

    return np_table


def display_table(table: pd.DataFrame, name):
    print(f"Table for {name}")
    print(table)


def main():
    # Fixing the annoying warning of the default value of n_init will change from 10 to 'auto' in 1.4
    warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4.")

    spacy.prefer_gpu()
    nlp: Language = spacy.load("en_core_web_sm")

    # Process documents
    raw_document_trump: str = open("APonTrump", "r").read()
    raw_document_s1: str = open("S1.txt", "r").read()

    # clean the text
    raw_document_trump = raw_document_trump \
        .replace("\n", "") \
        .replace("\n\n", " ") \
        .replace("``", "\"") \
        .replace(",", "") \
        .replace('"', '') \
        .replace("\\", "")

    raw_document_s1 = raw_document_s1 \
        .replace("\n", "") \
        .replace("\n\n", " ") \
        .replace("``", "\"") \
        .replace(",", "") \
        .replace('"', '') \
        .replace("\\", "")

    # Run the text through the SpaCy for preprocessing
    doc_trump: Doc = nlp(raw_document_trump)
    doc_s1: Doc = nlp(raw_document_s1)

    # Create tables for the data
    trump_table_1, trump_table_2 = build_tables(doc_trump)

    s1_table_1, s1_table_2 = build_tables(doc_s1)

    # Write the tables to csv files
    trump_table_1.to_csv('trump_table_1.csv', index=False)
    trump_table_2.to_csv('trump_table_2.csv', index=False)

    s1_table_1.to_csv('s1_table_1.csv', index=False)
    s1_table_2.to_csv('s1_table_2.csv', index=False)

    # Creating the clusters
    build_cluster_1(trump_table_1, 3)
    build_cluster_2(trump_table_2, 2)

    # Display the tables
    display_table(trump_table_1, "AP on Trump (Table 1)")
    display_table(trump_table_2, "AP on Trump (Table 2)")

    display_table(s1_table_1, "S1 (Table 1)")
    display_table(s1_table_2, "S1 (Table 2)")


if __name__ == '__main__':
    main()
