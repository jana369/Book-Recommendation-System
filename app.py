import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import train_test_split, cross_validate


# ------------------------------
# Load DataFrames
# ------------------------------
@st.cache_data()
def load_data():
    # Use the raw URL from GitHub
    books_df = pd.read_csv("https://raw.githubusercontent.com/jana369/Book-Recommendation-System/main/cleaned_books_data.csv")
    ratings_df = pd.read_csv("https://raw.githubusercontent.com/jana369/Book-Recommendation-System/main/ratings.csv")
    return books_df, ratings_df

books_df, ratings_df = load_data()


# ------------------------------
# Popularity-Based Recommendations
# ------------------------------
def popularity_recommendations(
    books_df, ratings_df, num_recommendations=10, metric="average_rating"
):
    if metric == "average_rating":
        popular_books = (
            ratings_df.groupby("book_id")
            .agg({"rating": "mean"})
            .rename(columns={"rating": "average_rating"})
        )
        popular_books = popular_books.merge(
            books_df, on="book_id", suffixes=("", "_books")
        ).sort_values("average_rating", ascending=False)

    elif metric == "ratings_count":
        popular_books = (
            ratings_df.groupby("book_id")
            .agg({"rating": "count"})
            .rename(columns={"rating": "ratings_count"})
        )
        popular_books = popular_books.merge(
            books_df, on="book_id", suffixes=("", "_books")
        ).sort_values("ratings_count", ascending=False)

    elif metric == "weighted_score":
        C = ratings_df["rating"].mean()
        m = ratings_df["book_id"].value_counts().quantile(0.9)
        q_books = ratings_df.groupby("book_id").agg(
            average_rating=("rating", "mean"), ratings_count=("rating", "count")
        )
        q_books = q_books[q_books["ratings_count"] >= m]
        q_books["weighted_score"] = (
            q_books["average_rating"] * q_books["ratings_count"] + C * m
        ) / (q_books["ratings_count"] + m)
        popular_books = q_books.merge(
            books_df, on="book_id", suffixes=("", "_books")
        ).sort_values("weighted_score", ascending=False)

    else:
        raise ValueError(
            "Metric not recognized. Choose from 'average_rating', 'ratings_count', 'weighted_score'"
        )
    popular_books.columns = popular_books.columns.str.replace(
        "_x", "", regex=True
    ).str.replace("_y", "", regex=True)
    # Drop duplicate columns if necessary
    popular_books = popular_books.loc[:, ~popular_books.columns.duplicated()]

    return popular_books.head(num_recommendations)


# ------------------------------
# Content-Based Filtering
# ------------------------------
@st.cache(allow_output_mutation=True)
def build_content_model(books_df):
    books_df["description"] = books_df["description"].fillna("")
    books_df["genres"] = books_df["genres"].fillna("")
    books_df["authors"] = books_df["authors"].fillna("")
    books_df["content"] = (
        books_df["description"] + " " + books_df["genres"] + " " + books_df["authors"]
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_df["content"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_df.index, index=books_df["title"]).drop_duplicates()
    return cosine_sim, indices


cosine_sim, indices = build_content_model(books_df)


def get_content_recommendations(
    title, books_df, cosine_sim, indices, num_recommendations=5
):
    if title not in indices:
        st.error(f"Book titled '{title}' not found in the database.")
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_recommendations + 1]
    book_indices = [i[0] for i in sim_scores]
    return books_df[["book_id", "title", "authors"]].iloc[book_indices]


# ------------------------------
# Collaborative Filtering
# ------------------------------
@st.cache(allow_output_mutation=True)
def train_svd_model(ratings_df):
    reader = Reader(
        rating_scale=(ratings_df["rating"].min(), ratings_df["rating"].max())
    )
    data = Dataset.load_from_df(ratings_df[["user_id", "book_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd_model = SVD(random_state=42)
    svd_model.fit(trainset)
    return svd_model


svd_model = train_svd_model(ratings_df)


def recommend_collaborative(
    user_id, ratings_df, books_df, svd_model, num_recommendations=5
):
    all_book_ids = books_df["book_id"].unique()
    rated_books = ratings_df[ratings_df["user_id"] == user_id]["book_id"].tolist()
    books_to_predict = [book for book in all_book_ids if book not in rated_books]
    predictions = [svd_model.predict(user_id, book_id) for book_id in books_to_predict]
    pred_df = pd.DataFrame(
        {
            "book_id": books_to_predict,
            "predicted_rating": [pred.est for pred in predictions],
        }
    )
    pred_df = pred_df.sort_values("book_id", ascending=False)
    top_recommendations = pred_df.head(num_recommendations)
    recommended_books = top_recommendations.merge(books_df, on="book_id")
    return recommended_books[["book_id", "title", "authors", "predicted_rating"]]


# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("📚 Book Recommendation System")

st.sidebar.title("Recommendation Methods")
recommendation_method = st.sidebar.selectbox(
    "Choose a recommendation method:",
    ("Popularity-Based", "Content-Based", "Collaborative Filtering"),
)

# Popularity-Based Recommendations
if recommendation_method == "Popularity-Based":
    st.header("📈 Popularity-Based Recommendations")
    metric = st.selectbox(
        "Choose a popularity metric:",
        ("average_rating", "ratings_count", "weighted_score"),
    )
    num_recommend = st.slider("Number of recommendations:", 1, 20, 10)
    if st.button("Show Recommendations"):
        top_books = popularity_recommendations(
            books_df, ratings_df, num_recommendations=num_recommend, metric=metric
        )
        if metric == "weighted_score":
            st.write(
                top_books[
                    [
                        "title",
                        "authors",
                        "average_rating",
                        "ratings_count",
                        "weighted_score",
                    ]
                ]
            )
        elif metric == "average_rating":
            st.write(top_books[["title", "authors", "average_rating"]])
        elif metric == "ratings_count":
            st.write(top_books[["title", "authors", "ratings_count"]])

# Content-Based Recommendations
elif recommendation_method == "Content-Based":
    st.header("🔍 Content-Based Recommendations")
    book_title = st.selectbox("Select a book you like:", books_df["title"].values)
    num_recommend = st.slider("Number of recommendations:", 1, 20, 5)
    if st.button("Show Recommendations"):
        recommended_books = get_content_recommendations(
            book_title, books_df, cosine_sim, indices, num_recommendations=num_recommend
        )
        if not recommended_books.empty:
            st.write(recommended_books)
        else:
            st.write("No recommendations found.")

# Collaborative Filtering Recommendations
elif recommendation_method == "Collaborative Filtering":
    st.header("👥 Collaborative Filtering Recommendations")
    user_id = st.number_input("Enter your User ID:", min_value=1, step=1)
    num_recommend = st.slider("Number of recommendations:", 1, 20, 5)
    if st.button("Show Recommendations"):
        # Check if user exists
        if user_id not in ratings_df["user_id"].unique():
            st.error("User ID not found. Please enter a valid User ID.")
        else:
            recommended_books = recommend_collaborative(
                user_id,
                ratings_df,
                books_df,
                svd_model,
                num_recommendations=num_recommend,
            )
            if not recommended_books.empty:
                st.write(recommended_books)
            else:
                st.write("No recommendations found.")

# Optional: Add a footer or additional information
st.markdown("---")
st.markdown("Developed by [Your Name](https://yourwebsite.com)")

# ------------------------------
# Run Streamlit App
# ------------------------------
# To run the app, execute the following command in your terminal:
# streamlit run app.py
