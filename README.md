# Build a Simple Recommender System 

# Overview
A content-based movie recommendation system designed to suggest movies based on features like genre, story, and lead studio. It leverages TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity to evaluate and recommend movies with similar characteristics.

# Features
Content-based Recommendations: Suggest movies by analyzing textual data such as genre, story, and lead studio.
Data Preprocessing: Handling missing/null values, normalization of numerical features, and duplicate removal.
TF-IDF Vectorization: Converts textual data into numerical representations.
Cosine Similarity: Measures similarity between movies based on their feature vectors.

# Usage
Load and preprocess the dataset: 
df = pd.read_csv(url)
df = preprocess_midi(df)

Create the TF-IDF matrix:
tfidf_matrix = TfidfVectorizer().fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix)

Generate movie recommendations:
recommended_movies = recommend_movies('Harry Potter and the Order of the Phoenix', cosine_sim)
print(recommended_movies)

# Dataset
Dataset Source: Reisner on GitHub
Columns included: Genre, Story, LeadStudio, RottenTomatoes, AudienceScore, Budget, and more.

# Libraries Utilized
Pandas: For data manipulation and analysis.
Scikit-learn: For TF-IDF vectorization and cosine similarity calculation.
Numpy: For numerical operations.
Matplotlib: For data visualization (optional).
Seaborn: For enhanced data visualization (optional).
