#!/usr/bin/env python3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns 
# Load the main dataset (UN general debates)
df_main = pd.read_csv('/Users/noraelsetouhi/Downloads/un-general-debates.csv')

# Load the methodology dataset with the correct delimiter
methodology_path = '/Users/noraelsetouhi/Downloads/UNSD—Methodology.csv'
try:
    # Specify the correct delimiter
    df_methodology = pd.read_csv(methodology_path, delimiter=';')
    
    # Display the first few rows and all column names of the methodology dataset
    print("\nMethodology Dataset:")
    print(df_methodology.head())
    print("\nAll Methodology Dataset Columns:")
    print(df_methodology.columns)

except FileNotFoundError:
    print(f"File not found at path: {methodology_path}")
except pd.errors.ParserError as e:
    print(f"Error reading CSV: {e}")

# Merge datasets on 'country' in df_main and 'ISO-alpha3 Code' in df_methodology
merged_df = pd.merge(df_main, df_methodology[['ISO-alpha3 Code', 'Region Name']], how='left', left_on='country', right_on='ISO-alpha3 Code')

# Drop rows with missing values in the 'Region Name' column
merged_df = merged_df.dropna(subset=['Region Name'])

# Display the merged dataset
print("\nMerged Dataset:")
print(merged_df.head())

# Display information about the classes
print("\nClass Information:")
print(merged_df['Region Name'].value_counts())

# Preprocess the text data
merged_df['processed_text'] = merged_df['text'].str.lower()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['processed_text'],
    merged_df['Region Name'],
    test_size=0.33,
    random_state=42
)

# Combine the text data from training and testing sets
all_text_data = pd.concat([X_train, X_test])

# Tokenize the text into words
all_words = ' '.join(all_text_data).split()

# Get the total number of unique features
total_unique_features = len(set(all_words))
print(f"Total Number of Unique Features: {total_unique_features}")

# Define custom stop words
custom_stop_words = ['africa', 'african', 'zealand', 'fiji', 'australia', 
                     'international', 'nations', 'united', 'peace', 'security', 'world', 'states', 
                     'countries', 'europe', 'people', 'general', 'assembly', 'government', 'session',
                     'american', 'america', 'council', 'europe', 'european', 'country', 'republic', 
                     'organization', 'efforts', 'new', 'delegation', 'latin', 'situation', 'political', 'order'
                     'national', 'south', 'east', 'region', 'order', 'peoples', 'support', 'community', 'national', 'like', 
                     'conference', 'problems', 'hope', 'time', 'cooperation', 'operation', 'important', 'state', 'developing',
                     'year', 'continue', 'namibia', 'guinea', 'continent', 'president', 'mr', 'small', 'papau', 'solomon']  # Add any words you want to exclude

# Combine standard English stop words with custom stop words
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# Vectorize text data using TF-IDF with combined stop words
vectorizer = TfidfVectorizer(max_features=3000, stop_words=all_stop_words)  # Adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=classifier.classes_)

# Visualize confusion matrix with seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Display the top N features for each class in the Naive Bayes classifier
top_n_features = 10
for i, region in enumerate(classifier.classes_):
    print(f"\nRegion: {region}")
    top_features_idx = classifier.feature_log_prob_[i, :].argsort()[::-1][:top_n_features]
    top_features = [feature_names[idx] for idx in top_features_idx]
    print(top_features)

# Apply clustering using K-means on TF-IDF vectors
num_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_test_tfidf)

# Visualize clusters using PCA (2D)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X_test_tfidf.toarray())
reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

# Scatter plot of clusters
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering of UN Speeches')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



