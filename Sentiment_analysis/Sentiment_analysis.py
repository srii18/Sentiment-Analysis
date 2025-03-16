# Import libraries and suppress warnings
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

# Download necessary NLTK packages
nltk.download('punkt', quiet=True)  # Added quiet=True to reduce output verbosity
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Load the Amazon review dataset
data = pd.read_csv('AmazonReview.csv')
# Display first 5 rows of the dataset (should capture the return value)
print(data.head())  
# Display summary information about the dataframe
data.info()  

# Remove rows with missing values
data.dropna(inplace=True)

# Convert sentiment ratings into binary classification:
# Ratings 1,2,3 -> negative (0)
data.loc[data['Sentiment']<=3,'Sentiment'] = 0
# Ratings 4,5 -> positive (1)
data.loc[data['Sentiment']>3,'Sentiment'] = 1

# Get English stopwords
stp_words = stopwords.words('english')

# Define function to clean reviews by removing stopwords
def clean_review(review):
    # Check if review is a string before processing
    if isinstance(review, str):
        cleanreview = " ".join(word for word in review.split() if word not in stp_words)
        return cleanreview
    return ""  # Return empty string for non-string inputs

# Apply cleaning function to all reviews
data['Review'] = data['Review'].apply(clean_review)
print(data.head())  # Display the cleaned data

# Count the distribution of sentiment classes and print it
print(data['Sentiment'].value_counts())

# Create a word cloud for negative reviews (sentiment=0)
consolidated = ' '.join(word for word in data['Review'][data['Sentiment']==0])  # Removed astype(str) as it's already a string
wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, background_color='white')  # Added background color
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Negative Reviews', fontsize=20)  # Added title for clarity
plt.tight_layout()  # Improved layout
plt.show()

# Convert text reviews to numerical features using TF-IDF vectorization
# Limiting to top 2500 features
cv = TfidfVectorizer(max_features=2500, stop_words='english')  # Added stop_words parameter as backup
X = cv.fit_transform(data['Review']).toarray()

# Split data into training (80%) and testing (20%) sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, data['Sentiment'],
                        test_size=0.2, random_state=42)

# Import model and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence

# Train the model on training data
model.fit(x_train, y_train)

# Make predictions on test data
pred = model.predict(x_test)

# Calculate and display model accuracy
print(f"Model accuracy: {accuracy_score(y_test, pred):.4f}")

# Display detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Import confusion matrix utilities
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Create and visualize confusion matrix
cm = confusion_matrix(y_test, pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                      display_labels=["Negative", "Positive"])  # Added meaningful labels
cm_display.plot(cmap='Blues')  # Added color map for better visualization
plt.title('Confusion Matrix')  # Added title
plt.tight_layout()
plt.show()