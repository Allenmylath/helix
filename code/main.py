from helix import Client, Loader
from helix.client import hnswload, hnswsearch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load and prepare the Titanic dataset
df = pd.read_parquet("code/titanic.parquet")

# Create numerical features for embedding
# Standard Titanic columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
numerical_features = []

# Convert categorical to numerical
if 'Sex' in df.columns:
    df['Sex_encoded'] = df['Sex'].map({'male': 1, 'female': 0})
    numerical_features.append('Sex_encoded')

if 'Embarked' in df.columns:
    df['Embarked_encoded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    numerical_features.append('Embarked_encoded')

# Add existing numerical columns
for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    if col in df.columns:
        numerical_features.append(col)

# Prepare the feature matrix
feature_matrix = df[numerical_features].copy()

# Handle missing values
imputer = SimpleImputer(strategy='median')
feature_matrix = pd.DataFrame(
    imputer.fit_transform(feature_matrix), 
    columns=numerical_features
)

# Standardize features
scaler = StandardScaler()
embeddings = scaler.fit_transform(feature_matrix)

# Save embeddings back to dataframe
df['embedding'] = [emb.tolist() for emb in embeddings]

# Save the processed data
df[['embedding']].to_parquet("titanic_with_embeddings.parquet")

# Now use with Helix
db = Client(local=True)
data = Loader("titanic_with_embeddings.parquet", cols=["embedding"])
ids = db.query(hnswload(data))

# Use a valid index (check dataset size first)
dataset_size = len(df)
query_idx = min(100, dataset_size - 1)  # Use index 100 or last available
my_query = embeddings[query_idx].tolist()

vecs = db.query(hnswsearch(my_query))
print("Search response:")
[print(vec) for vec in vecs]