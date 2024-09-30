import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
df = pd.read_csv('dataset.csv')

# Check if 'name' column exists
if 'name' in df.columns:
    # Fill NaN values in 'name' column
    df['name'] = df['name'].fillna('')

    # Initialize CountVectorizer (Bag of Words)
    vectorizer = CountVectorizer(max_features=500)  # Limit to 500 words or adjust as needed

    # Fit and transform the 'name' column
    bow_matrix = vectorizer.fit_transform(df['name'])

    # Convert the BoW matrix to a DataFrame for easier viewing
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenate BoW features with the original DataFrame
    df_bow = pd.concat([df, bow_df], axis=1)

    # Drop the original 'name' column if needed
    df_bow = df_bow.drop('name', axis=1)

    # Save the new dataset with BoW features
    df_bow.to_csv('airbnb_nyc_2023_bow.csv', index=False)

    print("Bag of Words transformation complete and saved as 'airbnb_nyc_2023_bow.csv'.")
else:
    print("The 'name' column does not exist in the dataset.")
