"""
*********************************************************************************
This code is a collaborative work between:

    Moayadeldin Hussain
    Muhammad Javed
    Salal Ali Khan

For CSCI-525.10 Project 1 Coursework submitted to Dr. Jacob Levmann.
*********************************************************************************
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import argparse


class PreprocessingData:

    def __init__(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path
        self.df = pd.read_csv(self.input_path)

    def removeColumns(self):
        """
        Removes the columns which wouldn' give too much information according to the discussion we had with Dr. Jacob in Project 1 Phase 1 in the office hours

        Returns:
            Pandas.DataFrame: DataFrame after we removed redundant features
        """

        self.df = self.df.drop(
            columns={
                "host_id",
                "host_name",
                "last_review",
                "reviews_per_month",
                "calculated_host_listings_count",
                "number_of_reviews_ltm",
                "license",
            }
        )

        return self.df
    
    
    def cleanPrices(self):

        """
        While exploring the dataset, we found out there are 27 records having the price "zero". We consider these records as noise outliers that should be removed because apparently no house rental is offered for free.

        Returns:
            Pandas.DataFrame: DataFrame after we removed outlier prices.
        """

        self.df = self.df[self.df['price']!=0]

        return self.df
    
    def cleanNights(self):

        """
        We believe also to be able to better predict the minimum number of nights, removing outliers is very essential for improving our results. In our implementation, we define outliers as the number of nights that appear only once. This means we retain all records where the minimum nights value is repeated at least twice.

        Retuns:
            Pandas.DataFrame: DataFrame after we removed outlier nights number.
        """

        self.df = self.df[self.df['minimum_nights']>=2]

        return self.df


    def gettingInsights(self):
        """
        Putting some Print Statements in our code showing the number of unique elements in the features "neighbourhood_group", "neighbourhood", and "room_type"
        """

        print(
            f"Length of Neighbourhood Set Elements: {len(set(self.df['neighbourhood']))}"
        )

        print(
            f"Length of Neighbourhood Group Set Elements: {len(set(self.df['neighbourhood_group']))}"
        )

        print(
            f"The Neighbourhood Group Set Elements are: {set(self.df['neighbourhood_group'])}"
        )

        print(f"Length of Room Type Set Elements: {len(set(self.df['room_type']))}")

        print(f"The Room types are: {set(self.df['room_type'])}")

    def labelEncoder(self, columns: list):
        """
        Label Encodes the columns we choose to encode in the dataset, from the gettingInsights print statements, it is quite convinient to label encode the biggest feature (neighbourhood_group) to avoid increasing dimensionality so much.

        Arguments:
            list: columns which contain the column(s) we want to label encode.

        Returns:
            Pandas.DataFrame: DataFrame containing labels label encoded.
        """

        le = LabelEncoder()

        # # encoding_dict={}

        if columns is not None:

            for col in columns:

                name_of_le_col = col + "_labelencoded"

                le_column = le.fit_transform(self.df[col])

                # encoding_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))

                # print(encoding_dict)

                self.df[name_of_le_col] = le_column

                self.df = self.df.drop(columns=[col])

            print("Label Encoding Done Successfully.")

        else:

            raise (
                ValueError(
                    "This function can not be called without specfiying columns to encode"
                )
            )

        return self.df

    def oneHotEncoder(self, columns: list):
        """
        One Hot Encodes the columns we choose to encode in the dataset, we are going to encode the neighbourhood & room_type features.

        Arguments:
            list: columns which contain the column(s) we want to one-hot encode.

        Returns:
            Pandas.DataFrame: DataFrame containing labels one-hot encoded.
        """

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
            transform="pandas"
        )

        if columns is not None:

            for col in columns:

                array = self.df[col].values.reshape(-1, 1)

                ohe_column = ohe.fit_transform(array)

                self.df = pd.concat([self.df, ohe_column], axis=1)

            # adjust the namings of one hot encoded columns

            self.df.columns = self.df.columns.str.replace("x0_", "oheencoded_")

            self.df = self.df.drop(columns=columns)

            print("One Hot Encoding Done Successfully.")

        else:

            raise (
                ValueError(
                    "This function can not be called without specfiying columns to encode"
                )
            )

        return self.df

    def wordsBag(self, column: str):
        """
        Apply Bag of Words technique to determine most relevant words to our price prediction. Pearson Correlation Coefficient was used to pick the highest correlated words to solve the High/Curse of Dimensionality problem.

        Arguments:
            list: Contains the Name of the column we are going to apply Bag Of Words to

        Returns:

        Pandas.DataFrame: DataFrame contains labels columns with the highest impactful Bag of Words

        """

        self.df["name"] = self.df[column].fillna("")

        # this vectorizer transforms a collection of text in the this column to a matrix/token counts.

        vectorizer = CountVectorizer(
            max_features=100
        )  # picks the top 100 frequent words

        bow_tokens = vectorizer.fit_transform(self.df[column].tolist())

        # now as we try to check if there are any names that has high correlation with the price feature, we will retrieve the price column from the dataframe and we will concatenate it with the tokens df.

        bow_tokens_df = pd.DataFrame(
            bow_tokens.toarray(), columns=vectorizer.get_feature_names_out()
        )

        bow_tokens_df["Price"] = self.df["price"]

        """The idea of using correlation is to prevent Curse of dimensionality as we discussed in the meeting with Dr. Jacob, he suggested that as a bonus thing to do correlation would be a proper thing then we take the highest correlated columns"""

        # now we check correlation

        bow_tokens_df_corrs = bow_tokens_df.corr(method="pearson")

        highest_corr = bow_tokens_df_corrs["Price"].sort_values(ascending=False)

        highest_corr = highest_corr[
            (highest_corr > 0.02) | (highest_corr < -0.02)
        ]  # we set the threshold to 0.02

        """You may check the words with highest correlation here. For example, the highest positive correlation with price is for the word units, which is pretty logic! if a house name mentions the number/type of units, it is usually more expensive
        """

        # print(highest_corr)

        # now we want to access these columns in the bag of words one hot encoded matrix and append it to our original dataframe.

        bow_columns = bow_tokens_df[highest_corr.index.tolist()]

        bow_columns = bow_columns.drop(columns=["Price"])

        for col in bow_columns:

            self.df[col] = bow_columns[col]

        print(
            "Bag of Words with Handling Correlation & Dimensionality Curse Done Successfully"
        )

        return self.df
    

    def saveDataframe(self):

        self.df.to_csv(self.output_path)

        print(
            f"The preprocessed df is saved successfully to the following path {self.output_path}"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Preprocessing our AirBNB 2023 NYC Data"
    )

    parser.add_argument("input_path", help="Path to the input CSV File")

    parser.add_argument("output_path", help="Path to save the Preprocessed CSV file")

    args = parser.parse_args()

    obj = PreprocessingData(args.input_path, args.output_path)

    obj.removeColumns()

    obj.gettingInsights()

    obj.labelEncoder(["neighbourhood"])

    obj.oneHotEncoder(["neighbourhood_group", "room_type"])

    obj.wordsBag("name")

    obj.cleanPrices()

    obj.cleanNights()

    obj.saveDataframe()


if __name__ == "__main__":

    main()
