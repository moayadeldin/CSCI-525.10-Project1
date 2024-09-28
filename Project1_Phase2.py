import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import argparse


class PreprocessingData:

    def __init__(self,input_path,output_path):

        self.input_path = input_path
        self.output_path = output_path
        self.df = pd.read_csv(self.input_path)

    
    def removeColumns(self):

        """
        Removes the columns which wouldn' give too much information according to the discussion we had with Dr. Jacob in Project 1 Phase 1 in the office hours

        Returns:
            Pandas.DataFrame: DataFrame after we removed redundant features
        """

        self.df = self.df.drop(columns={
            "host_id", 
            "host_name", 
            "last_review", 
            "reviews_per_month", 
            "calculated_host_listings_count", 
            "number_of_reviews_ltm", 
            "license"
        })
        
        return self.df
    

    def gettingInsights(self):

        """
        Putting some Print Statements in our code showing the number of unique elements in the features "neighbourhood_group", "neighbourhood", and "room_type"
        """        

        print(f"Length of Neighbourhood Set Elements: {len(set(self.df['neighbourhood']))}")

        print(f"Length of Neighbourhood Group Set Elements: {len(set(self.df['neighbourhood_group']))}")

        print(f"The Neighbourhood Group Set Elements are: {set(self.df['neighbourhood_group'])}")

        print(f"Length of Room Type Set Elements: {len(set(self.df['room_type']))}")

        print(f"The Room types are: {set(self.df['room_type'])}")

    
    def labelEncoder(self,columns:list):

        """
        Label Encodes the columns we choose to encode in the dataset, from the gettingInsights print statements, it is quite convinient to label encode the biggest feature (neighbourhood_group) to avoid increasing dimensionality so much.

        Arguments:
            list: columns which contain the column(s) we want to label encode.

        Returns:
            Pandas.DataFrame: DataFrame containing labels label encoded.
        """

        le = LabelEncoder()
        
        if columns is not None:

            for col in columns:

                name_of_le_col = col + "_labelencoded"
                
                le_column = le.fit_transform(self.df[col])

                self.df[name_of_le_col] = le_column

                self.df = self.df.drop(columns=[col])

            print("Label Encoding Done Successfully.")
        

        else:

            raise(ValueError('This function can not be called without specfiying columns to encode'))
        
        return self.df


    def oneHotEncoder(self, columns:list):

        """
        One Hot Encodes the columns we choose to encode in the dataset, we are going to encode the neighbourhood & room_type features.
       
        Arguments:
            list: columns which contain the column(s) we want to one-hot encode.

        Returns:
            Pandas.DataFrame: DataFrame containing labels one-hot encoded.
        """


        ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')

        if columns is not None:

            for col in columns:

                array = self.df[col].values.reshape(-1,1)

                ohe_column = ohe.fit_transform(array)

                self.df = pd.concat([self.df, ohe_column], axis=1)

            # adjust the namings of one hot encoded columns

            self.df.columns = self.df.columns.str.replace("x0_", "oheencoded_")

            self.df = self.df.drop(columns=columns)

            print("One Hot Encoding Done Successfully.")

        else:

            raise(ValueError('This function can not be called without specfiying columns to encode'))
        
        return self.df


    def saveDataframe(self):

        self.df.to_csv(self.output_path)

        print(f"The preprocessed df is saved successfully to the following path {self.output_path}")



def main():

    parser = argparse.ArgumentParser(description="Preprocessing our AirBNB 2023 NYC Data")

    parser.add_argument("input_path", help="Path to the input CSV File")

    parser.add_argument("output_path", help="Path to save the Preprocessed CSV file")

    args = parser.parse_args()

    obj = PreprocessingData(args.input_path, args.output_path)

    obj.removeColumns()

    obj.gettingInsights()

    obj.labelEncoder(['neighbourhood'])

    obj.oneHotEncoder(['neighbourhood_group', 'room_type'])

    obj.saveDataframe()

if __name__ == "__main__":

    main()









        


                


