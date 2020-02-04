# Imports
from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import xgboost


# Make the app
def pred_med():
    APP=Flask(__name__)
    @APP.route("/")
    def test():
        return "Pred Med Strain Predictor, hello"


    @APP.route("/pred", methods=(["GET"]))
    def pred_list():
        """
         x = string to predict from (description)
         1. Predict the nearest neighbors to the inputted description
         2. Predict what type of cannabis the user is looking for with probability
       
        """
        req_data = request.get_json()
        x = req_data["x"]

        # Read in data
        df = pd.read_csv("https://raw.githubusercontent.com/med-cabinet-5/data-science/master/data/canna.csv")
        # Fill NaN with empty strings
        df = df.fillna("")

        # Instantiate vectorizer object
        tfidf = TfidfVectorizer(stop_words="english", min_df=0.025, max_df=.98, ngram_range=(1,3))

        # Create a vocabulary and get word counts per document
        dtm = tfidf.fit_transform(df['alltext'])
 
        # Get feature names to use as dataframe column headers
        dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())
         
        # Fit on TF-IDF Vectors
        nn = NearestNeighbors(n_neighbors=5, algorithm="kd_tree", radius=0.5)
        nn.fit(dtm)

        # Turn Review into a list, transform, and predict
        review = [x]
        new = tfidf.transform(review)
        pred = nn.kneighbors(new.todense())[1][0]

    
        #create empty list
        pred_dict = []
        for x in pred:
            # add new dictionary to pred_dict containing predictions
            preds_list ={"strain":df["Strain"][x],
                        "type": df["Type_raw"][x],
                        "description": df["Description_raw"][x],
                        "flavor": df["Flavor_raw"][x],
                        "effects": df["Effects_raw"][x],
                        "ailments": df["Ailment_raw"][x]}
            pred_dict.append(preds_list)
    
        # Load data for model 2
        model = pickle.load(open("stretch.sav", "rb"))
        #Pull result out
        pred_2 = model.predict(review)[0]
 
        #Grab max predict proba                   
        predict_proba = model.predict_proba(review)[0].max() * 100

        # Mapper to change result into string
        mapper = ({5: "Hybrid",
            4: "Indica",
            3: "Sativa",
            2: "Hybrid, Indica",
            1: "Sativa, Hybrid"})
    
        # Apply mapper to newly made Series
        strain_type = pd.Series(pred_2).map(mapper)[0]
    
        # Create new dictionary element
        new_dict = {"proba":f"There is a {round(predict_proba, 2)}% that your looking for a {strain_type}"}
    
        # Add new dicitonary to list of dictionaries
        pred_dict.append(new_dict)

        return jsonify(pred_dict)
    
    return APP
