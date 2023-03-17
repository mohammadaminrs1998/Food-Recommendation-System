import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Foodie", page_icon=":pizza:")
st.title("Foodie | Food Recommendation System")
st.write("Made With ❤️ By Mohammad Amin Rezaei Sepehr")
st.image("https://images.unsplash.com/photo-1595854341625-f33ee10dbf94?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80")

st.subheader("Whats your preference?")
vegn = st.radio("Vegetables or Non-vegetables",["veg","non-veg"],index = 1) 

st.subheader("What Cuisine do you prefer?")
cuisine = st.selectbox("Choose your favourite!",['Healthy Food', 'Snack', 'Dessert', 'Japanese', 'Indian', 'French',
       'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai'])


st.subheader("How well do you want the dish to be?")
val = st.slider("From poor to the best!",0,10)

num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

food = pd.read_csv("input/food.csv")
ratings = pd.read_csv("input/ratings.csv")
combined = pd.merge(ratings, food, on='Food_ID')


ans = combined.loc[(combined.C_Type == cuisine) & (combined.Veg_Non == vegn)& (combined.Rating >= val),['Name','C_Type','Veg_Non']]
names = ans['Name'].tolist()
x = np.array(names)
ans1 = np.unique(x)

finallist = ""
bruh = st.checkbox("Choose your Dish")
if bruh == True:
    finallist = st.selectbox("Our Choices",ans1)


dataset = ratings.pivot_table(index='Food_ID',columns='User_ID',values='Rating')
dataset.fillna(0,inplace=True)
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_recommendation(Food_Name):
    n = num_recommendations
    FoodList = food[food['Name'].str.contains(Food_Name)]  
    if len(FoodList):        
        Foodi= FoodList.iloc[0]['Food_ID']
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Foodi],n_neighbors=n+1)    
        Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            Recommendations.append({'Name':food.iloc[i]['Name'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df['Name']
    else:
        return "No Similar Foods."


display = food_recommendation(finallist)

if bruh == True:
    bruh1 = st.checkbox(f"We also recommend {num_recommendations} more option: ")
    if bruh1 == True:
        for i in display:
            st.write(i)
