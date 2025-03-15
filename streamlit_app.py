import streamlit as st
import pandas as pd
import numpy as np


# Load pre-calculated data (adjust file paths as needed)
try:
    # Load dataframes and matrices needed for recommendations
    # Example data loading (adjust to your actual file paths and formats)
    product_avg_ratings = pd.read_csv('product_avg_ratings.csv') # For rank-based
    user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)  # For CF
    predicted_ratings_matrix = pd.read_csv('predicted_ratings_matrix.csv', index_col=0)

    # Ensure correct data types, especially for user_id or user_index
    user_item_matrix.index = user_item_matrix.index.astype(str)  # or int, if appropriate
    predicted_ratings_matrix.index = predicted_ratings_matrix.index.astype(str)


except FileNotFoundError as e:
    st.error(f"Error loading data: {e}.  Make sure the files exist and the paths are correct.")
    st.stop() # Stop the app if data loading fails

def get_rank_based_recommendations(N=10):
    """Returns top N products based on average rating and rating count."""
    top_products = product_avg_ratings.sort_values(by=['avg_rating', 'rating_count'], ascending=[False, False]).head(N)
    return top_products['prod_id'].tolist()


def get_user_based_cf_recommendations(user_id, N=10):
    """Returns top N products for a given user ID using collaborative filtering."""
    try:
        user_index = user_item_matrix.index.get_loc(str(user_id)) # Find user index
        # Get predicted ratings for the user
        user_ratings = predicted_ratings_matrix.iloc[user_index]

        # Get the indices of the top N predicted ratings
        top_indices = user_ratings.nlargest(N).index

        #Convert the indices into product IDs
        top_products = top_indices.tolist()
        return top_products

    except KeyError:
        return f"User ID {user_id} not found."
    except Exception as e:
        return f"An error occurred: {e}"

st.title("Product Recommendation Engine")

algorithm = st.selectbox("Select Algorithm:",
                        ["Rank-Based", "User-Based Collaborative Filtering"])

num_recommendations = st.slider("Number of Recommendations:", 1, 20, 10) # Min, Max, Default

user_id = None
if algorithm == "User-Based Collaborative Filtering":
    user_id = st.text_input("Enter User ID:")

if st.button("Get Recommendations"):
    if algorithm == "Rank-Based":
        recommendations = get_rank_based_recommendations(num_recommendations)
        st.write("Recommendations:")
        st.write(recommendations)  # Or format it better
    elif algorithm == "User-Based Collaborative Filtering":
        if user_id:
            recommendations = get_user_based_cf_recommendations(user_id, num_recommendations)
            st.write("Recommendations:")
            st.write(recommendations)
        else:
            st.warning("Please enter a User ID for Collaborative Filtering.")
