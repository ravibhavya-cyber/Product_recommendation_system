from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

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
    print(f"Error loading data: {e}.  Make sure the files exist and the paths are correct.")
    raise # Re-raise the exception to halt the app if data loading fails

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


@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        algorithm = request.form['algorithm']
        user_id = request.form.get('user_id')  # Use .get() in case user_id is not provided
        N = int(request.form['num_recommendations'])


        if algorithm == 'rank_based':
            recommendations = get_rank_based_recommendations(N)
        elif algorithm == 'user_based_cf':
            if user_id:
                recommendations = get_user_based_cf_recommendations(user_id, N)
            else:
                recommendations = "Please enter a User ID for Collaborative Filtering."
        else:
            recommendations = "Invalid algorithm selected."


    return render_template('index.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
