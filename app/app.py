from flask import Flask, render_template, request, redirect, url_for
from gensim.models import FastText
import pandas as pd
import pickle
import os
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(16) 

#load the datase
df = pd.read_csv('assignment3_II.csv')
# Load saved models and vectorizer
ft_model = FastText.load('fasttext.model')
with open('logistic_regression_model_tfidf_fasttext.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('tVectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


departments = df[['Department Name', 'Class Name']].drop_duplicates()

# Get unique departments
unique_departments = departments['Department Name'].unique()

# Create a dictionary mapping each department to its corresponding class names
department_classes = {}
for department in unique_departments:
    department_classes[department] = departments[departments['Department Name'] == department]['Class Name'].unique().tolist()


# Function to vectorize a single review
def vectorize_text(text, model, vectorizer):
    words = text.split()
    tfidf_weights = vectorizer.transform([text]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    weighted_vectors = []
    for word in words:
        if word in feature_names:
            try:
                weighted_vectors.append(model.wv[word] * tfidf_weights[feature_names.tolist().index(word)])
            except KeyError:
                pass
    if len(weighted_vectors) > 0:
        return sum(weighted_vectors) / len(weighted_vectors)
    else:
        return np.zeros(model.vector_size)

# Home page
@app.route('/')
def index():
    return render_template('home.html', department_classes=department_classes)

# Types of clothes
@app.route('/class/<class_name>')
def class_page(class_name):
    # Filter the dataset for items belonging to the selected class
    class_items = df[df['Class Name'] == class_name][['Clothing ID', 'Title', 'Review Text']].to_dict(orient='records')

    # Pass the items and class name to the template
    return render_template('class.html', class_name=class_name, items=class_items, department_classes=department_classes)

# Search functionality
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        keyword = request.form['keyword'].lower()
    else:
        keyword = request.args.get('keyword', '').lower()

    # Split the search query into multiple words (if applicable)
    search_terms = keyword.split()


    try:
        similar_words = [word for word, score in ft_model.wv.most_similar(keyword, topn=10)]
        similar_words.append(keyword)
    except KeyError:
        similar_words = [keyword]

    # Update the matching logic to check if any of the search terms are present in any of the fields
    matching_items = df[
        df['Class Name'].apply(lambda x: any(term in x.lower() for term in search_terms)) |
        df['Department Name'].apply(lambda x: any(term in x.lower() for term in search_terms)) |
        df['Division Name'].apply(lambda x: any(term in x.lower() for term in search_terms)) |
        df['Title'].apply(lambda x: any(term in x.lower() for term in search_terms))
    ]

    # Convert matching items to dict format
    items = matching_items.to_dict(orient='records')

    # Ensure department_classes is passed to the template
    return render_template('search_results.html', items=items, keyword=keyword, department_classes=department_classes)

@app.context_processor
def inject_departments():
    return {'department_classes': department_classes}


@app.route('/item/<clothing_id>')
def item_detail(clothing_id):
    # Get the specific item based on its Clothing ID
    item = df[df['Clothing ID'] == int(clothing_id)].to_dict(orient='records')[0]

    # Get the previous search keyword (if available)
    keyword = request.args.get('keyword', '')  # Default to empty string if no keyword is passed

    # Render the item detail page with the keyword
    return render_template('item_detail.html', item=item, keyword=keyword)

@app.route('/review', methods=['GET', 'POST'])
def review():
    global df  # Global dataframe to store reviews
    
    # Extract unique values from the dataset for dropdowns
    division_names = df['Division Name'].unique()
    department_names = df['Department Name'].unique()
    class_names = df['Class Name'].unique()
    
    # Initialize variables to None or empty string for GET requests
    title = ''
    description = ''
    rating = ''
    model_recommendation = ''
    division_name = ''
    department_name = ''
    class_name = ''
    
    if request.method == 'POST':
        # Get form data
        title = request.form['title']
        description = request.form['description']
        rating = request.form['rating']
        division_name = request.form['division_name']
        department_name = request.form['department_name']
        class_name = request.form['class_name']
        recommendation = request.form['recommendation']  # This will be user modified or model output

        # If the user requests a recommendation from the model
        if request.form['action'] == 'get_recommendation':
            # Combine the title and description for vectorization
            review_text = title + " " + description
            vectorized_review = vectorize_text(review_text, ft_model, tfidf_vectorizer)
            vectorized_review = vectorized_review.reshape(1, -1)  # Reshape for prediction
            
            # Predict the recommendation label (0 or 1)
            model_recommendation = lr_model.predict(vectorized_review)[0]
            return render_template('create_review.html', 
                                   title=title, description=description, rating=rating, 
                                   recommendation=model_recommendation, 
                                   division_names=division_names,
                                   department_names=department_names,
                                   class_names=class_names,
                                   selected_division_name=division_name,
                                   selected_department_name=department_name,
                                   selected_class_name=class_name)
        
        # If the user confirms the review
        elif request.form['action'] == 'submit_review':
            # Generate a unique Clothing ID based on the maximum Clothing ID in the current dataset
            if df['Clothing ID'].dtype == object:
                existing_ids = pd.to_numeric(df['Clothing ID'], errors='coerce').dropna()
            else:
                existing_ids = df['Clothing ID']
            new_clothing_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1206  # Start from 1206 if no IDs

            # Lookup Clothes Title and Description based on Division, Department, and Class
            clothes_match = df[
                (df['Division Name'] == division_name) & 
                (df['Department Name'] == department_name) & 
                (df['Class Name'] == class_name)
            ]
            
            if not clothes_match.empty:
                clothes_title = clothes_match['Clothes Title'].iloc[0]  
                clothes_description = clothes_match['Clothes Description'].iloc[0]  
            else:
                clothes_title = "Unknown Title"
                clothes_description = "No Description Available"


            # Save the review to the dataset
            new_review = pd.DataFrame([{
                'Clothing ID': new_clothing_id,  
                'Clothes Title': clothes_title,  
                'Clothes Description': clothes_description, 
                'Title': title ,
                'Review Text': description,
                'Rating': rating,
                'Division Name': division_name,
                'Department Name': department_name,
                'Class Name': class_name,
                'Recommended IND': recommendation
            }])
            
            df = pd.concat([df, new_review], ignore_index=True)  # Append the new review to the dataframe
            # Redirect to class page to display reviews for the selected class
            return redirect(url_for('class_page', class_name=class_name))

    # Render the form for GET requests
    return render_template('create_review.html', department_classes=department_classes, title=title, description=description, 
                           rating=rating, recommendation=model_recommendation, division_names=division_names, 
                           department_names=department_names, class_names=class_names,
                           selected_division_name=division_name, selected_department_name=department_name, selected_class_name=class_name)



if __name__ == '__main__':
    app.run(debug=True)