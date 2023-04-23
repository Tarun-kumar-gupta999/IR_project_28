from flask import Flask,render_template,request
app=Flask(__name__)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all' #this helps to full output and not only the last lines of putput
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.simplefilter('ignore')

df = pd.read_csv(r"D:\IIIT_delhi\2_Sem\Information_Retrieval\Mid project review\dataset.csv")
print(df.head())
df_copy=df.copy()

def collab(given_user,name):
    print("t1")
    data = df.loc[df["category"] == df.loc[df['Product Name']==name]['category'].values[0]]
    print("t2")
    print(df_copy)
    ratings = data.pivot(index='UserId', columns='Product Name', values='Rating').fillna(0)
    ratings['user_index'] = np.arange(0, ratings.shape[0], 1)
    ratings.set_index(['user_index'], inplace=True)
    from scipy.sparse.linalg import svds
    from scipy.sparse import csr_matrix
    # SVD
    s_ratings = csr_matrix(ratings)
    u, s, vT = svds(ratings.values, k=11)
    sigma = np.diag(s)
    ratings_pred = np.matmul(np.matmul(u, sigma), vT)
    pred_data = pd.DataFrame(ratings_pred, columns=ratings.columns, index=ratings.index)
    final_rating = df_copy.copy()
    user_similarity = {}
    given_user = 3
    given_user_vector = pred_data.iloc[given_user]
    given_user_vector_norm = np.linalg.norm(given_user_vector)
    for index,row in pred_data.iterrows():
        current_user_vector = pred_data.iloc[index]
        cosine_similarity = np.dot(given_user_vector,current_user_vector) / (given_user_vector_norm * np.linalg.norm(current_user_vector))
        user_similarity[index] = cosine_similarity
    user_similarity_df = pd.DataFrame.from_dict(user_similarity,orient='index',columns=['similarity_score'])
    user_similarity_df['user_id'] = user_similarity_df.index
    user_similarity_df = user_similarity_df.sort_values(by='similarity_score',ascending=False)
    user_similarity_df_threshold = user_similarity_df.query('similarity_score > 0.2')
    similar_user_vectors = pred_data.iloc[user_similarity_df_threshold['user_id'].values]
    pt=similar_user_vectors.transpose()
    idx = 5
    user_id_random = ratings.index[idx]
    pred_ratings_user = pred_data.iloc[idx,:][ratings.iloc[idx,:]==0].sort_values(ascending=False)
    pred_ratings_user = pred_ratings_user.reset_index()
    pred_ratings_user.rename(columns={'ProductID':'Product ID','U1092':'Predicted Rating'}, inplace=True)
    res=pred_ratings_user.head()
    filtered_rating = df_copy.copy()
    filtered_rating.groupby('ProductId').count()['Rating']
    y = filtered_rating.groupby('Product Name').count()['Rating'] >= 1
    famous_product = y[y].index
    final_rating = filtered_rating[filtered_rating['Product Name'].isin(famous_product)]
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_score = cosine_similarity(pt)
    df2 = pd.DataFrame(similarity_score)
    from sklearn.metrics.pairwise import euclidean_distances
    similarity_score_2 = euclidean_distances(pt)
    df3 = pd.DataFrame(similarity_score_2)
    re = []
    index = np.where(pt.index == name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    print("Product found similar to", end=' ')
    print(name, end=' ')
    print("are : ")
    for i in similar_items:
        re.append(pt.index[i[0]])
        print(pt.index[i[0]])
    final_rating['Product Name'].unique()

    return re

# def data():
#     return (final_rating['Product Name'].unique())
import random

def popularity():
    num_rating_df = df_copy.groupby('Product Name').count()['Rating'].reset_index()
    num_rating_df.rename(columns={'Rating': 'num_rating'}, inplace=True)

    avg_rating_df = df_copy.groupby('Product Name').mean()['Rating'].reset_index()
    avg_rating_df.rename(columns={'Rating': 'avg_rating'}, inplace=True)
    popular_df = num_rating_df.merge(avg_rating_df, on='Product Name')
    # Displaying top 5 products which have highest average ratings
    popular_df = popular_df[popular_df['num_rating'] >= 1].sort_values('avg_rating', ascending=False)
    res=popular_df['Product Name'].values
    res=list(res)
    random_elements = random.sample(res, 5)
    return(random_elements)



@app.route('/')
def index():
    return render_template('main.html')

@app.route('/yes_submit', methods=['POST'])
def yes_main():
    res=popularity()
    return render_template('popularity.html',r=res)

@app.route('/no_submit', methods=['POST'])
def no_submit():
    return render_template('index.html')

@app.route('/pass', methods=['POST'])
def getvalue():
    name = request.form['name']
    name1=request.form['name1']
    result1=collab(name1,name)
    return render_template('pass.html',n=name,r1=result1)



if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0')


