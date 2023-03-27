from flask import Flask,render_template,request
app=Flask(__name__)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all' #this helps to full output and not only the last lines of putput
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

df = pd.read_csv(r"D:\Downloads\final dataset_CB.csv")
print(df.head())
df_copy = df.loc[df["category"] == "Electronics"]
df_copy
df_copy.info()
print(df_copy.head())
num_rating_df = df_copy.groupby('Product Name').count()['Rating'].reset_index()
num_rating_df.rename(columns={'Rating':'num_rating'},inplace=True)
#Average rating of all products.
avg_rating_df = df_copy.groupby('Product Name').mean()['Rating'].reset_index()
avg_rating_df.rename(columns={'Rating':'avg_rating'},inplace=True)
avg_rating_df
popular_df = num_rating_df.merge(avg_rating_df,on='Product Name')
popular_df
#Displaying top 5 products which have highest average ratings
popular_df = popular_df[popular_df['num_rating']>=1].sort_values('avg_rating',ascending=False).head(5)
popular_df


result=popular_df['Product Name'].values
print(result)


#Content based

#Grouping data based on user ID and finding number of ratings given by each user
unum_rating_df = df_copy.groupby('UserId').count()['Rating'].reset_index()
unum_rating_df.rename(columns={'Rating':'unum_rating'},inplace=True)
#consider only those users which have been given number of ratings more than or equal to 1
x = df_copy.groupby('UserId').count()['Rating']>=1
good_users = x[x].index
good_users
filtered_rating = df_copy[df_copy['UserId'].isin(good_users)]
filtered_rating
#Count the number of ratings for each product
filtered_rating.groupby('Product Name').count()['Rating']
#Display all the products which have recieved more than 1 ratings
y = filtered_rating.groupby('Product Name').count()['Rating'] >= 1
famous_product = y[y].index
famous_product
final_rating = filtered_rating[filtered_rating['Product Name'].isin(famous_product)]
#Plot the pivot table with product Name as index and user ID as columns names
pt = final_rating.pivot_table(index='Product Name',columns='UserId',values='Rating')
pt
# fill missing values with zeros
pt.fillna(0,inplace=True)
pt
# measure using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_score = cosine_similarity(pt)
df2 = pd.DataFrame(similarity_score)
df2
# measure using euclidean distance
from sklearn.metrics.pairwise import euclidean_distances
similarity_score_2 = euclidean_distances(pt)
df3 = pd.DataFrame(similarity_score_2)
df3
# recommend using cosine similarity
import  numpy as np
def recommend(ProductID):
    re=[]
    index = np.where(pt.index==ProductID)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    print("Restaurants found similar to",end=' ')
    print(ProductID,end=' ')
    print("are : ")
    for i in similar_items:
        re.append(pt.index[i[0]])
        print(pt.index[i[0]])
    return re
# recommend using euclidean distance
def recommend2(ProductID):
    re=[]
    index = np.where(pt.index==ProductID)[0][0]
    similar_items = sorted(list(enumerate(similarity_score_2[index])),key=lambda x:x[1],reverse=True)[1:6]
    print("Restaurants found similar to",end=' ')
    print(ProductID,end=' ')
    print("are : ")
    for i in similar_items:
        re.append(pt.index[i[0]])
        print(pt.index[i[0]])
    return re
final_rating['Product Name'].unique()

#Collaborative

data=df_copy.copy()
#check details
data.info()
# check null
print(data.isnull().sum())
ratings = data.pivot(index = 'UserId', columns = 'Product Name', values = 'Rating').fillna(0)
ratings
import numpy as np
ratings['user_index'] = np.arange(0, ratings.shape[0],1)
ratings.set_index(['user_index'], inplace = True)
ratings
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
#SVD
s_ratings = csr_matrix(ratings)

u,s,vT = svds(s_ratings,k = 11)
#Construct diagonal array in SVD
sigma = np.diag(s)

print("Shape of U : {}".format(u.shape))
print("Shape of Sigma, S : {}".format(sigma.shape))
print("Shape of VT : {}".format(vT.shape))
#Reconstruct ratings with predicted values using SVD
ratings_pred = np.matmul(np.matmul(u,sigma),vT)
pred_data = pd.DataFrame(ratings_pred, columns = ratings.columns, index = ratings.index)
pred_data
# # Check MSE
a = ratings.values.reshape(-1)
b = ratings_pred.reshape(-1)
mse = np.sum((a-b)**2)/a.shape[0]
print("MSE : {:.4f}".format(mse))
# top ratings predicted for unrated restaurants by some user
def colla(name1):
    idx = name1
    user_id_random = ratings.index[idx]
    pred_ratings_user = pred_data.iloc[idx,:][ratings.iloc[idx,:]==0].sort_values(ascending=False)
    pred_ratings_user = pred_ratings_user.reset_index()
    pred_ratings_user.rename(columns={'ProductID':'Restuarant ID','U1092':'Predicted Rating'}, inplace=True)

    res=pred_ratings_user.head()
    res=res['Product Name'].values
    return(res)






@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def getvalue():
    name=request.form['name']
    # name1 = request.form['name1']
    print(name)
    # print(name1)
    result2=recommend(name)
    result3 = recommend2(name)
    # result4=colla(name1)
    return render_template('pass.html',n=name,r=result,r2=result2,r3=result3)


# re = recommend(resto)

if __name__=='__main__':
    app.run(debug=True)


# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all' #this helps to full output and not only the last lines of putput
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
# warnings.simplefilter('ignore')
#
# df = pd.read_csv(r"D:\Downloads\q - Sheet1 (2).csv")
# df.head()
# df_copy = df.copy()
# # df_copy=df_copy.drop('Unnamed: 0', axis=1)
# # df_copy=df_copy.drop('Timestamp', axis=1)
# df_copy.info()
# df_copy.head()
# df_copy.groupby('ProductId').count()
# num_rating_df = df_copy.groupby('ProductId').count()['Rating'].reset_index()
# num_rating_df.rename(columns={'Rating':'num_rating'},inplace=True)
# num_rating_df
# avg_rating_df = df_copy.groupby('ProductId').mean()['Rating'].reset_index()
# avg_rating_df.rename(columns={'Rating':'avg_rating'},inplace=True)
# avg_rating_df
# popular_df = num_rating_df.merge(avg_rating_df,on='ProductId')
# popular_df
# #We will suggest only those products to users which have recieved number of ratings more than 5
# popular_df[popular_df['num_rating']>=1]
# popular_df[popular_df['num_rating']>=1].sort_values('avg_rating',ascending=False)
# #Displaying top 5 products which have highest average ratings
# popular_df = popular_df[popular_df['num_rating']>=5].sort_values('avg_rating',ascending=False).head(5)
# result=popular_df[0].values()
#





