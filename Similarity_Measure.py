#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.spatial.distance import cosine,euclidean
from scipy.stats import pearsonr


# In[5]:


# sample user rating data
user1 = np.array([4,5,2,3,4])
user2 = np.array([5,3,2,4,5])


# In[7]:


cosine_similarity = 1 - cosine(user1,user2)
print(f"Cosine similarity:{cosine_similarity:.4f}")


# In[11]:


pearson_corr,_ = pearsonr(user1,user2)
print(f"pearson correlarion similarity:{pearson_corr:.4f}")


# In[15]:


euclidean_distance = euclidean(user1,user2)
euclidean_similarity = 1/(1+euclidean_distance)
print(f"Euclidean distance similarity:{euclidean_similarity:.4f}")


# In[17]:


raju = np.array([5,3,4,4,2])
john = np.array([3,1,2,3,3])


# In[19]:


cosine_similarity = 1 - cosine(raju,john)
print(f"Cosine similarity:{cosine_similarity:.4f}")


# In[21]:


euclidean_distance = euclidean(raju,john)
euclidean_similarity = 1/(1+euclidean_distance)
print(f"Euclidean distance similarity:{euclidean_similarity:.4f}")


# In[23]:


pearson_corr,_ = pearsonr(raju,john)
print(f"pearson correlarion similarity:{pearson_corr:.4f}")


# In[25]:


raju = np.array([5,3,4,4,2])
ramya = np.array([4,3,4,5,1])


# In[27]:


cosine_similarity = 1 - cosine(raju,ramya)
print(f"Cosine similarity:{cosine_similarity:.4f}")


# In[29]:


pearson_corr,_ = pearsonr(raju,ramya)
print(f"pearson correlarion similarity:{pearson_corr:.4f}")


# In[31]:


euclidean_distance = euclidean(raju,ramya)
euclidean_similarity = 1/(1+euclidean_distance)
print(f"Euclidean distance similarity:{euclidean_similarity:.4f}")


# In[35]:


import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

#User-item rating matrix for 4 users
ratings = np.array([
    [5, 3, 4, 4, 2],  # User A
    [3, 1, 2, 3, 3],  # User B
    [4, 3, 4, 5, 1],  # User C
    [2, 2, 1, 2, 4]   # User D
])

users = ["Raju", "John", "Ramya", "Kishore"]
df = pd.DataFrame(ratings, index=users, columns=["Bahubali","Mufasa","Interstellar","RRR","Mrs"])
print(df)

# Function to compute similarity
def compute_similarity(df):
    num_users = df.shape[0]
    similarity_results = []

    for i in range(num_users):
        for j in range(i + 1, num_users):  # Avoid redundant pairs
            user1, user2 = df.iloc[i], df.iloc[j]

            # Cosine Similarity
            cos_sim = 1 - cosine(user1, user2)

            # Pearson Correlation Similarity
            pearson_sim, _ = pearsonr(user1, user2)

            # Euclidean Distance Similarity
            euc_dist = euclidean(user1, user2)
            euc_sim = 1 / (1 + euc_dist)  # Normalize to [0,1]

            similarity_results.append([users[i], users[j], cos_sim, pearson_sim, euc_sim])

    return pd.DataFrame(similarity_results, columns=["User 1", "User 2", "Cosine Similarity", "Pearson Correlation", "Euclidean Similarity"])

# Compute similarity matrix
similarity_df = compute_similarity(df)

# Display results
print(similarity_df)


# In[ ]:




