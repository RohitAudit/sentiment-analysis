import panda as pd
import numpy as np
import glob,os

q = []
for file in os.listdir("c:/users/737ro/IItb/movie_reviews/movie_reviews/pos"):
	if file.endswith(".txt"):
		p = os.path.join(file)
		q.append(p)
		r = open("file","r+")

Review = pd.DataFrame({})

for i in q:
	