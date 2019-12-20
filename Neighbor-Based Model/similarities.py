import gensim
import numpy as np
import spacy
from sklearn.cluster import KMeans
import copy
import sys
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec

class ItemSimilarity(object):
    def __init__(self, data, M = 10,R=1):
        self.data = data
        self.clique=[]
        self.sim = {}
        self.calculateItemSim()
        self.user_predict=copy.deepcopy(data.data)
        self.M=M
        self.R=R
        self.record=[[0 for i in range(151)] for j in range(len(self.data.data))]

    def setR(self,R):
        self.R=R

    # calculate item similarity
    # input: self.data
    # output: self.sim, self.clique
    def calculateItemSim(self):
        for i in range(1, 151):
            self.sim.setdefault(i, {})
            list1 = []
            for j in range(1, 151):
                if (i == j):
                    continue
                count1 = 0
                count2 = 0
                count3 = 0
                for u in range(0, len(self.data.data)):
                    Rui = self.data.data[u][i]
                    Ruj = self.data.data[u][j]
                    if (Rui != 99 and Ruj != 99):
                        count1 += (Ruj - self.data.user_avg[u]) * (Rui - self.data.user_avg[u])
                        count2 += (Ruj - self.data.user_avg[u]) * (Ruj - self.data.user_avg[u])
                        count3 += (Rui - self.data.user_avg[u]) * (Rui - self.data.user_avg[u])
                if (count2 != 0 and count3 != 0):
                    self.sim[i][j] = count1 / ((count2 ** 0.5) * (count3 ** 0.5))
                    list2 = [count1 / ((count2 ** 0.5) * (count3 ** 0.5)), j]
                    list1.append(list2)
                else:
                    self.sim[i][j] = -999999999999
            print(i," item-sim finish")
            self.clique.append(list1)

    # getPrediction on matrix
    # input: self.data, self.clique, self.sim
    # output: self.user_predict, self.record
    def getPrediction(self):
        sum_missing=0
        sum_filling=0
        for i in range(150):
            self.clique[i].sort(key=lambda x: x[0])
            if(len(self.clique[i])>0):
                for user_id in range(0,len(self.data.data)):
                    cnt = 0
                    count1 = 0
                    count2 = 0
                    if(self.data.data[user_id][i+1]!=99):
                        continue
                    sum_missing+=1
                    for j in range(len(self.clique[i])):
                        index=self.clique[i][j][1]
                        if(self.data.data[user_id][index]!=99):
                            count1+=self.sim[i+1][index]*self.data.data[user_id][index]
                            count2+=abs(self.sim[i+1][index])
                            cnt+=1
                        if(cnt==self.M):
                            break
                    if(count2==0):
                        continue
                    else:
                        self.user_predict[user_id][i+1]=count1/count2
                        self.record[user_id][i+1]=1
                        sum_filling +=1
                        # print(user_id,i+1,count1/count2)
            else:
                for t in range(0,len(self.data.data)):
                    self.record[t][i+1]=-1
        print("smooth missing values finish!")

    # evaluate with MAE
    def evaluate(self,data):
        n=0
        MAE=0
        for i in range(len(data)):
            for j in range(1,151):
                if(data[i][j]!=99):
                    n+=1
                    MAE+=abs(data[i][j]-self.predict(i,j,data))
        return MAE/n

    # get prediction of user i on item j with data matrix
    def predict(self,user_id,item_id,data):
        count1=0
        count2=0
        cnt=0
        for k in range(len(self.clique[item_id-1])):
            index = self.clique[item_id-1][k][1]
            if (data[user_id][index] == 99):
                count1 += 0
                count2 += abs((1-self.R) * self.sim[item_id][index])
            else:
                count1 += self.R * self.sim[item_id][index] * data[user_id][index]
                count2 += abs(self.R * self.sim[item_id][index])
            cnt += 1
            if (cnt == self.M):
                break
        if(count2==0):
            return 0
        return count1 / count2

class UserSimilarity(object):
    def __init__(self, data, user_predict,k=5,R=1):
        self.data = data
        self.k=k
        self.R=R
        self.cluster={}
        self.centroids=[]
        self.inilizeKmeans()
        self.user_predict = user_predict
        self.cluster_Result=[]
        self.avg_Result=[]

    def setR(self,R):
        self.R=R

    def inilizeKmeans(self):
        arr = np.zeros((len(self.data.data), 150))
        for i in range(len(self.data.data)):
            for j in range(1,151):
                arr[i][j-1]=self.data.data[i][j]
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(arr)
        for i in range(self.k):
            self.cluster.setdefault(i, [])
        for i in range(len(self.data.data)):
            self.cluster[kmeans.labels_[i]].append(i)
        self.centroids=kmeans.cluster_centers_

    def getAvg(self,data):
        avg_Result = []
        print("getAvg start")
        for i in range(len(data)):
            sum = 0
            count = 0
            for j in range(1, 151):
                if (data[i][j] != 99):
                    count += 1
                    sum += data[i][j]
            if count==0:
                avg_Result.append(0)
            else:
                avg_Result.append(sum / count)
        self.avg_Result=avg_Result
        print("getAvg finish!")

    # find cluster belonging
    def findCluster(self,data):
        print("findCluster start!")
        result=[]
        for i in range(len(data)):
            index = -1
            min = sys.float_info.max
            for j in range(len(self.centroids)):
                dis=0
                for k in range(1,151):
                    diff=(data[i][k]-self.centroids[j][k-1])
                    dis+=diff*diff
                if(dis<min):
                    min=dis
                    index=j
            result.append(index)
        self.cluster_Result=result
        print("findCluster finish!")

    # get prediction of user i on item j with data matrix
    def predict(self,user_id,item_id):
        count1=0
        count2=0
        for k in self.cluster[self.cluster_Result[user_id]]:
            sim=self.getSim(user_id,k,self.avg_Result)
            if(self.user_predict[k][item_id]==99):
                count1+=(1-self.R)*sim*(0-self.data.user_avg[k])
                count2+=abs((1-self.R)*sim)
            else:
                count1 += (self.R) * sim * (self.user_predict[k][item_id]-self.data.user_avg[k])
                count2 += abs((self.R) * sim)
        if(count2==0):
            return self.avg_Result[user_id]
        return self.avg_Result[user_id] + count1 / count2

    def getSim(self,user_id,candidate_id,avg_Result):
        count1=0
        count2=0
        count3=0
        for item_id in range(1,151):
            Rak=self.user_predict[user_id][item_id]
            Ruk=self.user_predict[candidate_id][item_id]
            if(Rak!=99 and Ruk!=99):
                count1+=(Rak-avg_Result[user_id])*self.R*(Ruk-self.data.user_avg[candidate_id])
                count2+=(Rak-avg_Result[user_id])*(Rak-avg_Result[user_id])
                count3+=(self.R*(Ruk-self.data.user_avg[candidate_id]))*(self.R*(Ruk-self.data.user_avg[candidate_id]))
        if(count2==0 or count3==0):
            return 0
        return count1/((count2 ** 0.5)*(count3 ** 0.5))


class TextSimilarity(object):
    def __init__(self, data):
        self.data=data
        self.joke_sim = {}
        self.joke_sim_2 = {}
        self.joke_sim_3 = {}
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        self.vectorizer = TfidfVectorizer(tokenizer=self.normalize, stop_words='english')
        self.predict()
        self.text_predict={}
        self.text_predict_2 = {}
        self.text_predict_3 = {}

    def stem_tokens(self,tokens):
        return [self.stemmer.stem(item) for item in tokens]

    def normalize(self,text):
        return self.stem_tokens(nltk.word_tokenize(text.lower().translate(self.remove_punctuation_map)))

    def predict(self):
        print("enter predict")
        for i in range(150):
            self.joke_sim.setdefault(i,{})
            for j in range(150):
                if i==j:
                    self.joke_sim[i][j]=1
                else:
                    tfidf = self.vectorizer.fit_transform([self.data[i], self.data[j]])
                    self.joke_sim[i][j] = ((tfidf * tfidf.T).A)[0, 1]
            print(i,"finish")
        print("tf-idf finish!")

        nlp = spacy.load('en')
        for i in range(150):
            self.joke_sim_2.setdefault(i, {})
            for j in range(150):
                if i == j:
                    self.joke_sim_2[i][j] = 1
                else:
                    doc1 = nlp(self.data[i])
                    doc2 = nlp(self.data[j])
                    self.joke_sim_2[i][j] = np.dot(doc1.vector, doc2.vector) / (np.linalg.norm(doc1.vector) * np.linalg.norm(doc2.vector))
                print(i,j)
        print("cosine finish!")

        model = Doc2Vec.load("d2v.model")
        for i in range(150):
            self.joke_sim_3.setdefault(i, {})
            for j in range(150):
                if i == j:
                    self.joke_sim_3[i][j] = 1
                else:
                    self.joke_sim_3[i][j] = model.docvecs.n_similarity(str(i), str(j))
        print("w2v finish")

    def tf_idf(self,test_data):
        for i in range(len(test_data)):
            self.text_predict.setdefault(i,{})
            for j in range(150):
                if(test_data[i][j+1]!=99):
                    count1 = 0
                    count2 = 0
                    for k in range(150):
                        if(test_data[i][k+1]!=99 and self.joke_sim[j][k]>0 and j!=k):
                            count1+=self.joke_sim[j][k]*test_data[i][k+1]
                            count2+=abs(self.joke_sim[j][k])
                    if count2==0:
                        self.text_predict[i][j + 1] =0
                        continue
                    self.text_predict[i][j+1]=count1/count2
        print("text predict finish!")

    def cosine(self,test_data):
        for i in range(len(test_data)):
            self.text_predict_2.setdefault(i, {})
            for j in range(150):
                if (test_data[i][j + 1] != 99):
                    count1 = 0
                    count2 = 0
                    for k in range(150):
                        if (test_data[i][k + 1] != 99 and self.joke_sim_2[j][k] > 0 and j != k):
                            count1 += self.joke_sim_2[j][k] * test_data[i][k + 1]
                            count2 += abs(self.joke_sim_2[j][k])
                    if count2 == 0:
                        self.text_predict_2[i][j + 1] = 0
                        print(test_data[i][j + 1], self.text_predict_2[i][j + 1])
                        continue
                    self.text_predict_2[i][j + 1] = count1 / count2
                    print(test_data[i][j + 1], self.text_predict_2[i][j + 1])
        print("cosine predict finish!")

    def w2v(self,test_data):
        for i in range(len(test_data)):
            self.text_predict_3.setdefault(i, {})
            for j in range(150):
                if (test_data[i][j + 1] != 99):
                    count1 = 0
                    count2 = 0
                    for k in range(150):
                        if (test_data[i][k + 1] != 99 and self.joke_sim_3[j][k] > 0 and j != k):
                            count1 += self.joke_sim_3[j][k] * test_data[i][k + 1]
                            count2 += abs(self.joke_sim_3[j][k])
                    if count2 == 0:
                        self.text_predict_3[i][j + 1] = 0
                        print(test_data[i][j + 1], self.text_predict_3[i][j + 1])
                        continue
                    self.text_predict_3[i][j + 1] = count1 / count2
                    print(test_data[i][j + 1], self.text_predict_3[i][j + 1])
        print("w2v predict finish!")








