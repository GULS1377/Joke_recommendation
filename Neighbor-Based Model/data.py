import pandas as pd
class Data(dict):
    def __init__(self, data,user_avg):
        self.data= data
        self.user_avg = user_avg

def readData():
    filePath ="jesterfinal151cols.xls"
    df=pd.read_excel(filePath,encoding="utf-8")
    data_jokes = {}
    data_avg={}
    for i in range(5000):
        data_jokes.setdefault(i, {})
        count=0
        score_sum=0
        for j in range(1,151):
            data_jokes[i][j] = df.at[i, j]
            if(data_jokes[i][j]!=99):
                count+=1
                score_sum+=data_jokes[i][j]
        data_avg[i]=score_sum/count
    # print(data_avg)
    print("training data read finish!")
    return Data(data=data_jokes,user_avg=data_avg)

def readTest():
    filePath="test.xlsx"
    df = pd.read_excel(filePath, encoding="utf-8")
    data_jokes = {}
    data_avg = {}
    for i in range(len(df)):
        data_jokes.setdefault(i, {})
        count = 0
        score_sum = 0
        for j in range(1, 151):
            data_jokes[i][j] = df.at[i, j]
            if (data_jokes[i][j] != 99):
                count += 1
                score_sum += data_jokes[i][j]
        data_avg[i] = score_sum / count
    # print(data_avg)
    print("test data read finish!")
    return Data(data=data_jokes, user_avg=data_avg)

def read_Text():
    filePath = "joke_text.txt"
    f=open(filePath,'r')
    jokes=f.readlines()
    for i in range(150):
        jokes[i]=jokes[i][0:len(jokes[i])-1]
    return jokes

if __name__ == '__main__':
    read_Text()