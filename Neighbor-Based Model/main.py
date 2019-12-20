import data
from similarities import ItemSimilarity,UserSimilarity,TextSimilarity

if __name__ == '__main__':
    # read data
    text_data = data.read_Text()
    text_similarity=TextSimilarity(text_data)

    user_rating=data.readData()
    test_data=data.readTest()
    text_similarity.tf_idf(test_data.data)

    text_similarity.cosine(test_data.data)
    text_similarity.w2v(test_data.data)


    # compute item similarity
    item_similarity = ItemSimilarity(user_rating,M=5)

    # compute item-based rating prediction
    # item_similarity.getPrediction()

    # compute user similarity
    user_similarity = UserSimilarity(user_rating,item_similarity.user_predict,k=500)
    user_similarity.getAvg(test_data.data)
    user_similarity.findCluster(test_data.data)


    # get best combination of w and w_ for hybrid user-based and item-based model

    # evaluate
    # min=9999999
    # record1=-1
    # record2=-1
    # for i in range(1,10):
    #     w=i*0.1
    #     w_=1-w
    #     for j in range(1,10):
    #         n = 0
    #         MAE = 0
    #         user_similarity.setR(j*0.1)
    #         item_similarity.setR(j*0.1)
    #         for k in range(len(test_data.data)):
    #             for m in range(1, 151):
    #                 if (test_data.data[k][m] != 99):
    #                     n += 1
    #                     res=w*item_similarity.predict(k,m,test_data.data)+w_*user_similarity.predict(k,m)
    #                     MAE += abs(test_data.data[k][m] - res)
    #         print("i="+str(i)+", j="+str(j)+" MAE result: ",MAE/n)
    #         if(MAE/n<min):
    #             min=MAE/n
    #             record1=i*0.1
    #             record2=0.1*j
    # print(min,record1,record2)


    ww=0.1
    j=9
    user_similarity.setR(j * 0.1)
    item_similarity.setR(j * 0.1)

    min_1=9999999
    record1=-1
    min_2 = 9999999
    record2 = -1
    min_3 = 9999999
    record3 = -1
    for i in range(1,10):
        w=i*0.1
        w_=1-w

        n = 0
        MAE_tf = 0
        MAE_cosine=0
        MAE_w2v=0
        for k in range(len(test_data.data)):
            for m in range(1, 151):
                if (test_data.data[k][m] != 99):
                    n += 1
                    res = w*(ww * item_similarity.predict(k, m, test_data.data) + (1-ww) * user_similarity.predict(k, m))
                    MAE_tf += abs(test_data.data[k][m] - res-w_*text_similarity.text_predict[k][m])
                    MAE_cosine += abs(test_data.data[k][m] - res - w_ * text_similarity.text_predict_2[k][m])
                    MAE_w2v += abs(test_data.data[k][m] - res - w_ * text_similarity.text_predict_3[k][m])
        print("i="+str(i)+" MAE_tf result: ",MAE_tf/n)
        print("i=" + str(i) + " MAE_cosine result: ", MAE_cosine / n)
        print("i=" + str(i) + " MAE_w2v result: ", MAE_w2v/ n)
        if(MAE_tf/n<min_1):
            min_1=MAE_tf/n
            record1=i*0.1
        if (MAE_cosine / n < min_2):
            min_2 = MAE_cosine / n
            record2 = i * 0.1
        if (MAE_w2v / n < min_3):
            min_3 = MAE_w2v / n
            record3 = i * 0.1
    print(min_1,record1)
    print(min_2, record2)
    print(min_3, record3)