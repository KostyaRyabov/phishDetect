import feature_extractor as fe

if __name__ == "__main__":
    import pickle
    from tensorflow import keras

    import tensorflow.keras.backend as K


    def f_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    m1 = pickle.load(open('data/models/AdaBoost_DT/AdaBoost_DT.pkl', 'rb'))
    m2 = pickle.load(open('data/models/Bagging_DT/Bagging_DT.pkl', 'rb'))
    m3 = pickle.load(open('data/models/Bernoulli_NB/Bernoulli_NB.pkl', 'rb'))
    m4 = pickle.load(open('data/models/Complement_NB/Complement_NB.pkl', 'rb'))
    m5 = pickle.load(open('data/models/DT/DT.pkl', 'rb'))
    m6 = pickle.load(open('data/models/ET/ET.pkl', 'rb'))
    # m7 = pickle.load(open('data/models/Gaussian_NB', 'rb'))
    m8 = pickle.load(open('data/models/GradientBoost/GradientBoost.pkl', 'rb'))
    m9 = pickle.load(open('data/models/HistGradientBoost/HistGradientBoost.pkl', 'rb'))
    m10 = pickle.load(open('data/models/kNN/2NN.pkl', 'rb'))
    m11 = pickle.load(open('data/models/Multinomial_NB/Multinomial_NB.pkl', 'rb'))
    m12 = keras.models.load_model('data/models/neural_networks/nn1.h5', custom_objects={'f_score': f_score})
    m13 = pickle.load(open('data/models/RF/RF.pkl', 'rb'))
    m14 = pickle.load(open('data/models/Stacking (AdaBoost_DT, ET, DT, Bagging_DT, RF)/StackingClassifier.pkl', 'rb'))
    m15 = pickle.load(open('data/models/Stacking (All)/StackingClassifier.pkl', 'rb'))
    m16 = pickle.load(open('data/models/Stacking (CNB, MNB, BNB, GNB)/StackingClassifier.pkl', 'rb'))
    m17 = pickle.load(open('data/models/Stacking (RF,HGBC, GBC, AdaBoost, ET)/StackingClassifier.pkl', 'rb'))
    m18 = pickle.load(open('data/models/Stacking (SVM, kNN, DT)/StackingClassifier.pkl', 'rb'))
    m19 = pickle.load(open('data/models/SVM/SVM.pkl', 'rb'))

    data = [fe.extract_features('https://www.google.com/search?client=opera&q=tenserflow+load+model&sourceid=opera&ie=UTF-8&oe=UTF-8')]

    print('AdaBoost_DT', m1.predict_proba(data))
    print('Bagging_DT', m2.predict_proba(data))
    print('Bernoulli_NB', m3.predict_proba(data))
    print('Complement_NB', m4.predict_proba(data))
    print('DT', m5.predict_proba(data))
    print('ET', m6.predict_proba(data))
    # print('Gaussian_NB', m7.predict_proba(data))
    print('GradientBoost', m8.predict_proba(data))
    print('HistGradientBoost', m9.predict_proba(data))
    print('2NN', m10.predict_proba(data))
    print('Multinomial_NB', m11.predict_proba(data))
    print('neural_networks', m12.predict(data))
    print('RF', m13.predict_proba(data))
    print('Stacking (AdaBoost_DT, ET, DT, Bagging_DT, RF)', m14.predict_proba(data))
    print('Stacking (All)', m15.predict_proba(data))
    print('Stacking (CNB, MNB, BNB, GNB)', m16.predict_proba(data))
    print('Stacking (RF,HGBC, GBC, AdaBoost, ET)', m17.predict_proba(data))
    print('Stacking (SVM, kNN, DT)', m18.predict_proba(data))
    print('SVM', m19.predict(data))
