import feature_extractor as fe
import pickle
from tensorflow import keras
import tkinter as tk
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


def check_site():
    data = [fe.extract_features(url.get())]

    result.configure(state='normal')

    result.delete(1.0, tk.END)

    result.insert(tk.END, ('\nAdaBoost_DT', m1.predict(data)))
    result.insert(tk.END, ('\nBagging_DT', m2.predict(data)))
    result.insert(tk.END, ('\nBernoulli_NB', m3.predict(data)))
    result.insert(tk.END, ('\nComplement_NB', m4.predict(data)))
    result.insert(tk.END, ('\nDT', m5.predict(data)))
    result.insert(tk.END, ('\nET', m6.predict(data)))
    # # print('Gaussian_NB', m7.predict(data))
    result.insert(tk.END, ('\nGradientBoost', m8.predict(data)))
    result.insert(tk.END, ('\nHistGradientBoost', m9.predict(data)))
    result.insert(tk.END, ('\n2NN', m10.predict(data)))
    result.insert(tk.END, ('\nMultinomial_NB', m11.predict(data)))
    result.insert(tk.END, ('\nneural_networks', m12.predict(data)))
    result.insert(tk.END, ('\nRF', m13.predict(data)))
    result.insert(tk.END, ('\nStacking (AdaBoost_DT, ET, DT, Bagging_DT, RF)', m14.predict(data)))
    result.insert(tk.END, ('\nStacking (All)', m15.predict(data)))
    result.insert(tk.END, ('\nStacking (CNB, MNB, BNB, GNB)', m16.predict(data)))
    result.insert(tk.END, ('\nStacking (RF,HGBC, GBC, AdaBoost, ET)', m17.predict(data)))
    result.insert(tk.END, ('\nStacking (SVM, kNN, DT)', m18.predict(data)))
    result.insert(tk.END, ('\nSVM', m19.predict(data)))

    result.configure(state='disabled')


if __name__ == "__main__":
    window = tk.Tk()

    window.title("phishDetect")
    window.resizable(0, 0)

    url = tk.StringVar()

    textArea = tk.Entry(textvariable=url, width=80)
    textArea.grid(column=0, row=0, sticky=tk.N+tk.S+tk.W+tk.E)

    btn = tk.Button(window, text="check", command=check_site)
    btn.grid(column=1, row=0, sticky=tk.N+tk.S+tk.W+tk.E)

    # row 1 - loader widget

    scroll = tk.Scrollbar(window)
    scroll.grid(column=3, row=2, sticky=tk.N + tk.S + tk.W + tk.E)

    result = tk.Text(window,
                     height=15,
                     width=80,
                     state='disabled'
                     , yscrollcommand=scroll.set
                     )
    result.grid(column=0, row=2, columnspan=2)

    scroll.config(command=result.yview)


    window.mainloop()



