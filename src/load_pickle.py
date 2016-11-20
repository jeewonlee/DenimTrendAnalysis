import pickle

if __name__ == '__main__':
    mat = pickle.load(open("nmf.pkl", "rb"))
    print mat
