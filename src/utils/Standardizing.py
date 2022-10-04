from sklearn.preprocessing import StandardScaler 

class Standardizing:

    def __init__(self):
        self.scaler = StandardScaler() 
    

    def dic2list(self, data, keys):
        res = []
        for key, value in data.items():
            if key in keys:
                res.extend(value)
        return res

    def fit(self, data, keys):
        self.scaler.fit(self.dic2list(data, keys))

    def transform(self, data):
        return self.scaler.transform(data)