from weka.classifiers import IBk
c = IBk(K=1)
c.train('training.arff')
predictions = c.predict('query.arff')