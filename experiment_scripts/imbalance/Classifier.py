class Classifier:

    def __init__(self, alg, name):
        self._alg = alg
        self._name = name

    def get(self):
        return dict(alg=self._alg,
                    name=self._name)


a = Classifier(alg='DT',
               name='CART')

print(a._alg)