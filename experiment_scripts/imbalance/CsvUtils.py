import pandas as pd
import numpy as np
import random

class LoadData:
    def __init__(self, csvFile, dBaseClass='lastCol', sep=',', seed=1):
        self.sep = sep
        self.csvFile = pd.read_csv(csvFile, sep=sep)
        self.dBase = pd.DataFrame(self.csvFile, columns=list(self.csvFile.columns.values))
        self.dBaseClass = dBaseClass if dBaseClass != 'lastCol' else list(self.dBase.columns.values)[-1]
        self.nInstancias = self.dBase.shape[0]
        self.listaLabels = [el for el in self.dBase[self.dBaseClass].unique()]
        # stratifiedSplit
        self.classeDataSet = dict()
        self.listaInfos = list()
        self.classeMin = int()
        self.classeMaj = int()
        self.desbalanceamento = int()
        self.seed = seed

    def returnRawBase(self):
        return self.dBase
    
    def splitDataFromClass(self):
        return self.dBase.drop(self.dBaseClass, axis=1), self.dBase[self.dBaseClass]

    def splitDifferentClasse(self):
        for cadaLabel in self.listaLabels:
            self.classeDataSet[cadaLabel] = pd.DataFrame(np.array(self.dBase.loc[self.dBase[self.dBaseClass] ==
                                                                                 cadaLabel]),
                                                         columns=list(self.dBase.columns.values))
            self.listaInfos.append([cadaLabel, self.classeDataSet[cadaLabel].shape[0]])

        return self.listaInfos

    def getImbLevel(self):
        if self.listaInfos[0][1] > self.listaInfos[1][1]:
            self.classeMaj = self.listaInfos[0][0]
            self.classeMin = self.listaInfos[1][0]
            self.desbalanceamento = round(self.listaInfos[0][1] / self.listaInfos[1][1])
        else:
            self.classeMaj = self.listaInfos[1][0]
            self.classeMin = self.listaInfos[0][0]
            self.desbalanceamento = round(self.listaInfos[1][1] / self.listaInfos[0][1])

        return dict(classeMaj=self.classeMaj,
                    classeMin=self.classeMin,
                    desbalanceamento=self.desbalanceamento)


    def getStratifiedPortion(self, percent, n1, n2):
        self.returnRawBase()
        self.splitDifferentClasse()
        # seed
        random.seed(a=self.seed)
        np.random.seed(self.seed)
        # a quantidade que eu quero pegar
        percent = round(self.dBase.shape[0] * percent / 100)
        # variavel auxiliar para calcular quanto cada parte cabe a classe min
        s_min = round(percent / (self.getImbLevel()["desbalanceamento"] + 1))
        s_maj = self.getImbLevel()["desbalanceamento"] * s_min
        listaDataSets = list()
        for ql_classe, cl_elem in self.listaInfos:
            indicesParaSorteio = self.classeDataSet[ql_classe].shape[0]
            # calculo quanto esta classe representa do total
            if ql_classe == self.classeMin:
                qntos_elem = s_min
            else:
                qntos_elem = s_maj
            indices = random.sample(range(indicesParaSorteio), qntos_elem)
            listaDataSets.append(self.classeDataSet[ql_classe].ix[indices])
            # deleto as instancias usadas
            self.classeDataSet[ql_classe] = \
                pd.DataFrame(np.array(
                    self.classeDataSet[ql_classe].drop(self.classeDataSet[ql_classe].index[indices])),
                    columns=list(self.dBase.columns.values))

        dfFinal = pd.concat(listaDataSets, ignore_index=True)
        dfResto = pd.concat([self.classeDataSet[0], self.classeDataSet[1]], ignore_index=True)

        dfFinal.to_csv(n1, sep=',', columns=self.dBase.columns.values, index=False)
        dfResto.to_csv(n2, sep=',', columns=self.dBase.columns.values, index=False)

        print('Majority Class: {0} | Minority Class: {1} | Imb. Level 1:{2}'.format(self.classeMin, self.classeMaj,
                                                                                    self.desbalanceamento))
        print('Original database size: {0} instances'.format(self.dBase.shape[0]))
        print('The split corresponds to:{0}'.format(dfFinal.shape[0]))
        print('with {0} minority instances and {1} majority'.format(s_min, s_maj))
        print('#pas')


#
# a = CsvUtils('../pima.csv')
# X, y = a.splitDataFromClass()
# n1 = 'p2p_lendingclub_70_stratified.csv'
# n2 = 'p2p_lendingclub_30_stratified.csv'
