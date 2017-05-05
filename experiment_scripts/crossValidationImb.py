from flow_sampling import SamplingReg
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import tree, svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import math
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, ClusterCentroids, EditedNearestNeighbours
import time
import sqlite3
from pandas.parser import CParserError
from scipy.io import arff
from os import listdir
from os.path import isfile, join
import os
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from scipy import stats, spatial
from itertools import combinations
from operator import itemgetter
import random
from sklearn import linear_model

class CrossValidationStratified:
    """ Esta classe recebe como parâmetro 4 valores:
     -> dataset é uma string referente ao endereço e nome contendo a
     base em formato csv #######
     Exemplo: 'base1.csv' ou '/home/seuUsuario/pasta/base.csv'

     -> classe é o nome do atributo referente a classe, deve ser uma
     string. Se não informada a classe ajustara a ultima coluna do
     dataframe como classe
     Exemplo: 'Classe', 'TARGET'

     -> seed é a semente aleatória para parâmetros randômicos,
     deve ser um int.
     Exemplo: 1, 124522, 9

     -> sep é o separador do banco de dados. Valor é string.
     Exemplo: "," , ";"

     ->verbose é booleano responsável por mostrar algumas mensagens de
     sistema

       Desenvolvido por Luis Eduardo Boiko Ferreira novembro/2016
       luiseduardo.boiko@gmail.com """

    def __init__(self, dataset, classe="lastCol", seed=1, sep=",", verbose=True):
        self.dataSet = dataset
        self.classe = classe
        self.seed = seed
        self.sep = sep
        self.verbose = verbose
        # variáveis dos métodos
        self.listaLabels = list()
        self.classeDataSet = dict()
        self.listaInfos = list()
        self.lista_folds = list()
        self.listaCombinacoes = list()
        self.base = None
        self.n_elementos = 0
        self.n_instancias = 0
        self.n_folds = 0
        self.separator = 45 * '-'

    def splitClasses(self):
        try:
            self.base = pd.read_csv(self.dataSet, sep=self.sep)
        except pd.parser.CParserError:
            print('Ocorreu um erro ao ler a base!')
        except OSError:
            print('A base de dados não existe!')
        # verifico se foi fornecido uma classe, senão devo usar a ultima coluna
        if self.classe == "lastCol":
            self.classe = list(self.base.columns.values)[-1]
        # preencho a lista que vai armazenar os valores possíveis para classe
        self.listaLabels = [el for el in self.base[self.classe].unique()]
        # crio uma variavel que vai armazenar o total de instâncias do dataset
        self.n_instancias = self.base.shape[0]
        # para cada label presente eu crio um dataset contendo todos os elementos rotulados com ela e
        # adiciono a self.classeDataset
        for cadaLabel in self.listaLabels:
            self.classeDataSet[cadaLabel] = pd.DataFrame(np.array(self.base.loc[self.base[self.classe] == cadaLabel]),
                                                         columns=list(self.base.columns.values))
            self.listaInfos.append([cadaLabel, self.classeDataSet[cadaLabel].shape[0]])
        # para propositos de impressão
        if self.verbose:
            self.listaInfos = sorted(self.listaInfos)
            for dados in self.listaInfos:
                print('Classe {0:4} - {1:4} instâncias.'.format(dados[0], dados[1]))
            print(self.separator)
            print('Total {0:4} instâncias.'.format(self.n_instancias))
            print(self.separator)

        # retorno um dict contendo todos os datasets
        return self.classeDataSet

    def getImbalanceLevel(self):
        if self.listaInfos[0][1] > self.listaInfos[1][1]:
            imbLevel = "1:" + str(self.listaInfos[0][1] // self.listaInfos[1][1])
        else:
            imbLevel = "1:" + str(self.listaInfos[1][1] // self.listaInfos[0][1])
        return dict(classe_A=self.listaInfos[0][1],
                    classe_B=self.listaInfos[1][1],
                    imbLevel=imbLevel)

    def generateFolds(self, n_folds):
        self.n_folds = n_folds
        # o parâmetro base deve ser uma lista contendo dataframes, cada dataframe referente a uma classe
        # agora adiciono os elementos a cada fold
        self.lista_folds = []
        # random seed
        random.seed(a=self.seed)
        np.random.seed(self.seed)
        # primeiro eu calculo quantos elementos cada fold terá
        self.n_elementos = self.n_instancias // self.n_folds
        for foldAtual in range(self.n_folds):
            # crio uma lista que vai guardar os datasets temporarios antes de concatenar
            listaDataSets = list()
            # agora calculo quantos elementos de cada fold devo sortear por fold
            for ql_classe, cl_elem in self.listaInfos:
                indicesParaSorteio = self.classeDataSet[ql_classe].shape[0]
                # calculo quanto esta classe representa do total
                qntos_elem = int((cl_elem * self.n_elementos) // self.n_instancias)
                indices = random.sample(range(indicesParaSorteio), qntos_elem)
                listaDataSets.append(self.classeDataSet[ql_classe].ix[indices])
                # deleto as instancias usadas
                self.classeDataSet[ql_classe] = \
                    pd.DataFrame(np.array(
                        self.classeDataSet[ql_classe].drop(self.classeDataSet[ql_classe].index[indices])),
                        columns=list(self.base.columns.values))
                # verifico se eh o ultimo fold a ser gerado, se for deve receber os excedentes de cada classe
                if foldAtual == (self.n_folds - 1):
                    listaIndicesRestantes = [iR for iR in range(self.classeDataSet[ql_classe].shape[0])]
                    listaDataSets.append(self.classeDataSet[ql_classe].ix[listaIndicesRestantes])
            # finalmente concateno todos os dataframes em um soh fold
            self.lista_folds.append(pd.concat(listaDataSets, ignore_index=True))
        if self.verbose:
            for i in range(len(self.lista_folds)):
                valor = self.lista_folds[i][self.classe].value_counts()
                print("Fold {0} gerado com {1:4} instâncias ({2}/{3})".format(i+1, self.lista_folds[i].shape[0],
                                                                              valor[0], valor[1]))
            print(self.separator)
        # executo um teste de consistencia para ver se os folds gerados são distintos
        if self.checkConsistence(type='justFolds'):
            print('Ocorreu um erro na geração dos folds!')

        return self.lista_folds

    def generateFoldsCombinations(self):
        self.listaCombinacoes = list()
        for i in range(self.n_folds):
            testFold = self.lista_folds[i]
            remainFolds = [self.lista_folds[j] for j in range(len(self.lista_folds)) if j != i]
            # crio um df com os folds restantes
            trainFold = pd.concat(remainFolds, ignore_index=True)
            self.listaCombinacoes.append([trainFold, testFold])
        if self.checkConsistence(type='foldsCombination'):
            print("Falha na geração dos folds")
        return self.listaCombinacoes

    def checkConsistence(self, type):
        erro = False
        if type == 'justFolds':
            df_fixo = self.lista_folds[0]
            for df_i in range(1, len(self.lista_folds)):
                if df_fixo.equals(self.lista_folds[df_i]):
                    erro = True
        elif type == 'foldsCombinations':
            df_fixo = self.listaCombinacoes[0][0]
            for df_i in range(1, len(self.listaCombinacoes)):
                if df_fixo.equals(self.lista_folds[df_i][0]):
                    erro = True
        return erro


class ClassifyCV:
    """Esta classe é recebe 6 parâmetros:
    -> foldslist (list) eh a lista de combinações dos folds gerados pela classe CrossValidationStratified
    -> clf (método) é o classificador utilizado (único)
    -> classe (string) é a classe (deerrr), se não for passado nenhum parâmetro a ultima coluna do 
    dataset é utilizada
    -> resamplingTechnique (string) é a técnica de over / under sampling aplicada. Se não informado
    nenhuma técnica sera utilizada.
    -> toSql (bool) responsável por salvar os resultados ou não em um sqlite previamente criado
    ->verbose (bool) é referente a impressão de alguns valores na tela durante a execução.
    Classe criada por Luis Eduardo Boiko ferreira
    luiseduardo.boiko@gmail.com
    ultima atualização: 22/11/2016"""

    def __init__(self, foldslist, clf, classe="lastCol", resamplingtechnique=None,
                 tosql=False, verbose=False):
        self.foldsList = foldslist
        self.clf = clf
        self.classe = classe
        self.resamplingTechnique = resamplingtechnique
        self.toSql = tosql
        self.verbose = verbose
        # criando as listas
        self.listaCM = list()
        self.listaAcuracia = list()
        self.listaPrecision = list()
        self.listaRecall = list()
        self.listaAUC = list()
        self.listaMcc = list()
        self.listaF1 = list()
        # variaveis para as bases
        self.treino_X, self.treino_Y, self.teste_X, self.teste_Y = None, None, None, None
        self.y_pred = None

        # variaveis para as metricas
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
        self.accuracy, self.precisionMin, self.precisionMaj, self.recallMin, self.recallMaj = 0, 0, 0, 0, 0
        self.f1Min, self.f1Maj, self.gMin, self.gMaj, self.aucRes, self.mcc = 0, 0, 0, 0, 0, 0

        # separador
        self.separator = 45 * '-'

    def classify(self):
        """
        Este método é responsável por realizar a classificação das instâncias.
        Utiliza para isto a lista de folds combinadas geradas pela classe CrossValidationStratified.
        """
        if self.classe == "lastCol":
            self.classe = list(self.foldsList[0][0].columns.values)[-1]
        aux = 1
        for trainSet, testSet in self.foldsList:
            # contador de tempo
            startCl = time.time()
            # separando em X e Y
            self.treino_X = trainSet.drop(self.classe, axis=1)
            self.treino_Y = trainSet[self.classe]
            self.teste_X = testSet.drop(self.classe, axis=1)
            self.teste_Y = testSet[self.classe]
            # classificando
            if self.resamplingTechnique is None:
                x_resampled, y_resampled = self.treino_X, self.treino_Y
            else:
                x_resampled, y_resampled = self.resamplingTechnique.fit_sample(self.treino_X,
                                                                               self.treino_Y)
            self.y_pred = self.clf.fit(x_resampled, y_resampled).predict(self.teste_X)
            if self.verbose:
                # imprimo o contador de tempo
                print("Fold {0} - levou {1:2f} s para classificação.".format(aux, time.time()
                      - startCl))
                aux += 1
            # adiciono as metricas as listas
            self.addmetric()
        if self.verbose:
            print(self.separator)
        self.showresults()

    def addmetric(self):
        """
        Este método adiciona os valores de cada fold a lista referente a cada metrica
        """
        self.listaCM.append(confusion_matrix(self.teste_Y, self.y_pred))
        self.listaAcuracia.append(accuracy_score(self.teste_Y, self.y_pred))
        self.listaRecall.append(recall_score(self.teste_Y, self.y_pred, average=None))
        self.listaPrecision.append(precision_score(self.teste_Y, self.y_pred, average=None))
        self.listaF1.append(f1_score(self.teste_Y, self.y_pred, average=None))
        # auc
        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(self.teste_Y.ravel(), self.y_pred.ravel())
        # fpr, tpr, thresholds = roc_curve(self.teste_Y, self.y_pred, pos_label=1)
        self.listaAUC.append(auc(fpr, tpr))

    def showresults(self):
        """
        Este método apenas mostra os resultados, fazendo as médias dos valores
        """
        # matriz de confusão
        self.tp = [sum(i) for i in zip(*self.listaCM)][0][0] / len(self.foldsList)
        self.fn = [sum(i) for i in zip(*self.listaCM)][0][1] / len(self.foldsList)
        self.fp = [sum(i) for i in zip(*self.listaCM)][1][0] / len(self.foldsList)
        self.tn = [sum(i) for i in zip(*self.listaCM)][1][1] / len(self.foldsList)
        self.accuracy = sum(self.listaAcuracia) / len(self.foldsList)
        self.precisionMaj = [sum(i) for i in zip(*self.listaPrecision)][0] / len(self.foldsList)
        self.precisionMin = [sum(i) for i in zip(*self.listaPrecision)][1] / len(self.foldsList)
        self.recallMaj = [sum(i) for i in zip(*self.listaRecall)][0] / len(self.foldsList)
        self.recallMin = [sum(i) for i in zip(*self.listaRecall)][1] / len(self.foldsList)
        self.f1Maj = [sum(i) for i in zip(*self.listaF1)][0] / len(self.foldsList)
        self.f1Min = [sum(i) for i in zip(*self.listaF1)][1] / len(self.foldsList)

        self.gMaj = math.sqrt(self.precisionMaj * self.recallMaj)
        self.gMin = math.sqrt(self.precisionMin * self.recallMin)
        # mcc
        self.mcc = (self.tp * self.tn - (self.fp - self.fn)) / math.sqrt((self.tp + self.fp) *
                                                                         (self.tp + self.fn) *
                                                                         (self.tn + self.fp) *
                                                                         (self.tn + self.fn))
        # auc
        self.aucRes = sum(self.listaAUC) / len(self.foldsList)
        print('Acurácia:        {0:2f}'.format(self.accuracy))
        print('Precision Maj:   {0:2f}'.format(self.precisionMaj))
        print('Precision Min:   {0:2f}'.format(self.precisionMin))
        print('Recall Maj:      {0:2f}'.format(self.recallMaj))
        print('Recall Min:      {0:2f}'.format(self.recallMin))
        print('F1 Maj:          {0:2f}'.format(self.f1Maj))
        print('F1 Min:          {0:2f}'.format(self.f1Min))
        print('G-measure Maj:   {0:2f}'.format(self.gMaj))
        print('G-measure Min:   {0:2f}'.format(self.gMin))
        print('MCC:             {0:2f}'.format(self.mcc))
        print('AUC:             {0:2f}'.format(self.aucRes))
        print('[[{0} {1:4}] \n [{2:4} {3:4}]]'.format(int(round(self.tp)), 
              int(round(self.fp)), int(round(self.fn)), int(round(self.tn))))

        if self.toSql:
            # sqlite
            db = "resultados.sqlite"
            # if not os.path.isfile(db):
            #     conn = sqlite3.connect(db)
            #     c = conn.cursor()
            #     # Create table
            #     c.execute('''CREATE TABLE individualBancos
            #                                  (base text, tratamento text, instanciasA , instanciasB,
            #                               acuracia real, precisionMaj real, preccisionMin real, recallMaj real,
            #                               recallMin real, f1Maj real, f1Min real, gMaj real, gMin real, mcc real,
            #                               auc real, tp int, tn int, fp int,
            #                               fn int, clf text,imbLevel text)''')
            #     conn.commit()
            #     conn.close()
            # insiro os resultados novos
            conn = sqlite3.connect(db)
            c = conn.cursor()
            c.execute("INSERT INTO individualBancos(base, tratamento, instanciasA, instanciasB,"
                      "acuracia, precisionMaj, precisionMin, recallMaj, recallMin,"
                      "f1Maj, f1Min, gMaj, gMin, mcc, auc, tp, tn, fp, fn, clf,imbLevel)"
                      " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )", (
                          dataSet, str(self.resamplingTechnique), dictDesbalanceamento['classe_A'],
                          dictDesbalanceamento['classe_B'], self.accuracy, self.precisionMaj,
                          self.precisionMin, self.recallMaj, self.recallMin, self.f1Maj, self.f1Min,
                          self.gMaj, self.gMin, self.mcc, self.aucRes,
                          int(round(self.tp)), int(round(self.tn)), int(round(self.fp)),
                          int(round(self.fn)), str(self.clf), dictDesbalanceamento['imbLevel']))
            conn.close()

if __name__ == '__main__':
    # exemplo
    # defino a lista de datasets
    dataSets = ["pima.csv"]
    # a lista com as tecnicas de amostragem
    tecnicasAmostragem = [None, SMOTE(kind="regular", ratio=1.0), SMOTETomek(ratio=1.0),
                          SMOTE(kind="borderline1", ratio=1.0),
                           SMOTE(kind="borderline2", ratio=1.0)]
    clfs = [svm.LinearSVC(), BernoulliNB(), tree.DecisionTreeClassifier(criterion="entropy", max_depth=7)]

    for dataSet in dataSets:
        cv = CrossValidationStratified(dataset=dataSet, verbose=True)
        cv.splitClasses()
        # pego os dados de desbalanceamento para o sql
        dictDesbalanceamento = cv.getImbalanceLevel()
        cv.generateFolds(n_folds=5)
        meusFolds = cv.generateFoldsCombinations()
        for uniClf in clfs:
            print(50*'+')
            print(uniClf)
            for technique in tecnicasAmostragem:
                print(technique)
                m = ClassifyCV(foldslist=meusFolds,
                               clf=uniClf,
                               resamplingtechnique=technique,
                               tosql=True,
                               verbose=True)
                m.classify()
