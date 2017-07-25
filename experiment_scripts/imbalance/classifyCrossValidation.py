from sklearn.metrics import confusion_matrix, roc_curve, auc
import time
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import math
import sqlite3
import pandas as pd


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

    def __init__(self, foldslist, dataset, clf, clf_label, classe="lastCol", resamplingtechnique=None,
                 techiqueLabel=None, tosql=False, verbose=False):
        self.foldsList = foldslist
        self.dataset = dataset
        self.clf = clf
        self.clf_label = clf_label
        self.classe = classe
        self.resamplingTechnique = resamplingtechnique
        self.techniqueLabel = techiqueLabel
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

        # variaveis para estatistica
        self.statisticsList = list()

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
        # statistic
        colunas =['DataSet', 'Classifier', 'AUROC']
        self.statisticsList.append(pd.DataFrame(dict(DataSet=self.dataset,
                                                     Classifier=self.clf_label + '+' + self.techniqueLabel,
                                                     AUROC=self.listaAUC), columns=colunas))

        print('Class: {0:10d} {1:10d}'.format(*[int(el) for el in self.teste_Y.unique()], sep=','))
        print('Precision:   {0:10s} {1:10s}'.format('%.2f' % self.precisionMaj, '%.2f' % self.precisionMin))
        # print('Precision Min:   {0:2f}'.format(self.precisionMin))
        print('Recall:      {0:10s} {1:10s}'.format('%.2f' %  self.recallMaj, '%.2f' % self.recallMin))
        # print('Recall Min:      {0:2f}'.format(self.recallMin))
        print('F1:          {0:10s} {1:10s}'.format('%.2f' %  self.f1Maj, '%.2f' % self.f1Min))
        # print('F1 Min:          {0:2f}'.format(self.f1Min))
        print('G-measure:   {0:10s} {1:10s}'.format('%.2f' %  self.gMaj, '%.2f' % self.gMin))
        # print('G-measure Min:   {0:2f}'.format(self.gMin))
        print('\nGlobal metrics:')
        print(self.separator)
        print('Acurácia:    {0:10s}'.format('%.2f' % self.accuracy))
        print('MCC:         {0:10s}'.format('%.2f' % self.mcc))
        print('AUC:         {0:10s}'.format('%.2f' % self.aucRes))
        print('\nConfusion Matrix:')
        print(self.separator)
        print('Class:  {0:3d} {1:10d}'.format(*[int(el) for el in self.teste_Y.unique()], sep=','))
        print('{0}{1:10d} {2:10d}'.format([int(el) for el in self.teste_Y.unique()][0],
                                          int(round(self.tp)), int(round(self.fp))))
        print('{0}{1:10d} {2:10d}\n\n'.format([int(el) for el in self.teste_Y.unique()][1],
                                          int(round(self.fn)), int(round(self.tn))))
        # print('[[{0} {1:4}] \n [{2:4} {3:4}]]'.format(int(round(self.tp)),
        #       int(round(self.fp)), int(round(self.fn)), int(round(self.tn))))

        if self.toSql:
            # sqlite
            db = "resultados.sqlite"
            conn = sqlite3.connect(db)
            c = conn.cursor()
            c.execute("INSERT INTO individualBancos(base, tratamento, instanciasA, instanciasB,"
                      "acuracia, precisionMaj, precisionMin, recallMaj, recallMin,"
                      "f1Maj, f1Min, gMaj, gMin, mcc, auc, tp, tn, fp, fn, clf,imbLevel)"
                      " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )", (
                          'lendingClub', str(self.resamplingTechnique), 666,
                          666, self.accuracy, self.precisionMaj,
                          self.precisionMin, self.recallMaj, self.recallMin, self.f1Maj, self.f1Min,
                          self.gMaj, self.gMin, self.mcc, self.aucRes,
                          int(round(self.tp)), int(round(self.tn)), int(round(self.fp)),
                          int(round(self.fn)), str(self.clf), str("1:4")))
            conn.commit()
            conn.close()

    def showStats(self):
        return pd.concat(self.statisticsList, ignore_index=True)