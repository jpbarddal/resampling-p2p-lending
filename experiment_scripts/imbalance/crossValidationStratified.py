import pandas as pd
import numpy as np
import random


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
        if self.classe == "lastCol":
            self.classe = list(self.base.columns.values)[-1]
        self.listaLabels = [el for el in self.base[self.classe].unique()]
        self.n_instancias = self.base.shape[0]
        for cadaLabel in self.listaLabels:
            self.classeDataSet[cadaLabel] = pd.DataFrame(np.array(self.base.loc[self.base[self.classe] == cadaLabel]),
                                                         columns=list(self.base.columns.values))
            self.listaInfos.append([cadaLabel, self.classeDataSet[cadaLabel].shape[0]])

        if self.verbose:
            self.listaInfos = sorted(self.listaInfos)
            for dados in self.listaInfos:
                print('Classe {0:4} - {1:4} instâncias.'.format(dados[0], dados[1]))
            print(self.separator)
            print('Total {0:4} instâncias.'.format(self.n_instancias))
            print(self.separator)

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
