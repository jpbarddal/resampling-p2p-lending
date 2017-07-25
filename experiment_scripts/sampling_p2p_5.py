from imblearn.over_sampling import SMOTE
from imbalance import CsvUtils
from sklearn import tree, linear_model
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss, ClusterCentroids
from imblearn.over_sampling import ADASYN
from imbalance.classifyCrossValidation import ClassifyCV
from imbalance.crossValidationStratified import CrossValidationStratified
import pandas as pd
import time, datetime

if __name__ == '__main__':

    trainSets = ['p2p_lendingclub_70_5percent.csv']
    testSets = ['p2p_lendingclub_30.csv']
    classe = 'loan_status'
    # defino a lista de classificadores
    clfs = [GaussianNB(), tree.DecisionTreeClassifier(), linear_model.LogisticRegression()]
    names = ["Naive Bayes", "Decision Tree", "Logistic Regression"]
    final_names = list()
    for set in testSets:
        for name in names:
            final_names.append(str(name+'_'+set[:-4]))

    # defino a lista de tecnicas de sampling
    sTechniques = [RandomUnderSampler(random_state=1), SMOTE(random_state=1)]
    technique_names = ["RU", "SM"]


    def getParamsReSampling(reSamplingTechnique):
        if type(reSamplingTechnique) is SMOTETomek:
            return dict(smt__ratio=[0.8, 0.9, 1.0], smt__k=[1, 3, 5, 7], smt__m=[1, 3, 5, 7])

        elif type(reSamplingTechnique) is SMOTE:
            return dict(smt__kind=["regular", "borderline1", "borderline2"],
                        smt__ratio=[0.8, 0.9, 1.0], smt__k_neighbors=[1, 3, 5, 7])

        elif type(reSamplingTechnique) is RandomUnderSampler:
            return dict(smt__ratio=[0.8, 0.9, 1.0])

        else:
            return dict()

    def getParamsClassifier(classifierName):
        if classifierName == "Decision Tree":
            return dict(clf__criterion=["gini", "entropy"],
                        clf__splitter=["best", "random"],
                        clf__min_samples_split=[2, 10, 20],
                        clf__max_depth=[None, 2, 5, 10],
                        clf__min_samples_leaf=[1, 5, 10],
                        clf__max_leaf_nodes=[None, 5, 10, 20])
        elif classifierName == "Naive Bayes":
            return dict()
            # return dict(clf__priors=[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6],
            #                          [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])

        elif classifierName == "Logistic Regression":
            return dict(clf__C=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        clf__solver=["newton-cg", "lbfgs", "liblinear", "sag"])
        else:
            print("\nUnable to get classifier parameters!\n")
            return dict()

    def mergeDict(d1, d2):
            return dict(list(d1.items()) + list(d2.items()))


    for trainSet, testSet, name in zip(trainSets, testSets, final_names):
        # folds generation
        cv = CrossValidationStratified(dataset=testSet, classe=classe, verbose=True)
        cv.splitClasses()
        # pego os dados de desbalanceamento para o sql
        dictDesbalanceamento = cv.getImbalanceLevel()
        cv.generateFolds(n_folds=5)
        meusFolds = cv.generateFoldsCombinations()
        lista_statistic = list()
        # leitura da base para gridSearch
        a = CsvUtils.LoadData(trainSet, classe)
        X_train, y_train = a.splitDataFromClass()
        for clf, name in zip(clfs, names):
            for technique, t_name in zip(sTechniques, technique_names):
                paramsRs = getParamsReSampling(technique)
                paramsClf = getParamsClassifier(name)
                if technique == None:
                    pipeline = Pipeline([('clf', clf)])
                else:
                    pipeline = Pipeline([('smt', technique), ('clf', clf)])
                dParams = mergeDict(paramsRs, paramsClf)
                grid_search = GridSearchCV(pipeline, param_grid=[dParams], n_jobs=5, scoring='roc_auc')
                grid_search.fit(X_train, y_train)
                # print the classifier in order to show the parameterscos
                print(grid_search.best_estimator_)
                print(name, t_name, testSet + '\n')
                # defino a lista de tecnicas de amostragem como None pois ja possuo a info completa do codigo anterior
                m = ClassifyCV(foldslist=meusFolds,
                               dataset=testSet,
                               clf=grid_search.best_estimator_,
                               clf_label=name,
                               classe=classe,
                               resamplingtechnique=None,
                               techiqueLabel=t_name,
                               tosql=False,
                               verbose=True)
                m.classify()
                lista_statistic.append(m.showStats())

        df_final = pd.concat(lista_statistic, ignore_index=True)
        f_name = 'statistic_'+testSet+'_'+datetime.datetime.fromtimestamp(time.time()).\
            strftime('%Y_%m_%d__%H-%M-%S')+'.csv'
        print('Base para estatistica salva com nome {0}!\n'.format(f_name))
        #save the csv for statistics
        df_final.to_csv(f_name, columns=df_final.columns.values, index=False)

