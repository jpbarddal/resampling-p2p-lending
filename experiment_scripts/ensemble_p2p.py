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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier


def getParamsClassifierEnsamble(classifierName):
    # Importante: eu acho que soh usa max_depth quando max_leaf_nodes = None e vice-versa. Portanto nao aumenta muito
    # a quantidade de modelos a testar quando adicionamos mais valores para ambos (como eu fiz :P)
    if classifierName == "Random Forest":
        return dict(clf__n_estimators=[10, 50, 100],  # Good
                    # clf__criterion=["gini", "entropy"])#, # Good
                    clf__max_features=["auto", "sqrt", "log2", None],  # Esta ok, poderia ter mais valores
                    clf__max_depth=[5, 10,
                                    15])  # retirei o none para evitar testar com arvores sem limite de tamanho
        # clf__min_samples_split=[10, 50, 100], # Retirado por questoes de performance
        # clf__min_samples_leaf=[10, 50, 100, 200]) # Retirado por questoes de performance
        # clf__max_leaf_nodes=[None, 20, 50, 100], # Tirei esse. Ele eh mutualmente exclusivo com o max_depth e normalmente otimizamos o max_depth (eh mais compreensivel)
        # clf__bootstrap=[True, False]) # Good

    elif classifierName == "AdaBoost":
        return dict(clf__n_estimators=[10, 50, 100],  # Good, alinhado com RF
                    clf__learning_rate=[0.1, 1, 2],  # Mudei para 0.1, 1 e 2
                    clf__algorithm=["SAMME", "SAMME.R"])

    elif classifierName == "Bagging":
        return dict(clf__n_estimators=[10, 50, 100],  # Alinhado com os outros
                    clf__max_samples=[0.10, 0.25, 0.5, 0.75, 1.0],
                    # Coloquei mais um valor baixo (10%), provavelmente nao vai melhorar o resultado final, mas soh pra fins de teste mesmo.
                    clf__max_features=[0.10, 0.25, 0.5, 0.75,
                                       1.0])  # Mesma coisa, coloquei um outro valor baixo. Aqui tem mais chances the melhorar o resultado final, pois temos muitas features.
        # clf__bootstrap=[True, False], # Ok
        # clf__bootstrap_features=[True, False]) # Ok
        # clf__warm_start=[True, False]) # Nao se aplica a grid search, seria reutilizar o modelo utilizado na chamada anterior ao fit
    else:
        print("\nUnable to get classifier parameters!\n")
        return dict()


def mergeDict(d1, d2):
    return dict(list(d1.items()) + list(d2.items()))


def generate_names(dbases, labels):
    final_names = list()
    for set in dbases:
        for name in labels:
            final_names.append(str(name + '_' + set[:-4]))
    return final_names


if __name__ == '__main__':

    trainSets = ['p2p_lendingclub_70.csv']
    testSets = ['p2p_lendingclub_30.csv']
    classe = 'loan_status'
    clfs = [RandomForestClassifier(), AdaBoostClassifier(), BaggingClassifier()]
    names = ["Random Forest", "AdaBoost", "Bagging"]
    sTechniques = [None]
    technique_names = ["Ensemble-based"]
    final_names = generate_names(testSets, names)

    for trainSet, testSet, name in zip(trainSets, testSets, final_names):
        # folds generation
        cv = CrossValidationStratified(dataset=testSet, classe=classe, verbose=True)
        cv.splitClasses()
        cv.generateFolds(n_folds=5)
        meusFolds = cv.generateFoldsCombinations()
        lista_statistic = list()
        # leitura da base para gridSearch
        a = CsvUtils.LoadData(trainSet, classe)
        X_train, y_train = a.splitDataFromClass()
        for clf, name in zip(clfs, names):
            paramsClf = getParamsClassifierEnsamble(name)
            pipeline = Pipeline([('clf', clf)])
            grid_search = GridSearchCV(pipeline, param_grid=[paramsClf], n_jobs=5, scoring='roc_auc')
            grid_search.fit(X_train, y_train)
            # print the classifier in order to show the parameters
            print(grid_search.best_estimator_)
            print(name, testSet + '\n')
            # defino a lista de tecnicas de amostragem como None pois ja possuo a info completa do codigo anterior
            m = ClassifyCV(foldslist=meusFolds,
                           dataset=testSet,
                           clf=grid_search.best_estimator_,
                           clf_label=name,
                           classe=classe,
                           resamplingtechnique=None,
                           techiqueLabel="Ensemble-based",
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

