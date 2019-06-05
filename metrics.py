import numpy as np

def get_confusionMatrix(Y_test, Y_pred, classes):

    cm = np.zeros( [len(classes),len(classes)], dtype=int )
    for i in range(len(classes)):
        for j in range(len(classes)):
            cm[i][j] = len(np.intersect1d(np.where(Y_test==classes[i])[0], np.where(Y_pred==classes[j])[0]))

    return cm

def relatorioDesempenho(matriz_confusao, classes, imprimeRelatorio=False):

    n_teste = sum(sum(matriz_confusao))

    nClasses = len( matriz_confusao ) #numero de classes

    # inicializa as medidas que deverao ser calculadas
    vp=np.zeros( nClasses ) # quantidade de verdadeiros positivos
    vn=np.zeros( nClasses ) # quantidade de verdadeiros negativos
    fp=np.zeros( nClasses ) # quantidade de falsos positivos
    fn=np.zeros( nClasses ) # quantidade de falsos negativos

    acuracia = 0.0

    revocacao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    revocacao_macroAverage = 0.0
    revocacao_microAverage = 0.0

    precisao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    precisao_macroAverage = 0.0
    precisao_microAverage = 0.0

    fmedida = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    fmedida_macroAverage = 0.0
    fmedida_microAverage = 0.0

    acuracia = np.diag(matriz_confusao).sum() / np.sum(matriz_confusao)
    revocacao = np.diag(matriz_confusao) / np.sum(matriz_confusao, axis=1)
    precisao = np.diag(matriz_confusao) / np.sum(matriz_confusao, axis=0)
    fmedida = 2 * (revocacao * precisao) / (revocacao + precisao)
    revocacao_macroAverage = np.mean(revocacao)
    precisao_macroAverage  = np.mean(precisao)
    fmedida_macroAverage   = 2 * (revocacao_macroAverage * precisao_macroAverage) / (revocacao_macroAverage + precisao_macroAverage)
    revocacao_microAverage = np.diag(matriz_confusao).sum() / (np.sum(matriz_confusao))
    precisao_microAverage  = np.diag(matriz_confusao).sum() / (np.sum(matriz_confusao))
    fmedida_microAverage = 2 * (revocacao_microAverage * precisao_microAverage) / (revocacao_microAverage + precisao_microAverage)

    if imprimeRelatorio:
        print('\n\tRevocacao   Precisao   F-medida   Classe')
        for i in range(0,nClasses):
            print('\t%1.3f       %1.3f      %1.3f      %s' % (revocacao[i], precisao[i], fmedida[i],classes[i] ) )

        print('\t------------------------------------------------');

        #imprime as médias
        print('\t%1.3f       %1.3f      %1.3f      Média macro' % (revocacao_macroAverage, precisao_macroAverage, fmedida_macroAverage) )
        print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (revocacao_microAverage, precisao_microAverage, fmedida_microAverage) )

        print('\tAcuracia: %1.3f' %acuracia)


    # guarda os resultados em uma estrutura tipo dicionario
    resultados = {'revocacao': revocacao, 'acuracia': acuracia, 'precisao':precisao, 'fmedida':fmedida}
    resultados.update({'revocacao_macroAverage':revocacao_macroAverage, 'precisao_macroAverage':precisao_macroAverage, 'fmedida_macroAverage':fmedida_macroAverage})
    resultados.update({'revocacao_microAverage':revocacao_microAverage, 'precisao_microAverage':precisao_microAverage, 'fmedida_microAverage':fmedida_microAverage})
    resultados.update({'confusionMatrix': matriz_confusao})

    return resultados 


