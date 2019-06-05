from metrics import get_confusionMatrix, relatorioDesempenho
import numpy as np

cm = get_confusionMatrix( np.random.choice(2, 50), np.random.choice(2, 50), [0, 1] )
print(cm)
relatorioDesempenho(cm, [0, 1], imprimeRelatorio=True)

