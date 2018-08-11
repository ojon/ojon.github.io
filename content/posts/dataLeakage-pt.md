Title: Armadilhas do vazamento de dados em ciência de dados
Slug: dataLeakage
Date: 2018-08-11 16:00
Category: Ciência de dados
Tags: ciência de dados, seleção de atributos, validação, mineração de dados, scikit-learn, intermediário, aprendizado de máquina, inteligência artificial
Author: João Oda
Lang: pt
<!-- Status: draft -->

![mario_see_data_leakage](/images/leakimg.jpg)

Na tentativa de organizar meus projetos, acabei de refatorar um antigo [projeto de mineração de dados](https://github.com/ojon/MD_Proj) que fiz alguns anos atrás. Originalmente utilizei Python para o tratamento dos dados e a seleção de atributos(_feature selection_) e depois com [Weka](https://www.cs.waikato.ac.nz/ml/weka/) realizei o treinamento, validação e seleção dos modelos. Reimplentei tudo em Python com o [sckit-learn](http://scikit-learn.org/stable/), fazendo uso do recurso de [_pipelines_](http://scikit-learn.org/stable/modules/pipeline.html#pipeline) e aproveitei para corrigir um erro de vazamento de dados(_data leakage_).

No contexto de ciência de dados, de uma forma ampla temos dois casos de **vazamento de dados** em modelos preditivos:

## Atributos vazados

Ocorre quando o conjunto de atributos de treinamento, possuí informações(tipicamente com origem na variável que queremos prever) que não estarão presentes, quando realizarmos a predição em um ambiente de produção.

### Exemplos

Suponha que você deseja criar um modelo preditivo, que prevê se um empréstimo será pago em dia. Tomando um conjunto real de dados como exemplo, no site do [lendingclub](https://www.lendingclub.com/info/download-data.action) temos uma base de dados disponível, onde uma das colunas "total_rec_late_fee"(Late fees received to date), informa as taxas atrasadas recebidas até o momento. Caso o valor desta coluna seja diferente de zero, é claro que empréstimo não foi pago em dia. Não podemos utilizar esta informação, que somente estará disponível após a concessão do empréstimo, pois um modelo que realiza uma predição previamente ao se realizar um empréstimo, é o que se espera. Nesta mesma fonte de dados, existem muitas outras colunas a serem desconsideradas por motivo semelhante.

No contexto de saúde, um banco de dados pode apresentar uma lista de medicamentos, que uma pessoa toma e você esta tentando desenvolver um modelo para realizar um diagnóstico. Deve-se tomar o cuidado de descobrir, se a lista que consta no banco de dados foi atualizada após o diagnóstico do médico, devido a uma prescrição do mesmo ou se eram remédios que a pessoa já tomava previamente devido a alguma condição que possui.

Agora um caso que lidei no passado ao trabalhar com algorítimos de negociação. Imagine que seus dados são series temporais, onde $x_{t-1}$ e $x_{t+1}$ são conhecidos, porem $x_t$ é um dado faltante. Você decide então fazer uma interpolação onde $x_{t} = \frac{x_{t-1} + x_{t+1}}{2}$ para completar a lacuna faltante. Esta abordagem compromete a criação de um modelo que utiliza dados até instante t para prever o que ocorre t+1, pois ocorreu vazamento de dados do instante t+1 para o instante prévio t.


## Treinamento com exemplos vazados

Ocorre quando, informações provenientes do conjunto de validação, são utilizadas no treinamento do modelo. Você pode pensar que simplesmente particionar o seu conjunto de dados em treino e teste antes de realizar o treinamento é o suficiente. No entanto a situação pode ser um pouco mais complicada e o vazamento ter ocorrido previamente durante a preparação dos dados, seleção de atributos, redução de dimensionalidade e sendo purista até mesmo a visualização ampla dos mesmo.

Agora vou exemplificar um processo de seleção de atributos que utilizei no projeto mencionado no início deste post. Este projeto lida com dados de [expressão gênica](https://pt.wikibooks.org/wiki/Biologia_celular/Express%C3%A3o_gen%C3%A9tica)([_microarray_](https://pt.wikipedia.org/wiki/Microarranjo_de_DNA)) onde o número de atributos é muito maior que o número de amostras. Neste caso utilizar um número muito grande de atributos acarreta em um modelo cujo treinamento demora um tempo maior, com menos interpretabilidade e mais propenso a uma situação overfitting, com um maior erro de generalização.

### Seleção de atributos a partir do teste t

Uma forma simples de realizar a seleção de atributos é através de um filtro baseado no teste-t. Neste versão do filtro, para cada rotulo a ser prevista, o filtro biparticiona as amostra e realiza um teste t. São selecionados os `w` atributos, com maior t-valor absoluto para cada rótulo.

Eu implementei o filtro em python como um transformer, seguindo o padrão da biblioteca sckit-learn. Isso é feito estendendo-se as classes `BaseEstimator`,`TransformerMixin` e implementa-se os métodos `fit` e `transform`. Isto possibilita a utilização em [_pipelines_](http://scikit-learn.org/stable/modules/pipeline.html#pipeline) da mesma biblioteca. Para aplicar o filtro são chamados os dois métodos em sequência, `fit` e `transform`.


```python
class TtestScoreSelection(BaseEstimator, TransformerMixin):
    def __init__(self, w=3):
        self.w = w

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.labels_ = unique_labels(y)

        self.tValuesDF = pd.DataFrame(columns=self.labels_)
        self.sortedIndexes = pd.DataFrame(columns=self.labels_)

        for label in self.labels_:
            sample1 = X[y == label]
            sample2 = X[y != label]
            t = st.ttest_ind(sample1, sample2, equal_var=False)
            self.tValuesDF[label] = np.abs(t[0])                      
            self.sortedIndexes[label] = self.tValuesDF.sort_values(by=label,
                                                                   ascending=False).index

        # Return the transformer
        return self

    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        X = check_array(X)

        # union of indexes from the top w columns for each label
        self.selCols = np.unique(self.sortedIndexes[:][0:self.w].values.flatten())

        return X[:, self.selCols]
```  


### Predições a partir de números aleatórios

Vamos agora utilizar nosso filtro para selecionar atributos de um forma errônea. Considere o seguinte experimento didático:

1. Vamos gerar um conjunto aleatório de 100 amostras de 100.000 Atributos.
2. Realizar a redução de atributos por meio do filtro `TtestScoreSelection`
3. Treinar uma SVM utilizando validação cruzada de 5-folds


```Python
#random data generation
X = np.random.rand(100, 100000)
y = np.random.randint(2,size=100)

#feature selection
tscoreSel = TtestScoreSelection(w=30)
tscoreSel.fit(X,y)
selX = tscoreSel.transform(X)

#SVM training and acuracy estimation by cross-validation.
lr = svm.SVC()
k= 5
scores_leak = pd.DataFrame()
scores_leak['score'] = cross_val_score(lr, selX, y, cv=k)
```

Como resultado temos uma acurácia próxima de 100% ao longo dos 5 folds.

![resultados_com_vazamento_de_dados](/images/scores_with_leak.png)

Como isto é possível??? Realizar predições com acurácia próxima de 100%, a partir de números aleatórios?!

Se analisarmos cuidadosamente nosso procedimentos podemos observar que durante o passo 2, a seleção de variáveis ocorreu utilizando todo conjunto de dados. Como nosso filtro faz uso dos rótulos da classes para particionar as amostras e realizar o teste, nessa etapa propagamos indiretamente informações do conjunto de dados que futuramente será utilizado para validação para o conjunto de treino.

O treinamento depende de quais atributos foram selecionados e estes foram selecionados utilizando inclusive informações relevantes do conjunto do teste. Temos assim um treinamento com exemplos vazados.

### Corrigindo o vazamento de dados.

![Mario](/images/mario_wanna_fix_leakage.jpg)

Alguém avise o Mario que para corrigir este vazamento de dados que "permitiu" uma previsão a partir de números aleatórios não necessitamos de uma chave de grifo, mas utilizar a metodologia correta e escrever um bom código. Sendo assim, precisamos separar um conjunto de dados, sem qualquer intersecção com o conjunto de validação, onde a seleção de atributos e o treinamento ocorrem. Como estamos realizando validação cruzada de 5-folds, isto se repetirá 5 vezes, o conjunto total de dados será particionado em 5 folds e cada vez um fold será utilizado para validação e o complemento para seleção de atributos e treinamento.

O scikit-learn nos permite utilizar o recurso de [_pipelines_](http://scikit-learn.org/stable/modules/pipeline.html#pipeline) para implementar os passos de processamento que os dados são submetidos. A função `cross_val_score()` é capaz de receber um pipeline como argumento e realizar o treinamento e a validação cruzada em k folds. Não é a toa que implementei o filtro de forma compatível com pipelines. Assim o código fica:

```python
pipe = Pipeline([
    ('featureSelection', TtestScoreSelection(w=30)),
    ('classify', svm.SVC())
])
k= 5
scores = pd.DataFrame()
scores['score'] = cross_val_score(pipe, X, y, cv=k)
```
O resultado agora:

![resultados_sem_vazamento_de_dados](/images/scores_without_leak.png)

Como de se esperar, afinal não se pode esperar nada alem de uma acurácia em torno de 50% para uma classificação binária a partir de dados aleatórios.


## Considerações Finais

Vazamentos de dados podem acarretam em surpresas desagradáveis, como modelos que performam de maneira superior em ambiente de desenvolvimento do que em produção. Sempre fique atento, pois sempre o vazamento de dados nem sempre estará explicitamente exposto.


Um post similar que trata do treinamento vazado com uma abordagem em R pode ser encontrado no blog de [Johann de Jong](https://johanndejong.wordpress.com/2017/08/06/feature-selection-cross-validation-and-data-leakage/)
