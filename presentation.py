import streamlit as st
import pandas as pd
import yfinance as yf

st.title('Optimisation d un portfeuille de crypto utilisant le machine learning :bar_chart:')



with st.expander("Presentation"):

    st.markdown(
    """
> Cette presentation est faite dans le cadre du projet de data science sous le theme "optimisation d'un portefeuille de crypto utilisant le machine learning".
- Realiser par : Aboulaala Maria | Aberhouche Anass | Mohammed amine Mahdioui
- Encadrer par : Mr Najdi Lotfi
   """
)

st.subheader('Definition :book: :')
st.markdown(
    """
    >L'optimisation d'un portefeuille de crypto-monnaies en utilisant l'apprentissage automatique consiste à utiliser des algorithmes d'apprentissage automatique pour prédire la performance future des crypto-monnaies et ainsi ajuster la répartition des actifs dans le portefeuille pour maximiser les rendements et minimiser les risques.
   """
)

st.subheader(' Introduction:')
st.markdown(
    """
    >La régression logistique est un type d'analyse statistique qui permet de prédire une variable cible catégorique (comme un gain ou une perte) en fonction d'autres variables indépendantes. Elle peut être utilisée pour l'optimisation d'un portefeuille de crypto-monnaies en prédisant la performance future des crypto-monnaies et en ajustant la répartition des actifs dans le portefeuille en conséquence.
    Pour utiliser la régression logistique pour l'optimisation d'un portefeuille de crypto-monnaies, il faut d'abord:
    - :one: Collecter des données sur les prix historiques, les volumes de négociation et d'autres indicateurs pertinents pour les crypto-monnaies que l'on souhaite inclure dans le portefeuille.
    - :two: Entraîner le modèle de régression logistique qui prédit la performance future des crypto-monnaies a partir des donnees precedente.
    - :three: Une fois que le modèle est entraîné, il peut être utilisé pour répartir les actifs dans le portefeuille de manière à maximiser les rendements et minimiser les risques.
    Par exemple, on peut utiliser les prédictions du modèle pour investir davantage dans les crypto-monnaies qui sont prévues pour performer bien, tandis que les crypto-monnaies qui sont prévues pour performer moins bien peuvent être réduites ou éliminées du portefeuille.

   """
)

st.header(':one: Collecter les données ')
st.markdown(
    """
     **Definition**
    > La collecte de donnees se fait par yahoo finance qui est un site web permetant de suivre les cours en temps réel des actions, les actualités boursières, les données fondamentales des entreprises et les données économiques. Il permet également de suivre les portefeuilles et les actualités de certaines crypto monnaies.
   """
)
st.subheader("Le code pour collecter : :female-technologist: ")



code = '''
import yfinance as yf
import pandas as pd

# Définir les symboles des actions
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT','ADAUSDT','LTCUSDT','DOTUSDT']

# Définir la période pour laquelle les données doivent être récupérées
start_date = '2010-01-01'
end_date = '2020-01-01'

# Récupérer les données historiques pour chaque action
data = {}
for symbol in symbols:
    data[symbol] = yf.download(symbol, start=start_date, end=end_date)

# Concatenate the dataframes into a single one
df_ohlc = pd.concat(data.values(), keys=data.keys(), axis=1)

'''
st.code(code, 

language='python')

st.subheader('Le split des données en données d entraînement et de test:')

st.markdown(
    """
    **Definition**
    > Le split des données en données d'entraînement et de test est une étape cruciale dans le développement de modèles d'apprentissage automatique. Il permet de s'assurer que les modèles sont performants et généralisables aux données qu'ils n'ont pas encore vues, Les données d'entraînement sont utilisées pour entraîner le modèle. Les données de test, quant à elles, sont utilisées pour évaluer la performance du modèle une fois qu'il a été entraîné. En comparant les prédictions du modèle avec les valeurs réelles des données de test, on peut évaluer la qualité du modèle.
   """
)
code1 = '''
df_ohlc.index=pd.DatetimeIndex(df_ohlc.index)
df_Close=df_ohlc.swaplevel(axis=1)['Close']
instruments=df_ohlc.columns.get_level_values(0).unique()

df_train,df_validate,df_test=df_ohlc.loc[:'2022-01-01'],df_ohlc.loc['2022-01-01':'2022-09-01'],df_ohlc.loc['2022-09-01':]

'''
st.code(code1, 

language='python')

st.header(' Predicting Market Movements with Machine Learning ')
st.markdown(
    """
    **Explication:**
    > la bibliothèque scikit-learn pour importer la classe LogisticRegression, qui est utilisée pour la régression logistique, Ensuite, deux variables vides sont déclarées, LR_trained et LR_test. Ces variables sont des DataFrames vides qui pourraient être utilisées pour stocker les données d'entraînement et de test respectivement.
   """
)
code2 = '''
from sklearn.linear_model import LogisticRegression
LR_trained=pd.DataFrame()
LR_test=pd.DataFrame()
'''
st.code(code2, 
language='python')


st.markdown(
    """
    **Pre traitement de donnees:**
    > Les deux fonctions sont utilisées pour nettoyer et préparer les données pour l'entraînement et l'évaluation d'un modèle de régression logistique.
    - La première fonction, "log_return", prend en entrée le DataFrame et calcule la variation logarithmique des données. La variation logarithmique est utilisée pour normaliser les données et éviter les problèmes d'échelle qui peuvent survenir lorsque les données ont des ordres de grandeur différents. La fonction retourne les variations logarithmiques des données, remplissant les valeurs manquantes par 0.
    - La deuxième fonction, "get_clean_Xy", prend en entrée un DataFrame de données et retourne deux sous-ensembles de données: les données d'entraînement (X) et les données cibles (y). Les données d'entraînement sont obtenues en supprimant la première ligne du DataFrame d'entrée, car elle contient des valeurs manquantes après le calcul de la variation logarithmique. Les données cibles sont obtenues en utilisant la fonction de signe pour identifier les gains et les pertes (positifs et négatifs) sur la base des variations logarithmiques calculées
   """
)

code3 = '''
def log_return(data):
    return np.log(data/data.shift()).fillna(0)
def get_clean_Xy(df):
    return df.iloc[1:], np.sign(log_return(df.Close).iloc[1:])

'''
st.code(code3, 
language='python')

st.markdown(
    """
    **definir la strategie**
    > La classe LogesticMLStrategy une classe personnalisée qui hérite de la classe de base Strategy de la bibliothèque Backtesting. Elle est utilisée pour définir une stratégie de trading basée sur la régression logistique. 

    > La classe définit plusieurs variables d'instance qui peuvent être utilisées pour configurer le comportement de la stratégie, telles que le solveur utilisé pour entraîner le modèle de régression logistique, la pénalité utilisée pour réguler les paramètres du modèle, et le facteur de régulation C.

    > La méthode init est utilisée pour initialiser la stratégie en instanciant un objet de régression logistique en utilisant les paramètres de configuration spécifiés, puis en entraînant le modèle en utilisant les données d'entraînement spécifiées. Elle utilise également les fonctions "get_clean_Xy" et "log_return" pour nettoyer et préparer les données d'entraînement
   """
)
code4 = '''
from backtesting import Strategy
from backtesting import Backtest

class LogesticMLStrategy(Strategy):
    TrainOn=None
    solver='lbfgs'
    penalty='l2'
    c=7
    n_wait=3
    price_delta = .08
    def init(self):        
        self.lm=LogisticRegression(C=np.exp(self.c*np.log(10)), solver=self.solver)

        if self.TrainOn is None:
            self.TrainOn=self.data.df.drop(columns='Volume')
        # Train the classifier in advance
        X, y = get_clean_Xy(self.TrainOn)
        self.lm.fit(X,y)

        # Plot y for inspection
        self.I(lambda: np.sign(log_return(self.data.df.Close)), name='y_true')
        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

'''
st.code(code4, 
language='python')

st.markdown(
    """
    > La classe LogesticMLStrategy définit également la méthode next() qui est appelée à chaque itération de la boucle de trading. Cette méthode utilise les données en cours pour effectuer une prédiction de la prochaine variation des prix en utilisant le modèle de régression logistique entraîné précédemment.

    > Ensuite, la prédiction est utilisée pour décider si la stratégie doit acheter ou vendre en utilisant les critères de prise de décision spécifiés (n_wait, price_delta). Il utilise également les limites de stop loss pour fermer les positions qui ont été ouvertes depuis plus de deux jours.
   """
)

code5 = '''
def next(self):

        # Forecast the next movement
        X=self.data.df.iloc[-1:].fillna(0).drop(columns='Volume')
        forecast = self.lm.predict(X)[-1]

        # Update the plotted "forecast" indicator
        self.forecasts[-1] = forecast
        # Defining TP and SL values
        upper, lower = self.data.Close[-1] * (1 + np.r_[1, -1]*self.price_delta)

        if  sum(self.forecasts[-self.n_wait:])>1 and not self.position.is_long:
            self.buy(tp=upper, sl=lower)
        elif sum(self.forecasts[-self.n_wait:])<-1 and not self.position.is_short:
            self.sell(tp=lower, sl=upper)

        # Additionally, set aggressive stop-loss on trades that have been open 
        # for more than two days
        current_time = self.data.index[-1]
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('1 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, self.data.Low[-1])
                else:
                    trade.sl = min(trade.sl, self.data.High[-1])

'''
st.code(code5, 
language='python')

st.markdown(
    """
    **Le ratio de sharpe**
    > La fonction Metric_to_optimse prend en entrée une série de données de rendements d'un portefeuille, généralement obtenues à partir d'une simulation de trading. La fonction retourne une métrique spécifique qui est utilisée pour optimiser la performance du portefeuille, dans ce cas spécifique c'est le "Sharpe Ratio"

    > Le Sharpe Ratio est un indicateur de performance utilisé pour mesurer le rendement d'un portefeuille par rapport à un actif sans risque, tel que les bons du Trésor américain. Il mesure le rendement excessif d'un portefeuille par rapport à un actif sans risque, en tenant compte du risque pris. Plus le ratio est élevé, plus le rendement est élevé par rapport au risque.
   """
)

code5 = '''
def Metric_to_optimse(series):
    return series['Sharpe Ratio']

'''
st.code(code5, 
language='python')



st.header(':two: Entrainer le modele ')

st.markdown(
    """
    > utilisent la bibliothèque Backtrader pour entraîner un modèle de régression logistique pour chaque instrument de la liste "instruments" en utilisant les données de formation spécifiées dans le DataFrame "df_train".

    > La boucle for itère sur chaque instrument dans la liste "instruments" et utilise les données de formation correspondantes pour instancier un objet "Backtest", qui utilise la stratégie de régression logistique personnalisée définie précédemment. Il utilise également la méthode "optimize" pour optimiser les paramètres de la stratégie en utilisant la métrique de performance spécifiée dans la fonction "Metric_to_optimse" pour maximiser la performance du modèle. La méthode d'optimisation utilisée ici est "skopt".

    > Pour chaque instrument, un "Equity Curve" est créé et stocké dans une variable, et les paramètres optimaux sont enregistrés dans un dictionnaire "parameters". Les paramètres optimaux sont utilisés pour entraîner le modèle pour les données de test pour évaluer la performance du modèle sur des données indépendantes.
   """
)

code6 = '''
from backtesting import Strategy
from backtesting import Backtest
from sklearn.linear_model import LogisticRegression

parameters={}
for instrument in instruments:
    data = df_train[instrument]
    bt_Train = Backtest(data, LogesticMLStrategy , cash=1000000)
    stats_Train = bt_Train.optimize(
        n_wait=(1,5,1),
        price_delta=(0,1,0.01),
        # c = (7,10,1),
        maximize=Metric_to_optimse,
        method='skopt',
    )
    LR_trained[instrument]=stats_Train._equity_curve.Equity/1000000
    params={'c':stats_Train._strategy.c,'solver':stats_Train._strategy.solver}
    parameters[instrument]=params
'''
st.code(code6, 
language='python')






st.markdown(
    """
    **Tester le modele**
    >On utilise la bibliothèque Backtrader pour tester le modèle de régression logistique entraînés sur les données de validation spécifiées dans le DataFrame "df_validate".

    > La boucle for itère sur chaque instrument dans la liste "instruments" et utilise les données de validation correspondantes pour instancier un objet "Backtest", qui utilise la stratégie de régression logistique personnalisée définie précédemment. Il utilise également les paramètres optimaux trouvés lors de l'entraînement pour tester le modèle sur les données de validation.

    >Pour chaque instrument, un "Equity Curve" est créé et stocké dans une variable LR_test. Le plot de l'équity curve est ensuite affiché pour visualiser les résultats obtenus.
   """
)

code7 = '''
for instrument in instruments:
    bt=Backtest(df_validate[instrument], LogesticMLStrategy , cash=1000000)
    stats=bt.run(TrainOn=df_train[instrument],**parameters[instrument])
    LR_test[instrument]=stats._equity_curve.Equity/1000000
    print(instrument)
    bt.plot()
    
'''
st.code(code7, 
language='python')



st.header(':three: Affichage des performances du modele')

st.markdown(
    """
    > On utilise la méthode plot() pour afficher un graphique des performances des modèles de régression logistique testés sur les données de validation pour chaque instrument. Il affichera un graphique comparant les performances de chaque instrument sur la même échelle pour une analyse plus facile.

    > Cela permet de visualiser de manière globale les résultats des différents modèles testés. Il est important de noter qu'il est important de considérer les différents facteurs tels que les risques, les coûts de transaction, les impôts et les frais pour évaluer la performance réelle d'un portefeuille.
   """
)

code8 = '''
LR_test.plot()
'''
st.code(code8, 
language='python')

st.image('GRAPH.png')


st.markdown(
    """
    > la méthode mean() est utilisée pour calculer la moyenne des performances de tous les instruments pour obtenir une estimation globale de la performance des modèles testés.
   """
)

code9 = '''
LR_test.iloc[-1].mean()
'''
st.code(code9, 
language='python')

st.success('1.3259550248963596')

st.markdown(
    """
    > enregistrer les données de performance de l'entraînement et de test des modèles de régression logistique dans des fichiers CSV (Comma Separated Values) pour une utilisation future. Les données sont enregistrées dans les fichiers 'OptimizedLR_trained.csv' et 'OptimizedLR_test.csv' respectivement dans le répertoire 'ModelsResult'.

    > La dernière ligne de code affiche les paramètres optimaux utilisés pour entraîner et tester les modèles pour chaque instrument, ceci est utile pour conserver une trace des paramètres utilisés pour chaque instrument et pour enregistrer les résultats de l'optimisation pour une utilisation future.
   """
)

code10 = '''
LR_trained.to_csv('ModelsResult/OptimizedLR_trained.csv')
LR_test.to_csv('ModelsResult/OptimizedLR_test.csv')
parameters
'''
st.code(code10, 
language='python')

st.image('results.png')

st.title(':pushpin: Phase finale')
st.header('Optimiation du modele')


st.markdown(
    """
    > Le code contient:
    - La première fonction "mean_ret" prend en entrée un dataframe et une longueur, elle calcule la somme des rendements sur la période spécifiée par la longueur et retourne la moyenne des rendements.
    - La seconde fonction "monthdelta" prend en entrée une date et un delta (nombre de mois), elle retourne la date qui est delta mois après la date d'entrée.
    - La troisième fonction "windowGenerator" prend en entrée un dataframe, une longueur de fenêtre, une longueur de horizon, un pas de déplacement et un booléen (cummulative). Elle retourne une liste de fenêtres temporelles qui peuvent être utilisées pour les entraînements et les tests pour les modèles de machine learning.
    > La fonction windowGenerator utilise la fonction monthdelta pour générer les fenêtres temporelles en fonction de la longueur de fenêtre, de la longueur de horizon et du pas de déplacement spécifiés. Elle retourne également une liste des fenêtres d'horizon qui peuvent être utilisées pour évaluer la performance des modèles entraînés sur les fenêtres temporelles.
   """
)

code11 = '''
#Mean returns function
def mean_ret(data,length):
    return data.sum()/length

def monthdelta (date,delta):
    m,y = (date.mounth+delta)%12 , date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31, 29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    new_date = (date.replace(day=d,month=m, year=y))
    return parse(new_date.strftime('%Y-%m-%d'))

def windowGenerator (dataframe, lookback, horizon, step, cummulative = False):
    if cummulative:
        c = lookback
        step = horizon
        
    initial = min(dataframe.index)
    windows = []
    horizons = []

    while initial <= monthdelta(max(dataframe.index), -lookback):
        windowStart = initial
        windowEnd = monthdelta(windowStart, lookback)
        if cummulative:
            windowStart = min(dataframe.index)
            windowEnd = monthdelta(windowStart, c) + timedelta(days=1)
            c += horizon
        horizonStart = windowEnd + timedelta(days=1)
        horizonEnd = monthdelta(horizonStart, horizon)

        lookbackWindow = dataframe[windowStart:windowEnd]
        horizonWindow = dataframe[horizonStart:horizonEnd]

        windows.append(lookbackWindow)
        horizons.append(horizonWindow)

        initial = monthdelta(initial, step)

    return windows, horizons
'''
st.code(code11, 
language='python')

st.markdown(
    """
    > le code définit deux fonctions :
    - La première fonction "actual_return" prend en entrée les rendements réels et les poids d'un portefeuille, calcule le rendement et la covariance du portefeuille et retourne ces valeurs.
    - La deuxième fonction "optimisation" prend en entrée les rendements prévus, les rendements réels, les paramètres lam1 et lam2. Elle utilise ces données pour optimiser les poids de portefeuille en utilisant la méthode de minimisation de scipy
    > La fonction optimisation utilise des contraintes d'égalité pour garantir que la somme des poids de chaque actif est égale à 1. La fonction utilise également un critère de coût pour maximiser le ratio de sharpe en minimisant la variance du portefeuille.
    Enfin, la fonction retourne un dictionnaire qui contient les poids de portefeuille optimaux, les rendements et les variances prévus et réels et le ratio de sharpe. Il est important de noter que les rendements prévus utilisés pour l'optimisation doivent être obtenus à partir d'un modèle de prévision efficace pour obtenir des résultats pertinents. Il est également important de considérer les limites de risque et les coûts de transaction lors de la mise en œuvre de cette méthode
   """
)

code12 = '''
def actual_return(actual_ret, weight):
    mean_return = mean_ret(actual_ret,actual_ret.shape[0])
    actual_cov=actual_ret.cov()

    portfolio_return = mean_ret.T.dot(weight)
    protfolio_cov = weight.T.dot(actual_cov).dot(weight)
    return portfolio_return,protfolio_cov

def optimisation(predicted_ret, actual_ret, lam1, lam2):
    mean_return = mean_ret(predicted_ret, predicted_ret.shape[0])
    predicted_covariance = predicted_ret.cov()
#Cost function
    def f(weight):
        return -(mean_return.T.dot(weight) - lam1*(weight.T.dot(predicted_covariance).dot(weight)) + lam2*norm(weight, ord=1))

    opt_bounds = Bounds(0, 1)

#Equality Constraints
    def h(weight):
        return sum(weight) - 1

#Constraints Dictionary
    cons = ({
        'type' : 'eq',
      'fun' : lambda weight: h(weight)
  })

#Solver
    sol = minimize(f,
                 x0 = np.ones(mean_return.shape[0]),
                 constraints = cons,
                 bounds = opt_bounds,
                 options = {'disp': False},
                 tol=10e-10)  

#Predicted Results
    weight = sol.x
    predicted_portfolio_ret = weight.dot(mean_return)
    portfolio_STD = weight.T.dot(predicted_covariance).dot(weight)
  
#Actual Results
    portfolio_actual_returns, portfolio_actual_variance = actual_return(actual_ret, weight)
    sharpe_ratio = portfolio_actual_returns/np.std(portfolio_actual_variance)

    ret_dict = {'weights' : weight,
              'predicted_returns' : predicted_portfolio_ret,
              'predicted_variance' : portfolio_STD,
              'actual_returns' : portfolio_actual_returns,
              'actual_variance' : portfolio_actual_variance,
              'sharpe_ratio': sharpe_ratio}
  
    return ret_dict
'''
st.code(code12, 
language='python')