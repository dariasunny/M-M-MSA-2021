import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.formula.api import ols
import sklearn
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.optimize import minimize
import scipy.stats as stats
import pylab 
from scipy.stats import norm
from scipy.stats import kde
from scipy.stats import gamma
import scipy
from scipy import stats


df = pd.read_csv("SUPERLAST.csv")

df = df[["date", "Adj Close",'gtrends','Comments_int','tweet_num','meme_Reddit']]

#Step 1
#figure 1
plt.figure(figsize=(10, 8))
plt.scatter(df['meme_Reddit'],df['Adj Close'])
plt.xticks(rotation=45)
plt.xlabel(u'meme_Reddit', fontsize = 20)
plt.ylabel(u'Adj Close', fontsize = 20)
plt.show()

#figure 2
plt.figure(figsize=(10, 8))
plt.scatter(df['tweet_num'],df['Adj Close'])
plt.xticks(rotation=45)
plt.xlabel(u'tweet_num', fontsize = 20)
plt.ylabel(u'Adj Close', fontsize = 20)
plt.show()

#figure 3
plt.figure(figsize=(10, 8))
plt.scatter(df['Comments_int'],df['Adj Close'])
plt.xticks(rotation=45)
plt.xlabel(u'Comments_int', fontsize = 20)
plt.ylabel(u'Adj Close', fontsize = 20)
plt.show()


#Step 2



for i in df.columns[1:]:
    x = df[str(i)]
    density = kde.gaussian_kde(x)
    xgrid = np.linspace(x.min(), x.max(), 100)
    plt.hist(x, bins=8,density=True, stacked=True)
    plt.plot(xgrid, density(xgrid), 'r-')
    plt.show()



#Analog
#sns.distplot(df.gtrends, kde=True, norm_hist=True)


#Step 3



print(min(df['gtrends']), min(df['Adj Close']), min(df['Comments_int']), min(df['tweet_num']), min(df['meme_Reddit']))
print(max(df['gtrends']), max(df['Adj Close']), max(df['Comments_int']), max(df['tweet_num']), max(df['meme_Reddit']))
print(df['gtrends'].median(), df['Adj Close'].median(), df['Comments_int'].median(), df['tweet_num'].median(),df['meme_Reddit'].median())
print(np.quantile(df['gtrends'], 0.25), np.quantile(df['Adj Close'], 0.25), np.quantile(df['Comments_int'], 0.25), np.quantile(df['tweet_num'], 0.25),  np.quantile(df['meme_Reddit'], 0.25))
print(np.quantile(df['gtrends'], 0.75), np.quantile(df['Adj Close'], 0.75), np.quantile(df['Comments_int'], 0.75), np.quantile(df['tweet_num'], 0.75),  np.quantile(df['meme_Reddit'], 0.75))

plt.boxplot(df.gtrends)
for i in df.columns[2:12]:
    plt.boxplot(df[str(i)])
    plt.title(str(i)+ " " + "boxplot" )
    plt.show()  

# Step 4

dist_name = 'norm'
data = df.gtrends
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme", "exponpow", "gamma"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))
    print(params)
    return best_dist, best_p, params[best_dist]

data_names = [df.gtrends, df.Comments_int, df.tweet_num, df.meme_Reddit, df['Adj Close']]

for data in data_names:
  print(data.name)
  get_best_distribution(data)
  print(' ')


# Estimating wil MLE

def getLL_normal(params, data):
    mu,sigma = params
    neg_log_lik = -np.sum(np.log(1/(sigma*np.sqrt(2*np.pi))) - 1/2 * ((data - mu)/sigma)**2)
    return neg_log_lik


guess = np.array([1,3])
results_ML = minimize(getLL_normal, [1,3], args = (df.gtrends))

mu_est_ML, sigma_est_ML = results_ML.x

#estimating with OLS

def OLS(params, data):
    mu,sigma = params
    quantiles = [0.15,0.25,0.35,0.45,0.5,0.65,0.75,0.85,0.95]
    #s = np.random.normal(mu, sigma, 1000)
    err = 0
    for i in quantiles:
        err += (np.quantile(data, i) - norm.ppf(i, loc=mu, scale=sigma))**2
    return err




results_OLS = minimize(OLS, [12,13], args = (df.gtrends),method = 'Powell')
mu_est_OLS, sigma_est_OLS = results_OLS.x

#with standart func
norm.fit(df.gtrends)


x = np.arange(min(df.gtrends), max(df.gtrends), 1)
plt.plot(x, norm.pdf(x, mu_est_OLS, sigma_est_OLS),label="OLS")
plt.plot(x, norm.pdf(x, mu_est_ML, sigma_est_ML),label = "ML")
plt.hist(df.gtrends, density=True)
plt.legend()





# Step 5

stats.probplot(df.gtrends, dist="norm", plot=pylab)
pylab.show()


# Step 6

#kolmogorov test
stats.kstest(df.gtrends, np.random.normal(mu_est, sigma_est, 1000)  ,alternative='two-sided', mode='auto')

#Chi-Squared

chi2 = scipy.stats.chisquare(df.gtrends)

#Wilcoxon rank-sum
scipy.stats.ranksums(df.gtrends, np.random.normal(13, 12, 1000))
