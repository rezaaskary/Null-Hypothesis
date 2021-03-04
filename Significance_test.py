def Regression(X,Y,alpha):        # both continuous t test for independent variables
    import numpy as np
    from scipy import stats
    mean1, mean2 = np.mean(X), np.mean(Y)
    std1, std2 = np.std(X, ddof=1), np.std(Y, ddof=1)
    n1, n2 = len(X), len(Y)
    se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2)
	# calculate standard errors
    se1= stats.sem(X)
    se2= stats.sem(Y)
	# standard error on the difference between the samples
    sed = np.sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
    t_stat = (mean1 - mean2) / sed
	# degrees of freedom
    df = len(X) + len(Y) - 2
	# calculate the critical value
    cv = stats.t.ppf(1.0 - alpha, df)
	# calculate the p-value
    P_val = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
	# return everything
    R_2=stats.pearsonr(X, Y)
    ###############
    # t2, p2 = stats.ttest_ind(X,Y, equal_var = False)
    # print("t = " + str(t2))
    # print("p = " + str(p2))

    return t_stat, df, cv, P_val,R_2[0]**2
###############################################################################################
def Students_T_test(X,Y,alpha):         # Students_T_test for dependent variables regression
    import numpy as np
    from scipy import stats
    mean1, mean2 = np.mean(X), np.mean(Y)
	# number of paired samples
    n = len(X)
	# sum squared difference between observations
    d1 = sum([(X[i]-Y[i])**2 for i in range(n)])
	# sum difference between observations
    d2 = sum([X[i]-Y[i] for i in range(n)])
	# standard deviation of the difference between means
    sd = np.sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
    sed = sd / np.sqrt(n)
	# calculate the t statistic
    t_stat = (mean1 - mean2) / sed
	# degrees of freedom
    df = n - 1
	# calculate the critical value
    cv = stats.t.ppf(1.0 - alpha, df)
	# calculate the p-value
    p = (1.0 - stats.t.cdf(abs(t_stat), df)) * 2.0
	# return everything
    return t_stat, df, cv, p
####################################################################
def Chi_Square(     # for both categorical variables
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):
    import pandas as pd
    import numpy as np
    import scipy
    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)
    cramerV = chiSqStat / nTotal
    
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig,cramerV)
################################################################################################
def DevianceTest (xInt,yCat,debug = 'N'):
    # input interval feature
    # input categorical target variable
    # debugging flag (Y/N) 
    # import sklearn
    # from sklearn import metrics
    # from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import numpy as np
    import scipy
    import statsmodels.api as smodel
    #########################################################################
    # New_y=np.zeros(yCat.shape)  
    # New_y[yCat==2]=1
    # New_y[yCat==3]=2
    # New_y[yCat==5]=3
    # del yCat
    # yCat=New_y
    #########################################################################    
    if type(yCat)==pd.core.frame.DataFrame:
        pass
    else:
        yCat=pd.DataFrame(yCat, index=None)

    y = yCat.astype('category')

    # # Model 0 is yCat = Intercept
    X = np.where(yCat.notnull(), 1, 0)
    objLogit = smodel.MNLogit(y, X)
    thisFit = objLogit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK0 = objLogit.loglike(thisParameter.values)

    # Debugging codes omitted to enhance readability
    # Model 1 is yCat = Intercept + xInt
    X = smodel.add_constant(xInt, prepend = True)
    objLogit = smodel.MNLogit(y, X)
    thisFit = objLogit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK1 = objLogit.loglike(thisParameter.values)
    # Debugging codes omitted to enhance readability
    # Calculate the deviance
    devianceStat = 2.0 * (LLK1 - LLK0)
    # devianceDf = (len(y.cat.categories) - 1.0)
    devianceDf = (len(np.unique(y.values)) - 1.0)
    devianceSig = scipy.stats.chi2.sf(devianceStat, devianceDf)
    mcFaddenRSq = 1.0 - (LLK1 / LLK0)

    return(devianceStat, devianceDf, devianceSig,mcFaddenRSq)
###########################################################################################
def Anova_test(X,Y):        # one way anova test for categorical input and continuous output
    import scipy.stats as stats
    (F_value,P_value)=stats.f_oneway(X,Y)
    return (F_value,P_value)
#################################################################################
    
    
    
    
    
    
    
    
    
    