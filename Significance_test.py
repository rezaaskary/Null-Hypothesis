def Regression(X,Y,alpha):      
    """
    both continuous t test for independent variables

    Parameters
    ----------
    X : TYPE
        Input values.
    Y : TYPE
        Output values.
    alpha : TYPE
        P_value threshold

    Returns
    -------
    t_stat : TYPE
        t statistical test value.
    df : TYPE
        degree of freedom.
    cv : TYPE
        The critical value
    P_val : TYPE
        P value
    R^2 : 
            TYPE
    R squared used for regression evaluation

    """
    
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
def Students_T_test(X,Y,alpha):         
    """
    Students_T_test for dependent variables regression
    """
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
def Chi_Square(xCat,yCat,debug = 'N'):
    """
    Parameters
    ----------
    # for both categorical variables    xCat : TYPE
        DESCRIPTION.
    # input categorical feature    yCat : TYPE
        DESCRIPTION.
    # input categorical target variable    debug : TYPE, optional
        DESCRIPTION. The default is 'N'     # debugging flag (Y/N).

    Returns
    chiSqStat, chiSqDf, chiSqSig,cramerV
    None.

    """
     
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
    """
    # input interval feature
    # input categorical target variable

    Parameters
    ----------
    xInt : TYPE
        DESCRIPTION.
    yCat : TYPE
        DESCRIPTION.
    debug : TYPE, optional
        DESCRIPTION. The default is 'N'.

    Returns
    -------
    None.

    """
    
    
    import pandas as pd
    import numpy as np
    import scipy
    import statsmodels.api as smodel
    #########################################################################
  
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
def Anova_test(X,Y): 
    """
           # one way anova test for categorical input and continuous output
    Parameters
    ----------
    X : TYPE
        input X.
    Y : TYPE
        Output Y

    Returns
    -------
    F_value : TYPE
        DESCRIPTION.
    P_value : TYPE
        DESCRIPTION.

    """
    import scipy.stats as stats
    (F_value,P_value)=stats.f_oneway(X,Y)
    return (F_value,P_value)
#################################################################################
def SHAPIRO_Wilk(data,P_value_threshold):
    """
    #     Tests whether a data sample has a Gaussian distribution.
    #     Assumptions
    #     Observations in each sample are independent and identically distributed (iid).
    # Interpretation
    
    # H0: the sample has a Gaussian distribution.
    # H1: the sample does not have a Gaussian distribution.
    Reference: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
    """
    from scipy.stats import shapiro
    stat, p = shapiro(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
        print('Probably Gaussian')
    else:
	    print('Probably not Gaussian')
    return stat,p
###################################################################################    
def D_Agostino_s_Ksquared_Test(data,P_value_threshold):
    """
    Tests whether a data sample has a Gaussian distribution.
    Assumptions
    Observations in each sample are independent and identically distributed (iid).
    Interpretation
    
    H0: the sample has a Gaussian distribution.
    H1: the sample does not have a Gaussian distribution.
    Reference: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
    Returns
    -------
    None.

    """
    from scipy.stats import normaltest
    stat, p = normaltest(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably Gaussian')
    else:
	    print('Probably not Gaussian')
    return stat,p
#######################################################################
def Anderson_Darling_Test(data,P_value_threshold):
    """
    Tests whether a data sample has a Gaussian distribution.

    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Interpretation
    
    H0: the sample has a Gaussian distribution.
    H1: the sample does not have a Gaussian distribution.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    from scipy.stats import anderson
    result = anderson(data)
    print('stat=%.3f' % (result.statistic))
    for i in range(len(result.critical_values)):
         sl,cv = result.significance_level[i], result.critical_values[i]
         if result.statistic < cv:
             print('Probably Gaussian at the %.1f%% level' % (sl))
         else:
             print('Probably not Gaussian at the %.1f%% level' % (sl))
#####################################################################################
def Spearmans_Rank_Correlation(data_1,data_2,P_value_threshold):
    """
    Tests whether two samples have a monotonic relationship.

    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample can be ranked.
    Interpretation
    
    H0: the two samples are independent.
    H1: there is a dependency between the samples.
    Ref: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

    Returns
    -------
    None.

    """
    from scipy.stats import spearmanr
    stat, p = spearmanr(data_1, data_2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably independent')
    else:
    	print('Probably dependent')
    return stat, p
############################################################################
def Kendalls_Rank_Correlation(data_1,data_2,P_value_threshold):
    """
    Tests whether two samples have a monotonic relationship.

    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample can be ranked.
    Interpretation
    
    H0: the two samples are independent.
    H1: there is a dependency between the samples.
    Ref: https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
        Returns
        -------
        None.

    """
    from scipy.stats import kendalltau
    stat, p = kendalltau(data_1, data_2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably independent')
    else:
    	print('Probably dependent')
    return  stat, p 
##############################################################################
def Chi_Squared_Test(data,P_value_threshold):
    """
    Tests whether two categorical variables are related or independent.

    Assumptions
    
    Observations used in the calculation of the contingency table are independent.
    25 or more examples in each cell of the contingency table.
    Interpretation
    
    H0: the two samples are independent.
    H1: there is a dependency between the samples.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from scipy.stats import chi2_contingency

    stat, p, dof, expected = chi2_contingency(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably independent')
    else:
    	print('Probably dependent')
    return stat, p, dof, expected
##############################################################################
def Augmented_Dickey_Fuller_Unit_Root_Test(data,P_value_threshold):
    """
    Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive.

    Assumptions
    
    Observations in are temporally ordered.
    Interpretation
    
    H0: a unit root is present (series is non-stationary).
    H1: a unit root is not present (series is stationary).

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from statsmodels.tsa.stattools import adfuller
    stat, p, lags, obs, crit, t = adfuller(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably not Stationary')
    else:
    	print('Probably Stationary')
    return stat, p, lags, obs, crit, t
################################################################################
def Kwiatkowski_Phillips_Schmidt_Shin(data,P_value_threshold):
    """
    Tests whether a time series is trend stationary or not.

    Assumptions
    
    Observations in are temporally ordered.
    Interpretation
    
    H0: the time series is not trend-stationary.
    H1: the time series is trend-stationary.
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from statsmodels.tsa.stattools import kpss
    stat, p, lags, crit = kpss(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably not Stationary')
    else:
    	print('Probably Stationary')
    return stat, p, lags, crit
###################################################################
def Paired_independent_student_t_test(data_1,data_2,P_value_threshold,mode):
    """
    Tests whether the means of two independent samples are significantly different.
    Assumptions

    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample are normally distributed.
    Observations in each sample have the same variance.
    Interpretation
    
    H0: the means of the samples are equal.
    H1: the means of the samples are unequal.
    
    ############################################################    
    Tests whether the means of two paired samples are significantly different.
    
    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample are normally distributed.
    Observations in each sample have the same variance.
    Observations across each sample are paired.
    Interpretation
    
    H0: the means of the samples are equal.
    H1: the means of the samples are unequal.





    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.
    mode : "independent" or "dependent"
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if mode=="independent":
        from scipy.stats import ttest_ind
        stat, p = ttest_ind(data_1,data_2)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > P_value_threshold:
        	print('Probably the same distribution')
        else:
        	print('Probably different distributions')
        return stat, p
    elif mode=="dependent":
        from scipy.stats import ttest_rel

        stat, p = ttest_rel(data_1,data_2)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p >P_value_threshold:
        	print('Probably the same distribution')
        else:
        	print('Probably different distributions')
        return stat, p
##################################################################################
def Mann_Whitney_U_Test(data_1,data_2,P_value_threshold):
    """
    Tests whether the distributions of two independent samples are equal or not.

    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample can be ranked.
    Interpretation
    
    H0: the distributions of both samples are equal.
    H1: the distributions of both samples are not equal.
    

    Parameters
    ----------
    data_1 : TYPE
        DESCRIPTION.
    data_2 : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(data_1,data_2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably the same distribution')
    else:
    	print('Probably different distributions')
    return stat, p
##################################################################################
def Wilcoxon_Signed_Rank_Test(data_1,data_2,P_value_threshold):
    """
    Tests whether the distributions of two paired samples are equal or not.

    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample can be ranked.
    Observations across each sample are paired.
    Interpretation
    
    H0: the distributions of both samples are equal.
    H1: the distributions of both samples are not equal.

    Parameters
    ----------
    data_1 : TYPE
        DESCRIPTION.
    data_2 : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from scipy.stats import wilcoxon

    stat, p = wilcoxon(data_1,data_2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably the same distribution')
    else:
    	print('Probably different distributions')
    return stat, p
##################################################################################   
def Kruskal_Wallis_H_Test(data_1,data_2,P_value_threshold):
    """
    Tests whether the distributions of two paired samples are equal or not.

    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample can be ranked.
    Observations across each sample are paired.
    Interpretation
    
    H0: the distributions of both samples are equal.
    H1: the distributions of both samples are not equal.

    Parameters
    ----------
    data_1 : TYPE
        DESCRIPTION.
    data_2 : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from scipy.stats import kruskal
    stat, p = kruskal(data_1,data_2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably the same distribution')
    else:
    	print('Probably different distributions')
    return stat, p
##################################################################################
def Friedman_Test(data,P_value_threshold):
    """
    Tests whether the distributions of two or more paired samples are equal or not.

    Assumptions
    
    Observations in each sample are independent and identically distributed (iid).
    Observations in each sample can be ranked.
    Observations across each sample are paired.
    Interpretation
    
    H0: the distributions of all samples are equal.
    H1: the distributions of one or more samples are not equal

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    P_value_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
        
    
    from scipy.stats import friedmanchisquare

    stat, p = friedmanchisquare(data[:,0],data[:,1])
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > P_value_threshold:
    	print('Probably the same distribution')
    else:
    	print('Probably different distributions')
    return stat, p
#####################################################################################
        