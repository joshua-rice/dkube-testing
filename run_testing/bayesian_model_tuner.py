import numpy as np

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import get_scorer
from datetime import datetime
from inspect import isclass

import warnings
warnings.filterwarnings("ignore")

class BayesianModelTuner:
    """
    Class definition for the hyperparameter tuning wrapper.
    """

    ## class init method
    def __init__(self, search_params, model_class, metadata, fixed_params=None,
                 fit_params=None, maximize=True):
        """
        Class init method

        Parameters
        ----------

        search_params : dict
            The parameter space to be searched during optimization

        model_class : recognized model class
            Model class object to be tuned. Note that the model class should have already 
            been imported, but not initialized, when passed as a parameter here.

        metadata : dict
            Metadata associated with the experiment being ran.

        fixed_params : dict
            A set of fixed parameters required for initializing a model, but that don't 
            require tuning. Defaults to None, a dict should be passed otherwise.

        fit_params : dict
            An optional dictionary with additional parameters to be passed to a model 
            object during fitting. Useful for leveraging capabilities such as early 
            stopping in XGBoost and lightGBM.

        maximize : boolean
            Whether the problem should be treated as a minimization problem or a 
            maximization problem. Defaults to True (maximization), set to False for a 
            minimization problem.
        """
        # check for existence of required params
        if not search_params:
            raise Exception("Required input 'search_params' not provided.")

        if not model_class:
            raise Exception("Requried input 'model_class' not provided.")

        if not metadata:
            raise Exception("Required input 'metadata' not provided.")


        # type checks on params
        if not isclass(model_class):
            raise Exception("Input 'model_class' not recognized.")

        if type(search_params) != dict:
            raise Exception("A dict was not provided for input parameter 'search_params'.")

        if fixed_params != None and type(fixed_params) != dict:
            raise Exception("A dict was not provided for input parameter 'fixed_params'.")

        if fit_params != None and type(fit_params) != dict:
            raise Exception("A dict was not provided for input parameter 'fit_params'.")

        if type(maximize) != bool:
            raise Exception("A boolean was not provided for input parameter 'maximize'.")


        # create attributes from passed params
        self.search_params = search_params
        self.fixed_params = fixed_params
        self.model_class = model_class
        self.metadata = metadata
        self.maximize = maximize
        self.fit_params = fit_params

        # create empty Trials object to be populated during tuning
        self.trials = Trials()


    ## tuning execution and summarization
    def tune(self, X, y, n_iter, scoring_string):
        """
        Class method to run the actual hyperparameter optimization step.

        Parameters
        ----------
        X : pandas datafream or numpy array
            Feature df or array

        y : pandas series or numpy array
            Model target (response)

        n_iter : int
            How many optimization iterations to run?

        scoring_string : str
            A string reference to a scoring metric recognized by sklearn and its 
            'cross_val_score()' function.
        """
        ## check on scoring string
        try:
            score_check = get_scorer(scoring_string)
        except:
            raise Exception("Input 'scoring_string' not recognized by sklearn.")

        
        ## internal method - objective function definition focused on optimizing CV score
        def objective(params):
            # initialize model with logical check to control usage of fixed params
            if self.fixed_params == None:
                temp_model = self.model_class(**params)
            else:
                temp_model = self.model_class(**self.fixed_params, **params)

            # run CV with check on usage of fit params
            if self.fit_params == None:
                score = np.mean(cross_val_score(temp_model, X, y, cv=5, 
                                                scoring=scoring_string))
            else:
                score = np.mean(cross_val_score(temp_model, X, y, cv=5, 
                                                scoring=scoring_string,
                                                fit_params=self.fit_params))

            # return score with check on optimization type (max vs. min)
            if self.maximize == True:
                return {"loss":-score, "status":STATUS_OK}
            if self.maximize == False:
                return {"loss":score, "status":STATUS_OK}


        ## run the actual tuning
        best = fmin(fn=objective, space=self.search_params, trials=self.trials,
                    algo=tpe.suggest, max_evals=n_iter)
        hyperparams = space_eval(self.search_params, best)


        ## summarize the results and populate additional class attributes
        self.n_iter_to_best = self.trials.best_trial["tid"]
        self.best_score = np.abs(self.trials.best_trial["result"]["loss"])
        self.best_params = hyperparams


        ## generate summary object and assign it to a separate class attribute for easy 
        # retireval
        self.summary = {
            "n_iter_to_best": self.n_iter_to_best,
            "scoring_method": scoring_string,
            "best_score": self.best_score,
            "best_params": self.best_params
        }