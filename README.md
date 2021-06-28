# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
We use a bank marketing dataset containing selected features related to a Portuguese retail bank's telemarketing campaigns
(telemarketing attributes, product details, client information) enriched with social and economic influence features. We seek to 
**predict the success of telemarketing calls** for selling bank long-term deposits.

The standard Scikit-learn Logistic Regression model reached accuracy 0.898, which was higher than accuracy of the AutoML model (0.739), as its hyperparameters were optimized for accuracy. On the other side, the AutoML model showed AUC 0.737, which is significantly higher than AUC of the Logistic Regression model (0.583). As AUC is more relevant metrics for classification tasks, we can choose the AutoML model as more appropriate.

![Models' Metrics (click to see the image)](img/AML-Results.png?raw=true)

Figure 1: Accuracy and AUC of two models

## Scikit-learn Pipeline
The Scikit-learn pipeline uses the Logistic Regression model, which, despite its name, is a linear model for classification rather than regression. The learning dataset is a subset of the bank marketing dataset (see References below). An exploratory data analysis showed the strong correlation between the feature "duration" and the target. To prevent data leakage, features related with the last contact in the current campaign, including "duration", were deleted, as in case of preparing a new campaign this features will not be available. This decision means worse performance of the model during training, but possibly better performance in production.

The script for hyperparameter tuning prepares the data (function `get_X_y()` instead of `clean_data()` from the starter file), splits data into train and test sets, instantiates the model (`LogisticRegression`) applying the regularization strength and max iterations arguments, trains the model, evaluates the accuracy of the model and saves the model to the output folder.

For hyperparametr tuning itself, the pipeline uses the random hyperparameter sampler, as it supports discrete and continuous hyperparameters and also early termination of low-performance runs, which was enabled by the bandit policy. Benefit of both these design decisions is lower costs of the experiments. Figure 2 shows the parameters tested in the experiment.

![HyperDrive Accuracy (click to see the image)](img/AML-HyperDrive-Acuracy-2.png?raw=true)

Figure 2: HyperDrive optimizing for accuracy

To compare the performance of the Logistic Regression model with the AzureML Model, figure 3 shows the ROC curve and AUC using same data and algorithms as in the script, the best parameters provided by the HyperDrive, and the `roc_curve` and `roc_auc_score` functions from the sklearn.metrics module.

![LR ROC Curve (click to see the image)](img/AML-ROC-LR.png?raw=true)

Figure 3: ROC curve of the Logistic Regression

## AutoML
The Azure Machine Learning SDK with AutoML automates model creation, hyperparametr tuning, and model interpretation. The AutoML pipeline uses the same reduced dataset as the Scikit-learn pipeline. The function `get_X_y` in the AutoML context concatenates X and y to one dataset. Instead of splitting data to train and test datasets, the AutoML pipeline uses the cross-validation algorithm. The number of cross-validations as well as other parameters for AutoML are set in the AutoML configuration. Figure 4 shows the resulting voting ensemble of models provided by the AutoML experiment.

![Voting Ensemble Clfs (click to see the image)](img/AML-VE-clfs.png?raw=true)

Figure 4: Classifiers and their weights in the Voting Ensemble

The resulting AutoML model includes `XGBoostClassifier`, `RandomForest`, and `LightDBM` classifiers, but no `LogisticRegression` class. Figure 5 documents that these classifiers has better AUC even with default parameters then `LogisticRegression` with tuned parameters.

![ROC of Voting Ensamble Clfs (click to see the image)](img/AML-VE-clfs-ROC.png?raw=true)

Figure 5: ROC curves of the Voting Ensamble

Figure 6 shows the resulting ROC of the voting ensemble. This ROC curve proves the better performance of the AutoML pipeline.

![AutoML ROC (click to see the image)](img/AML-AutoML-ROC.png?raw=true)

Figure 6: AutoML ROC

## Pipeline comparison
Both the HyperDrive and AutoML pipelines have their pros and cons. The HyperDrive model offers greater control over the model selection and the hyperparameter space definition. The AutoML pipeline is easier to set up (see Figure 7) and offers greater range of models to try. Direct comparison of the pipelines is difficult (see Figure 1), as they were optimized for different metrics.

![Pipeline Architecture](http://www.plantuml.com/plantuml/png/NP2_pjDG3CLtFuN7xri0CJ0aLLMWI0mW8M1WgzpKIKtkdojVtqgH-l3SfBHgkzg_StnyDjb8hNW7pUSO0dU65l4JXH5_isDFEh99671BHl3-7NRH5H_o2ffjtHlZX-i8NgeoAPyu61ugZevff86HmW5BaD17zMHoOfIGYkNN5feVoecK5m6iP6rA4juCRBi_k-dbTVLMrYrKuGZDrocnRdNM9qdi9F2vnZx6c1a9EqSvIo-irVPNSc9enuln_DDYl4GnpeQPhOKyKpFCdVGJr23asLWXaq-EzJ-1D5JjZKCcTHDaYodLQY159ztqnDOsUncACYo7P-vltBV0DS02tY5uzvcMCCSlUo_sF5zwk1xuM2n-__RGnCtlidlmUfZD8IwTVZcHArRjEHTdUGtDJnua6-jIdey7TcVOL47nxXy0)

Figure 7: Pipeline architecture

## Future work
These are the guidelines and improvements for future work:
+ Check assumptions and business requirements, e.g. will the model be used before a campaign or also during a campaign? Availability of data from the current campaign can improve the predictions!
+ Use the AutoML to identify best performing model first, use HyperDrive to optimize hyperparameters second.
+ After using random hyperparameter sampling in the initial search, refine the search space with the grid or Bayesian sampling to improve results.
+ When comparing models, optimize for the same metric.
+ Consider using the Neural Network (deep learning), as it proved the best performance in the original research paper (Moro et all, 2014).

## Wrap up
The compute cluster deleted using SDK (see [udacity-project.ipynb](https://github.com/lustraka/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/udacity-project.ipynb)). The compute instance deleted using Azure ML Studio GUI. The VM disconnected.

## References
+ [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
+ [\[Moro et al., 2014\]](https://core.ac.uk/download/pdf/55631291.pdf) S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014 (preprint dated 2014-02-19)
+ [Train scikit-learn models at scale with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-scikit-learn?view=azure-ml-py)



<!--
![(click to see the image)](img/?raw=true)

@startuml
:Connect to Bank Telemarketing Data
20 columns<
:Delete 6 columns
(to prevent data leakage and uninformative inputs);
:Prepare data
<i>train.get_X_y()</i>;
fork
:HyperDrive pipeline|
:Choose a classifier;
:Split the data to train and test set;
split
:Specify
parameter
sampler;
split again
:Specify early
stopping
policy;
split again
:Configure
training
job;
end split
:Configure HyperDrive run;
:Submit HyperDrive run;
fork again
:AutoML pipeline|
:Configure AutoML run;
:Submit AutoML run;
end fork
:Evalute results>
:Register the model|
@enduml
-->
