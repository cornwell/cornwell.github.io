---
layout: distill
title: Analyzing CDC Diabetes Health Indicators Data (and SVMs)
description: a post about this publicly-available data, and using kernel-based SVMs
tags: svm data-quality courses coding-challenge
date: 2025-05-30
featured: false
thumbnail: assets/img/12.jpg
pretty_table: true
tikzjax: true

authors:
  - name: Christopher Cornwell
    url: "https://cornwell.github.io"
    affiliations:
      name: Towson University

toc:
  - name: the data
    subsections:
      - name: Previous work
      - name: Reconsidering the data
  - name: gaussian kernel SVMs
  - name: results on CDC data
  - name: takeaway

---

## the data

As part of their class project, my students looked at the CDC Diabetes Health Indicators survey data &ndash; a data set taken from the 2015 BRFSS Survey, made [available on the CDC website](https://www.cdc.gov/brfss/annual_data/annual_2014.html). The data set contains 253,680 instances (each the survey responses of an individual), which is a _clean_ subset of the original 441,455 responses to the CDC survey.  The original survey data has 330 features; however, the data that the students used has just 21 indicators (features) tracked. Also, there is a target variable with three classes: diabetic, pre-diabetic, and no diabetes. My students used a data set that was extracted from this one: first, the diabetic and pre-diabetic classes were put into one class, making it a binary classification problem; second, a subset (of 70,692 instances) was taken so that the two classes are balanced, having 50% of the instances have no diabetes for target value. The extracted data set (and a notebook with how preprocessing was done) is available through the [UCI Machine Learning repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) and on [kaggle.com](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), from the contributor Alex Teboul.

As mentioned, many of the features are responses to a survey; however, some are calculated variables based on participant responses. The data includes answers from respondents in the United States (from all 50 states and Washington, DC), as well as from Guam and Puerto Rico.

**Features.** Two thirds of the features in the data are binary variables that correspond to a response of _yes_ or _no_ to a survey question. These binary variables are listed in the table below. 

| feature       | description                                                                     |
| :-----------: | :------------                                                                   |
| `HighBP`      |  _ever been told [by health professional] that you have high blood pressure?_   |
| `HighChol`    |  _ever been told [by health professional] that your blood cholesterol is high?_ |
| `CholCheck`   |  _have you had a cholesterol check within past 5 years?_                        |
| `Smoker`     |  _...smoked at least 100 cigarettes in your entire life?_                       |
| `Stroke`      |  _have you ever had a stroke?_                                                  |
| `HeartDiseaseorAttack` | _ever told you have coronary heart disease or had a heart attack?_  |
| `PhysActivity`|  _...done physical activity or exercise (other than your job) in past 30 days?_ |
| `Fruits`       |  _do you consume fruit 1 or more times per day?_                                |
| `Veggies`     |  _do you consume vegetables 1 or more times per day?_                           |
| `HvyAlcoholConsump` |  _more than 14 drinks per week (men)? more than 7 drinks per week (women)?_|
| `AnyHealthcare`| _have any kind of health care coverage?_                                       |
| `NoDocbcCost` | _in past 12 months, have you needed to see doctor but could not because of cost?_|
| `DiffWalk`    |  _do you have serious difficulty walking or climbing stairs?_                   |
| `Sex`         |  the sex of the respondent                                                       |

<p></p>

Other features are integers, with various meanings. 

| `BMI` | `GenHlth` | `MentHlth` | `PhysHlth` | `Age` | `Education` | `Income` |

* `GenHlth` is a Likert-scale response in range 1-5. 
* Two of them &ndash; `MentHlth` and `PhysHlth` &ndash; are in range 1-30, corresponding to number of days within the past 30 days that mental, resp. physical, health were _not good_. 
* Respondents reported their age, education level, and income which were binned into categories, giving the features `Age`, `Education`, and `Income`. 
* The feature `BMI` is the body mass index, calculated from other responses.

### Previous work
On Kaggle, Teboul provided a link to a relevant study, [posted on the CDC website](https://www.cdc.gov/pcd/issues/2019/19_0109.htm), by Xie et al., which aimed to use predictive models for identifying type 2 diabetes. This study selected 26 of the original variables from the CDC survey. 

The target variable fetched by Teboul (for the 3-class data set) is the same as the linked study. However, most of the independent variables in the study are **not** those used by Teboul. A few variables align &ndash; e.g., `GenHlth`, `MentHlth`, `AnyHealthcare` (`HLTHPLN1` originally); some contain similar or reformatted information &ndash; e.g., the study uses a categorized version of `BMI`; but, many are completely distinct. (See Appendix A in the study.)

In the study by Xie et al., the most accurate model was a neural network model, with 82.41% accuracy. However, this model only had a recall of 0.3781. The researchers fit a decision tree model which had accuracy 74.26% and recall of 0.5161. They also used a support vector machine with Gaussian kernel, achieving accuracy of 81.78% and recall of 0.4014.

My students used a support vector machine with Gaussian kernel to fit a predictive model to the balanced data posted on Kaggle by Teboul. They obtained between 74% and 75% accuracy on test data, but did not report on the recall for the model.

### Reconsidering the data
Here, I'll work with the balanced data that my students used. Before jumping into training a model, let's think about properties of this data. Of the 21 independent variables, 14 are binary. Consider just these 14 columns in the data; that is, for the moment, remove the other columns so that we have just over 70,000 instances, each a vector of fourteen 0's and 1's. There are $$2^{14} \approx 16,000$$ possible vectors, so we are guaranteed that there are some distinct rows (instances) that are identical in these binary columns. 

I come from a background that is on the geometry side of mathematics, so I think of each of these binary vectors as representing a vertex on a hypercube $$[0,1]^{14} \subset \mathbb R^{14}$$. While some of the vertices will represent multiple instances in the data, how _mixed_ are the vertices in terms of the target label (on average)? In other words, are most instances at a vertex where the proportion of the other instances at the same vertex, that have the same label, is something close to 0.5? Or are most of the vertices "_mostly just one label_"?

First off, since 70,692 is a bit shy of $$4.5*16000$$, if the points were evenly spread about the $$2^{14}$$ vertices then each vertex would correspond to either 4 or 5 instances in the data (i.e., it would have close to $$6\times10^{-3}$$ percent of the data). 

Below, I find instances which correspond to the same vertex as row $$0$$ (see second Jupyter notebook cell). This is about $$0.6\ \%$$ of the data. I then compute the average of $y$-labels of these instances, getting about $$49.3\ \%$$. In other words, the labels of the data at that same vertex are nearly evenly split. Luckily, this is not the case for all vertices. In the later notebook cells, we see that for the instances that match row $$20$$ (which is over $$4.5\ \%$$ of the data), only $$11.6\ \%$$ of them have $y$-label equal to 1; and, for the instances that match row $$2725$$ (which is about $$0.51\ \%$$ of the data), over $$84\ \%$$ of them have $y$-label equal to 1.

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/vertex-label-mix.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/vertex-label-mix.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path prompt: false %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

What do we learn from this?  While these 14 binary variables are not "just noise," <d-footnote>They would be if most of the possible hypercube vertices either had around a 50% mix of labels or did not correspond to any instance.</d-footnote> the inability to separate some 0-labeled points from the 1-labeled points (that are at the same vertex) will present difficulty for classification &ndash; unless the remaining 7 variables provide a clear separation of such points. Considering those remaining variables, the `BMI` and `Age` variables seem to have the best chance of separating the data.

## gaussian kernel SVMs

## results on CDC data

## takeaway
