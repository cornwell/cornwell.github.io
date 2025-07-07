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

As part of their class project, my students looked at the CDC Diabetes Health Indicators survey data &ndash; a data set taken from the 2015 BRFSS Survey, made [available on the CDC website](https://www.cdc.gov/brfss/annual_data/annual_2014.html). The data set contains 253,680 instances (each being the survey responses of an individual), which is a _clean_ subset of the original 441,455 responses to the CDC survey.  The original survey data has 330 features; however, the data that the students used has just 21 features tracked. Also, there is a target variable with three classes: diabetic, pre-diabetic, and no diabetes. My students used a data set derived from this one, but with a change to the target: first, the diabetic and pre-diabetic classes were put into one class, making it a binary classification problem; second, a subset (of 70,692 instances) was taken so that the two classes are balanced, with 50% of the instances have 'no diabetes' for target value. The derived data set (and a notebook with how preprocessing was done) is available through the [UCI Machine Learning repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) and on [kaggle.com](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), from the contributor Alex Teboul.

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

* `GenHlth` is a Likert-scale response in range 1-5, the respondent's perception of their general health.
* Two of them &ndash; `MentHlth` and `PhysHlth` &ndash; are in range 1-30, corresponding to number of days within the past 30 days that mental, resp. physical, health were _not good_. 
* Respondents reported their age, education level, and income which were binned into categories, giving the features `Age`, `Education`, and `Income`. 
* The feature `BMI` is the body mass index, calculated from other responses.

### Previous work
On Kaggle, Teboul provided a link to a relevant study, [posted on the CDC website](https://www.cdc.gov/pcd/issues/2019/19_0109.htm), by Xie et al., which aimed to use predictive models for identifying type 2 diabetes. This study selected 26 of the original variables from the CDC survey. 

The target variable fetched by Teboul (for the 3-class data set) is the same as the linked study. However, most of the independent variables in the study are **not** those used by Teboul. A few variables align &ndash; e.g., `GenHlth`, `MentHlth`, `AnyHealthcare` (`HLTHPLN1` originally); some contain similar or reformatted information &ndash; e.g., the study uses a categorized version of `BMI`; but, many are completely distinct. (See Appendix A in the study.)

For some circumstances, it is of interest to consider other metrics besides just the overall classification accuracy. For example, for a model that predicts whether an individual may have diabetes or pre-diabetes, it is reasonable to have a priority for a high true positive rate (here, the proportion of individuals that do have diabetes or pre-diabetes who were identified as such by the model). This measure is called the _recall_.

In the study by Xie et al., the most accurate model was a neural network model, with 82.41% accuracy. However, this model only had a recall of 0.3781. The researchers fit a decision tree model which had accuracy 74.26% and recall of 0.5161. They also used a support vector machine with Gaussian kernel, achieving accuracy of 81.78% and recall of 0.4014.<d-footnote>Note, these metrics are for classification with three labels (keeping diabetes and pre-diabetes separate). I presume that the recall is the true positive rate where "positive" refers only to diabetes.</d-footnote>

My students used support vector machines, one with polynomial kernel and another with Gaussian kernel to fit predictive models to the balanced data posted on Kaggle by Teboul. Both models had between 74% and 75% accuracy on test data<d-footnote>Recall, the variables in their data are different than those in the study by Xie et al.</d-footnote>; the students did not report on the recall for the model.

### Reconsidering the data
Here, I'll work with the balanced data that my students used. Before jumping into training a model, let's think about properties of this data. Of the 21 independent variables, 14 are binary. Consider just these 14 columns in the data; that is, for the moment, remove the other columns so that we have just over 70,000 instances, each a _binary vector_ of fourteen 0's and 1's. There are $$2^{14} \approx 16,000$$ possible vectors, so we are guaranteed that there are some distinct instances (rows) that have identical binary vectors. 

I come from a background that is on the geometry side of mathematics, so I think of each of these binary vectors as representing a vertex on a hypercube $$[0,1]^{14} \subset \mathbb R^{14}$$. While some of the vertices will represent multiple instances in the data, how _mixed_ are the vertices in terms of the target label (on average)? In other words, at a given vertex, do most instances have the same label, or is it closer to 50/50? The answer will certainly vary from vertex to vertex, but what happens on average?

First off, since there are about $$16000$$ vertices here, if the points were evenly spread amongst them then each vertex would correspond to $$1/16000$$ of the data (which is about $$0.006\ \%$$ of the data, or 4 to 5 instances at each vertex). The data is not that spread out. When selecting a row at random and considering its corresponding vertex, you often get between $$0.5\ \%$$ and $$2\ \%$$ of the data at that vertex. (There are likely some vertices with no corresponding instances in the data, but I have not checked.) 

When we find data points (instances) corresponding to the same vertex as row 0 (see second Jupyter notebook cell below), we get about $$0.6\ \%$$ of the data. The average of $$y$$-labels among these instances is $$0.493$$. In other words, the labels of the data at the same vertex as row 0 are nearly evenly split. Luckily, this is not the case for all vertices. In the later notebook cells below, we see that for the instances that match row 20 (which is over $$4.5\ \%$$ of the data), only $$11.6\ \%$$ of them have $$y$$-label equal to 1.

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/vertex-label-mix.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/vertex-label-mix.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path prompt: false %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

For an overall summary of how mixed the vertices are in terms of labels, we can use a decision tree model to help. Fit a decision tree to all of the points in the data, and put no restriction on the maximum depth of the tree. After importing `DecisionTreeClassifier` from `sklearn.tree`, and having assigned the Pandas DataFrame `Xbinary` in the above notebook cells, this can be done as follows.

```python
tree = DecisionTreeClassifier()
tree.fit(Xbinary, y)
tree.score(Xbinary, y)
```

Doing so gives a tree with depth $$14$$. As there are only two possible values of each coordinate feature (and there are $$14$$ features), this means that the tree has a splitting along every coordinate.  Hence, each of its leaves either contains exactly one vertex of the hypercube, or all vertices in that leaf are represented by instances with only one label. The fitted model will label an $$\textbf{x}$$ that is in the data set, which is at some vertex, by assigning it the majority label of the instances at that vertex. 

At one vertex, the percent of the data that will be correctly labeled is precisely the percentage that are in the majority, something between $$50\ \%$$ and $$100\ \%$$. The overall accuracy score of the model, on all of the data, is the weighted average of these percentages. This score on our data (the output from the last command above) is $$0.728$$. Hence, in these 14 variables, the data is about halfway between being pure noise and pure signal. (A score of $$0.5$$ would say that knowing these 14 features gives no information &ndash; you may as well flip a coin. A score of $$1.0$$ would say that knowing these 14 features informs you perfectly on what the target label should be.)

What do we learn from this?  While these 14 binary variables are not "just noise," the inability to separate some 0-labeled points from the 1-labeled points (that are at the same vertex) will present difficulty for classification &ndash; unless the remaining 7 variables provide a clear separation of such points. Considering those remaining variables, the `BMI` and `Age` variables seem to have the best chance of separating the data.

## gaussian kernel SVMs

A standard support vector machine (SVM) attempts to separate data $$X$$ in $$\mathbb R^n$$ in a linear way.  It finds the _hyperplane_ in $$\mathbb R^n$$ which strikes the best balance between having a large margin (the "gap" between it and the data) and good classification accuracy &ndash; having the 1-labeled data in the positive half-space and the 0-labeled data in the negative half-space, as much as possible.  Since data is often not linearly separable, SVMs are often used in combination with a _kernel_.  That is, there is a transformation of the coordinates of the data, as would be achieved by a mapping $$\Phi:\mathbb R^n\to\mathbb H$$ into some space<d-footnote>The space $$\mathbb H$$ can be guaranteed to be some <it>Hilbert space</it>, allowing for certain mathematical operations.</d-footnote> $\mathbb H$ and then an SVM is used on the image $$\Phi(X)\subset \mathbb H$$ to perform classification, separating the data by a hyperplane in $$\mathbb H$$.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Hfit-simulated-marginal-hyperplanes.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
  A set of data points in $\mathbb R^2$, with labels shown by color, that are separable by a hyperplane (which in $\mathbb R^2$ is a line).  The <it>margin</it> is the distance, from the (center) separating hyperplane, to the marginal hyperplanes.
</div>

A rather clever aspect of using SVMs with a kernel is that you can achieve this without actually determining the mapping $\Phi$.  Instead, you choose a _kernel function_ $$K(x,x')$$, whose output will be the inner product of $$\Phi(x)$$ and $$\Phi(x')$$ in $$\mathbb H$$, for a pair of vectors $$x, x'$$ in $$\mathbb R^n$$.  One common choice for the kernel function is a Gaussian applied to the distance between $$x$$ and $$x'$$, i.e., $$\lVert x-x'\rVert$$.  In other words, a constant $$\gamma > 0$$ is chosen and you define  
<div style="text-align:center;">$K(x, x') = e^{-\gamma\lVert x-x'\rVert^2}.$</div>

With this definition, if $$x$$ and $$x'$$ are far apart then $$K(x,x')$$ will be close to zero,<d-footnote>How quickly this happens depends on how large $\gamma$ is.</d-footnote> and it will be close to one when $$x$$ and $$x'$$ are very close.  

After fitting such an SVM to labeled data $$X = \{(x_i, y_i)\}_{i=1}^m$$, a set of _support vectors_, $$x_{i_1}, x_{i_2}, \ldots, x_{i_k}$$ from the data are determined, as well as coefficients $$\alpha_1,\alpha_2,\ldots,\alpha_k$$.  These are the only $$k$$ points that are needed to determine the predicted label on a new point; for $$x\in\mathbb R^n$$, the SVM gives $$x$$ a positive label when $$\Sigma_{j=1}^k\alpha_j K(x, x_{i_j}) > 0$$.  Roughly speaking, for a support vector $$x_{i_j}$$ that has label 1, one should expect $$\alpha_j$$ to be positive; if, however, $$x_{i_j}$$ has label 0, one should expect $$\alpha_j$$ to be negative.  Hence, thinking about the discussion of the previous paragraph, there is a nice interpretation: the support vectors that are closest to $$x$$ will have greatest influence on the predicted label of $$x$$, such that the predicted label is more likely to match their own (and the closer they are relative to the other support vectors, the greater the influence is).

If training data mostly has a region filled with data of one label, then some of the points near the "margin" of that region will be support vectors, particularly those that are opposite another region which has points of the other label. 

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/support-vecs-Gaussian-SVM.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

Because of what we observed about a fair number of the features being binary and being somewhat classified just in those coordinates, a Gaussian kernel SVM seems like a natural choice for ML model on this data.  We'll see that it effectively allows us to use only 5 of the features in the data to do our best possible on accuracy and recall.

## results on CDC data

## takeaway
