---
layout: post
title: 100-day Coding Challenge Log
#description: a log to record my coding activities
tags: coding-challenge
date: 2025-05-29
featured: false
related_posts: false
#thumbnail: assets/img/12.jpg
pretty_table: true
tikzjax: true

authors:
  - name: Christopher Cornwell
    url: "https://cornwell.github.io"
    affiliations:
      name: Towson University

---

<style>
table th:first-of-type {
    width: 15%;
}
table th:nth-of-type(2) {
    width: 15%;
}
table th:nth-of-type(3) {
    width: 70%;
}
</style>

From May 27 to Sept 03, 2025.

| date          | topic                           | activity            |
| :-----------: | :------------                   | :------------       |
| may 27        | svm classifier, course projects | initial read-in of 2015 cdc survey data (from balanced kaggle dataset); checked results from students' fit to data using polynomial kernel svm, rbf kernel svm; decided rbf kernels were best type, and student models were underfitting; performed a search for best hyperparameters - student used data: $$\gamma\in [0.1,0.2,0.3,\ldots,1]$$, $$C\in [0.5, 0.75, 1,\ldots, 2.5]$$); after that, tuned precision - result: $$\gamma = 0.28, C = 0.87$$. Test acc.: $$\approx$$ 74%. |
| may 28        | svm classifier, course projects | considered properties of 2015 survey data - binary variable types; used unrestricted decision tree to check label-purity of vertices (hypercube vx's corresp. to binary variables); large proportion of training were support vectors (would increase in-sync with increasing amount of data used); almost all dual coefficients equal to -1.5 or 1.5; computed distance to marginal curves - want to understand more. |
| may 29        | svm classifier, course projects | included `Income`, `Education`, and `NoDocbcCost` columns (diff.from students); searched for best with this data: $$\gamma\in [0.25,0.3,0.35,\ldots,0.7]$$, $$C\in [1, 1.1, 1.2,1.3, 1.4, 1.5]$$; best result: $$\gamma=0.45, C=1.4$$, test accuracy = 74.9% and recall = 0.80; also tried wildly different values for $$\gamma, C$$ (other orders of magnitude), without improvement in accuracy. |
| may 30        | boosting, course projects | also attempted to fit data with an ensemble method &ndash; boosted decision trees (using AdaBoost); after a search for the value of hyperparameter `max_depth` that would not overfit, used decision tree with depth 5 as base estimator; test accuracy of 74.6% and recall of 0.786. |
| may 31        | blog post, course projects | made visuals: on toy data for discussing support vectors of kernel-based svms; confusion matrices. | 
| june 1        | blog post, course projects | began writing blog post on this post-course project exploration; figured some ways to adjust style of <code>layout: post</code>; looked up how to include Jupyter notebook output into distill-style blog. |
| june 2        | code for bird data analysis            | reacquainted myself with a class that I wrote for analyzing the shape data of bird bones, using a technique in tda; wrote in comments to clarify the functionality |