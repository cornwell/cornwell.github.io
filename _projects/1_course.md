---
layout: page
title: math 265
description: an introductory course in linear algebra 💚
img: assets/img/24a.jpg
importance: 4
category: undergrad
---

A major part of this course is concerned with procedures and computations that deal with vectors and matrices, and it is important to gain skill in such computations. Assigned work and quizzes will reflect this.

<div class="row justify-content-sm-center">
    <div class="col-sm-3 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Class_snippings/RReduce.jpg" title="row reducing a matrix" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-3 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Class_snippings/OrthoProjection.jpg" title="computing a projection" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Manipulating matrices (on the left). Computing a projection based on a set of vectors (on the right).
</div>

Even so, there is a famous quote from Richard Hamming.
<blockquote>The purpose of computation is insight, not numbers. (Hamming, 1962)</blockquote>

Exactly. What he said &ndash; it is about gaining _insights_. On that note, there are very few topics that can match linear algebra in terms of the number (and variety) of new technologies that have been developed by using insights from linear algebra. In fact, it is so essential to how we study mathematics itself that anytime math gets involved in some application, linear algebra is likely involved.

**Some examples.** Linear algebra ideas are at the core of signal processing, letting us do such things as communicate remotely (so, the ability of your phone to send and receive information), figure out what distant stars are made of, and monitor what is happening in the body with, for example, a CT-scan or an MRI. In computing, ideas from linear algebra have been key to developing improved computation times, file compression, effective internet searching (Google's PageRank), and realistic 3D graphics systems, to name a few. As an additional note, the field of AI (or, more generally, machine learning and deep learning) would be _nonexistent_ without linear algebra.


<div class="row">
    <div class="col-sm mt-4 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Class_snippings/spectra.jpg" title="detecting elements from spectroscopy" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-8 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Class_snippings/RotationVideo.gif" title="3D graphics - rotating perspective" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The spectra (which represent coefficients in some basis) for elements in the periodic table (left). Making a rotating perspective look right in 3D graphics (right).
</div>

In class, our purpose is to learn how the computations and mathematical methods work. With that goal, we won't be able to spend large amounts of time on striking applications of linear algebra. However, we will include in our discussions conceptual ideas and insights which, when mixed with the computational skills, have given rise to so much innovation.
