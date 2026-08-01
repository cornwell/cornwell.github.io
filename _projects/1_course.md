---
layout: page
title: math 265
description: an introductory course in linear algebra 🎉
img: assets/img/13.jpg
importance: 4
category: undergrad
---

A major part of this course is concerned with procedures and computations dealing with vectors and matrices, and it is important to gain skill in these computations. Assigned work and quizzes will reflect this.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Class_snippings/RReduce.jpg" title="row reducing a matrix" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/Class_snippings/OrthoProjection.jpg" title="computing a projection" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    There will be a lot of algebraic work, manipulating matrices (to the left); computing something related to a set of vectors (to the right).
</div>

However, there is a famous quote from Richard Hamming.
<blockquote>The purpose of computation is insight, not numbers. (Hamming, 1962)</blockquote>

Yes, what he said &ndash; it's about _insight_. And on that note, there are very few topics that can match linear algebra in terms of the number (and variety) of insights that it has produced. Insights from linear algebra are at the core of signal processing and related techniques, letting us do things such as communicate remotely (so, the ability of your phone to send and receive information), figure out what distant things in space are made of, and monitor what is happening in the body with, for example, a CT-scan or an MRI. In computing, various ideas from linear algebra have been key to developing techniques for: algorithmic speed up for computation times, effective internet searching (Google's PageRank), and the development of 3D graphics systems, to name a few. Additionally, there would be no field of AI (or, more generally, machine learning or deep learning) without linear algebra. It is fundamental.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Class_snippings/spectra.jpg" title="detecting elements from spectroscopy" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/RotationVideo.mp4" title="3D graphics - rotating" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Class_snippings/mri.jpg" title="mri of brain" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>

In our class, the purpose is to learn how the mathematical methods work. With that goal, we won't have time to spend large amounts of time on how these applications work. However, I will emphasize a way to think about linear algebra that focuses on concepts, particularly those that, sometimes in surprising ways, led to so many insights and innovation.
