---
layout: post
title: Refreshed Coding Challenge Log
#description: a log to record my coding activities
tags: coding-challenge
date: 2025-06-25
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

From June 25 to Oct 02, 2025 (inclusive).

I'm pressing refresh on my coding challenge. It's not that I haven't coded since the end of the last log (June 2), but I needed a break on some days &ndash; I was writing with deadlines.

This time, I've some valuable ground rules. First, the following things count: writing comments in code, image generation for posts and manuscripts, and also, any testing-with-code of a mathematical idea. Second, don't be hard on yourself. If you don't have time to do more than 5-10 minutes worth, then do that. Test an idea, or just do some Project Euler or Leetcode challenge problem.

| date          | topic                           | activity            |
| :-----------: | :------------                   | :------------       |
| june 25       | svm classifiers                 | reviewed what I had done on the diabetes data and svm post; looked at variable correlations and tested increasing variable usage |
| june 26       | svm classifiers                 | by successively dropping a variable, found that only 5 variables were needed to get nearly all the accuracy (did not check that my method gave same 5 variables as the one suggested by a blogger's post); then did a small grid search for best params and got a recall of about 0.8 |
| june 27       | svm classifiers; blog post      | wrote some markdown for general discussion about gaussian svm's; also migrated and cleaned up my notebooks with comments |
| june 28       | summer undergrad research       | created images to support an argument for Malcolm's conics project (in Euclidean, focus & directrix imply cone intersection) |
| june 29       | grading                         | ran some of student's code in her project, compared |
| june 30       | using gen AI                    | checked (non-freemium) AI assistant's ability to do math &ndash; it did poorly, of course |
| july 1        | svm classifiers, blog post      | compared my 5 variables in svm observation to method used with decision trees on risk data, by blogger |
| july 2        | project euler task              | was on trip this day, just did one project euler question | 
| july 3        | project euler task              | did another project euler question |
| july 4        | vscode; svm classifiers         | reminded myself about environment control, noted RFECV (recursive feature elimination) wouldn't work with the svm model; organized notebook and wrote comments for future self |
| july 5        | project euler task              | solved palindrome project euler question |
| july 6        | svm classifiers                 | filled out background on gaussian svm; included images in post that are expandable |
| july 7        | project euler task; html-website| just the PE task seemed a bit too easy, even for a day with very little time; spent some additional time adjust the style elements for the code blocks on my site; it is a bit more pleasing |
| july 8        | svm classifiers                 | checking that the feature elimination gives 5 (the same 5) variables when using all the training data |
| july 9        | blog post, images               | filled in the section on feature elimination and started the results for the svm post; made and included images for the post |