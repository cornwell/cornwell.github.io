---
layout: post
title: Testing how posts work
date: 2024-12-18 19:36:00
description: testing out functionality of blog posts
tags: testing formatting
categories: sample-posts
related_posts: false
thumbnail: assets/img/9.jpg
tikzjax: true
---

First, this theme uses MathJax 3. So we can write in math mode with `$$ $$` as expected, for example: $$e^{i\theta} = \cos(\theta) + i\sin(\theta)$$, but not using single dollar-signs `$`.

I am wondering if the MathJax used in Wordpress will cross-over. It would be nice if I could transfer some posts and not rewrite them. This uses `\( \)`. Tested `\(e^{i\theta} = \cos(\theta) + i\sin(\theta)\)`, but it does not work.

Apparently you can also use tikz in this theme, as long as you set the `tikzjax` property to true:

<div align="center">
    <script type="text/tikz">
    \begin{tikzpicture}[>=stealth]
        \draw[red,fill=black!60!red] (0,0) circle [radius=1.5];
        \draw[green,fill=black!60!green] (0,0) circle [x radius=1.5cm, y radius=10mm];
        \draw[blue,fill=black!60!blue] (0,0) circle [x radius=1cm, y radius=5mm, rotate=30];
        \draw[->, thick] (2.5,1.8) --node[at start, right]{$f(\alpha)$} (1.2,1);
        % a commented line? next: a Bezier curve
        \draw[->, thick] (3,0) ..controls (2,0) and (2.2,-1.75) ..(1.2, -1);
        \draw (3,0) node[right]{$\dag$};
    \end{tikzpicture}
    </script>
</div>

---

You can also embed lines of code. Like this?

```python
def get_primes_lessthan(n):
    primes = []
    for i in range(2, n):
        is_composite = test_relcomposite(i, primes)
        if is_composite:
            pass
        else:
            primes.append(i)
    return primes
def test_relcomposite(num, primelist):
    # catch the exceptional cases
    try:
        assert num > 1
    except AssertionError:
        return True
    # now test for divisors in primelist (assume doesn't contain 1)
    found_divisor = False
    if len(primelist) == 0:
        found_divsor = not (num > 1)
    else:
        for p in primelist:
            if num%p == 0:
                found_divisor = True
                break
            else:
                continue
    return found_divisor
```
That works. I want to tweak the bg color though, at least in dark mode.

<mark>Reader beware! The default tab behavior for this theme is 4 spaces; some want only 2.</mark>
