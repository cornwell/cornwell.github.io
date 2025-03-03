---
layout: post
title: Pre-decisive sets for ReLU networks
date: 2025-03-02 14:59:00
description: on minimal sets that are subsets of decisive sets for ReLU networks
tags: research thoughts ReLU_networks
categories: math neural_networks
related_posts: false
thumbnail: assets/img/2.jpg
toc:
  beginning: true
---

In a [post last December](https://cornwell.github.io/blog/2024/precomposing-funcdim/), I wrote some of my thoughts about Section 8 in the [paper by Grigsby, Lindsey, Meyerhoff, and Wu](https://arxiv.org/abs/2209.04036), and the effect of precomposing with a layer. Something that I continued to think about, with collaborator Na Zhang, was how to leverage what that Section says about $$\dim_{fun+}(\theta)$$ versus $$\dim_{fun}(\theta)$$ for this purpose. 

It lead to some basic observations, but we think they'll be helpful. 

--- 
## Notation 
We are considering feed-forward neural networks with ReLU activation function. Suppose that the input space is $$\mathbb R^{n_0}$$ and that the parameter space is $$\Omega$$ (which may be identified with $$\mathbb R^D$$ as a set, $$D$$ being the total number of parameters &ndash; weights and biases). We fix a parameter $$\theta\in\Omega$$, and so determine a network $$\mathcal N(\theta)$$, and we assume $$\theta$$ is "nice" in the sense of the source paper (it is _generic_, _transversal_, and _combinatorially stable_). The network has an associated network function $$F_\theta:\mathbb R^{n_0}\to\mathbb R$$.[^1]

Associated to $$\mathcal N(\theta)$$ is a _canonical polyhedral complex_ $$\mathcal C(\theta)$$ that decomposes $$\mathbb R^{n_0}$$. A **top cell** of $$\mathcal C(\theta)$$ refers to an $$n_0$$-dimensional cell of that polyhedral complex. Note that the restriction of $$F$$ to any top cell consists of an affine linear function on that top cell (cf. Lemma 2.8 of [GLMW]). 

Consider a finite subset $$Z = \{z_1,\ldots,z_m\}\subset\mathbb R^{n_0}$$.  For a parameter $$v$$, the evaluation map $$E_Z$$ is defined by $$E_Z(v) = (F_v(z_1),\ldots,F_v(z_m))$$. Supposing that points in $$Z$$ are parameterically smooth for $$\theta$$, there is a neighborhood $$V\subset\Omega$$ of $$\theta$$ on which $$E_{Z}:V \to \mathbb R^{m}$$ is defined and differentiable. I will pay particular attention to the singleton case, $$Z = \{z\}$$, where $$z$$ is a point in the interior of some top cell for $$\theta$$. In this case, we write the Jacobian (gradient) as $$JE_z\vert_{\theta}$$ (or, simply $$JE_z$$). 

Let $$R$$ be a top cell of the canonical polyhedral complex for $$\mathcal N(\theta)$$ and suppose that $$\{x_0,x_1,\ldots,x_{n_0}\}$$ is affinely independent and contained in the interior of $$R$$. I observed in my last post, using some basic linear algebra, that if $$x$$ is any point in the interior of $$R$$ then we have the equation 

\begin{equation}
\label{eqn:JE-lincombo}
JE_x - JE_{x_0} = \sum_{i=1}^{n_0} c_i(JE_{x_i} - JE_{x_0}),
\end{equation}
where the vector $$x-x_0$$ satsifies $$x-x_0 = \sum_{i=1}^{n_0} c_i(x_i - x_0)$$. In short, the reason for this is that, while $$JE_z$$ consists of components that are polynomials in the parameters of the network, if we compute the partial derivatives and then consider it as a function of $$z$$ (in $$R$$), rather than parameters, then it is affine linear.

Note that (\ref{eqn:JE-lincombo}) fails if $$x$$ is contained in another top cell than $$R$$. However, if we replace $$F$$ by the affine linear function (extended over $$\mathbb R^{n_0}$$) that agrees with $$F\vert_{R}$$, and discuss the Jacobian of the evaluation map of that function, then the expression would be valid for all $$x\in\mathbb R^{n_0}$$. For that Jacobian, of the evaluation map corresponding to $$F\vert_{R}$$, at $$x$$, we use the notation $$JE_x^{R}$$ and we call this the **Jacobian at** $$R$$.

---
## Relations on Jacobians at adjacent top cells

Consider two hyperplanes $$H_1$$ and $$H_2$$ in $$\mathbb R^{n_0}$$ which correspond to two of the neurons from layer 1 of $$\mathcal N(\theta)$$. As $$\theta$$ is generic, $$H_1\cap H_2$$ is an $$(n_0-2)$$-dimensional affine subspace. Choose a point $$q$$ in the intersection that is not contained in any other (bent) hyperplane from another layer (i.e., $$q$$ is in the relative interior of an $$(n_0-2)$$-dimensional face of $$\mathcal C(\theta)$$ contained in both $$H_1$$ and $$H_2$$). Suppose that we have top cells $$R, S, A,$$ and $$B$$ which all contain $$q$$ in one of their $$(n_0-2)$$-dimensional faces; every neighborhood of $$q$$ intersects in a non-empty set with each of $$R, S, A,$$ and $$B$$, and this is true of no other top cell. Furthermore, choose our naming so that: 
* $$R$$ and $$S$$ do not share a facet;
* $$A$$ and $$B$$ do not share a facet.

We'll check a certain relation on Jacobians at these cells. Let $$x$$ be a point, and say that $$z^R_0,z^S_0,z^A_0,$$ and $$z^B_0$$ are points in $$R, S, A,$$ and $$B$$, respectively.

Since each component function of $$JE^R$$ is affine linear, there is some matrix $$W^R$$ so that $$JE^R_x - JE^R_{z^R_0} = W^R(x - z^R_0)$$.

Choose one component (entry) of $$JE_x^R$$, denote it by $$p^R_x$$, and denote the corresponding entry of $$JE_x^A$$ by $$p^A_x$$. WLOG, we may assume that $$R$$ and $$A$$ are on the same side of the hyperplane $$H_2$$, meaning that they share a facet that is contained in $$H_1$$. Denote the weights and bias corresponding to $$H_1$$ by $$w^1_1, w^1_2, \ldots, w^1_{n_0}, b^1$$. Also, for the set of all weight/bias (variables) for the network except those $$n_0+1$$ variables for $$H_1$$, use notation $$\Theta$$. 

We may write $$p^R_x$$ and $$p^A_x$$ as polynomials in $$(\mathbb R[\Theta])[w^1_1,\ldots,w^1_{n_0},b^1]$$. These polynomials have degree at most 1 (in these variables). 

Furthermore, since the ternary labelings of $$R$$ and $$A$$ agree for all neurons except $$H_1$$, the only difference between $$JE^R_x - JE^R_{z^R_0}$$ and $$JE^A_x - JE^A_{z^R_0}$$ is that one of them has degree 1 terms in $$(\mathbb R[\Theta])[w^1_1,\ldots,w^1_{n_0},b^1]$$ and those are missing in the other. In other words, $$JE^R_x - JE^A_x$$ entirely consists of such degree 1 terms, each of which contains a factor of a component in $$(x - z^R_0)$$.

One can do the same thing with $$JE^B_x-JE^S_x$$, and must subtract in this order to get the signs on these degree 1 terms to match. There are no additional (or missing) non-zero terms in this, since, even though the ternary label for $$H_2$$ has changed in these cells, we have a lemma.

##### Lemma. 
{: .env-title }

For $$X \in\{R,S,A,B\}$$, if a term in $$p^X_x$$ is degree 1 in one of the weights and biases from $$H_2$$, then it is degree 0 in $$\{w^1_1,\ldots,w^1_{n_0},b^1\}$$. 

> _Proof_. &nbsp;&nbsp; The statement holds because neurons for $$H_1$$ and $$H_2$$ are in the same layer. Thus, from the compositional structure of the feed-forward neural network, their weights/biases are never jointly in the same monomial of $$E_x$$ (as a polynomial in the parameter variables). $$\blacksquare$$

We have, therefore, that $$JE^R_x - JE^A_x = JE^B_x - JE^S_x$$. By rearranging, 
\begin{equation}
JE^R_x + JE^S_x = JE^A_x + JE^B_x 
\end{equation}

[^1]: Often, I will simply write the network function as $$F:\mathbb R^{n_0}\to\mathbb R$$, leaving the parameter of the network as understood.) 