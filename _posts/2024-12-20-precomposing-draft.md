---
layout: post
title: Questions on functional dimension of ReLU networks
date: 2024-12-20 10:39:00
description: understanding functional dimension by precomposing
tags: research thoughts ReLU_networks
categories: math neural_networks
related_posts: false
thumbnail: assets/img/9.jpg
toc:
  beginning: true
---

I first starting writing this post as a draft at my old Wordpress website. I'm attempting to port it here. Please be patient if I missed a spot where some syntax or formatting needed to be translated.

---

This is one of a series of posts about ReLU neural networks. My focus in this post is on the **functional dimension** of (a parameter for) a ReLU neural network. <br>
&nbsp;&nbsp;&nbsp;&nbsp; See _Functional dimension of feedforward ReLU neural networks_ by Grigsby, Lindsey, Meyerhoff, and Wu ([link to the preprint](https://arxiv.org/abs/2209.04036)). 

I will abbreviate references to this paper by writing \[GLMW22\].  In particular, I want to work on understanding Lemmas 8.2 and 8.5, as well as Theorem 8.7 in that paper – having to do with how the functional dimension behaves under composition of networks.

## Setup and notation
---

The notation used will largely match that of \[GLMW22\]. We'll be working with feedforward ReLU neural networks and their associated network function. Here is the general setup.

**Definition 1.** Let $$n_0\in \mathbb N$$ and $$d \in \mathbb N$$ and let $$\phi:\mathbb R \to \mathbb R$$ be a continuous function. A **feedforward neural network** $$\mathcal N$$ defined on $$\mathbb R^{n_0}$$, with _depth_ $$d$$ and _activation function_ $$\phi$$, is an ordered sequence of $$d$$ affine maps $$A^1, A^2, \ldots, A^d$$ defined with $$A^{\ell}:\mathbb R^{n_{\ell-1}} \to \mathbb R^{n_{\ell}}$$ for every $$1 \le \ell \le d$$. For any $$n\in \mathbb N$$, define $$\Phi:\mathbb R^n\to\mathbb R^n$$ as the component-wise application of the function $$\phi$$ on the input. The **network function** associated to $$\mathcal N$$ is a function $$F:\mathbb R^{n_0}\to \mathbb R^{n_d}$$ that is defined by

$$ F(\mathbf{x}) = A^d\circ \Phi\circ A^{d-1} \circ \ldots \circ \Phi\circ A^{2} \circ \Phi\circ A^{1}(\mathbf{x}) $$

for every $$\mathbf{x} \in \mathbb R^{n_0}$$. The list $$(n_0, n_1, \ldots, n_d)$$ is called the **architecture** of the neural network $$\mathcal N$$.

Here the focus is exclusively on _ReLU (feedforward) neural networks_ which are feedforward neural networks which have an activation function given by $$\phi(x) = \max\{0, x\}$$, for $$x\in \mathbb R$$.

For every $$1\le \ell\le d-1$$, the **$$\ell$$-th layer map** $$F^{\ell}:\mathbb R^{n_{\ell-1}} \to \mathbb R^{n_{\ell}}$$ is defined by $$F^{\ell} = \Phi\circ A^{\ell}$$, with the $$d$$-th layer map being simply $$A^d$$.  For every $$1\le \ell\le d$$ and every $$1\le j\le n_{\ell}$$, the pair $$(\ell, j)$$ is referred to as a **neuron** of $$\mathcal N$$; sometimes this **neuron** is refers to the function $$z^{\ell}_j : \mathbb R^{n_0} \to \mathbb R$$ that takes on the ("pre-activation") values at $$ (\ell, j) $$, i.e., letting $$\pi_j$$ denote projection to the $$j$$-th coordinate, we have that $$z^{\ell}_j = \pi_j \circ A^{\ell}\circ F^{\ell-1} \circ\ldots\circ F^1$$.

Note that each affine map $$A^{\ell}$$ may be expressed in coordinates, so that for $$\mathbf{x}\in\mathbb R^{n_{\ell-1}}$$, we have $$A^{\ell}(\mathbf{x}) = W^{\ell}\mathbf{x} + \mathbf{b}^{\ell}$$ where $$W^{\ell}$$ is a $$n_{\ell} \times n_{\ell-1}$$ "weight" matrix and $$\mathbf{b}^{\ell} \in \mathbb R^{n_\ell}$$ is a "bias" vector.

Thinking of ReLU neural networks with a given architecture $$(n_0, n_1, \ldots, n_d)$$ – or, more appropriately, the associated network functions for such networks – as a parameterized class of functions, the entries in the weight matrices and bias vectors for the various layers make up the parameters. We organize the parameters as $$\theta = (W^1, \mathbf{b}^1, \ldots, W^d, \mathbf{b}^d)$$, which we associate in some chosen manner to a point in $$\mathbb R^D$$, with $$D = D(n_0,\ldots,n_d) := \sum_{i=1}^d n_i(n_{i-1}+1).$$ We write $$\Omega := \mathbb R^D$$ in order to emphasize the space of parameters – and how it corresponds to weights and biases in $$\mathcal N$$. At times, in order to denote the neural network with parameters $$\theta$$, for an understood architecture, we write $$\mathcal N(\theta)$$. Additionally, the associated network function may be written as $$F_\theta: \mathbb R^{n_0} \to \mathbb R^{n_d}$$.

## Functional Dimension of a Parameter for a ReLU Neural Network
---

The Lemmas and Theorem that we want to better understand a related to a quantity called $$\dim_{fun}(\theta)$$, the functional dimension of $$\theta$$ (for a ReLU network $$\mathcal N(\theta)$$). We'll spend a bit of time to define this quantity. For the definition, there are a few technical assumptions that must be made about $$\theta$$. These assumptions are not very restrictive. To the contrary, they rule out some atypical cases for how $$\theta$$ affects the network. To arrive at the definition quickly, for now we will bypass describing the technical assumptions and simply say that $$\theta$$ is "nice" (or, satisfies "nice conditions") and move forward in defining $$\dim_{fun}(\theta)$$.

Consider a fixed architecture $$(n_0, n_1, \ldots, n_d)$$ and the class of ReLU networks with this architecture. Given some $$\theta\in\Omega$$, for $$1\le j\le n_d$$, use $$F_{\theta,j}:\mathbb R^{n_0}\to\mathbb R$$ for the $$j$$-th coordinate function of $$F_\theta$$, so for every $$\mathbf{x}\in\mathbb R^{n_0}$$ we have $$F_\theta(\mathbf{x}) = (F_{\theta,1}(\mathbf{x}), \ldots, F_{\theta,n_d}(\mathbf{x}))$$.[^1] Finally, consider a finite ordered set of points $$Z = \{z_1, z_2, \ldots, z_k\} \subset \mathbb R^{n_0}$$.  The evaluation map at $$Z$$, $$E_Z: \Omega \to \mathbb R^{k\cdot n_d}$$ is given by setting, for every $$\theta\in\Omega$$,

$$E_Z(\theta) = (F_{\theta,1}(z_1), \ldots, F_{\theta,n_d}(z_1), \ldots, F_{\theta,1}(z_k), \ldots, F_{\theta,n_d}(z_k)).$$

To define the functional dimension, we use the Jacobian matrix of the evaluation map, which we denote by $$\mathbf{J}E_Z$$.  In other words, for each $$z_i\in Z$$, let $$\mathbf{J}E_{z_i}\lvert_{\theta}$$ be the $$n_m \times D$$ matrix, the $$j$$-th row of which is the vector of partial derivatives of $$F_{\theta, j}(z_i)$$ _with respect to the_ $$D$$ _parameters_,[^2]  evaluated at $$\theta$$. The matrix $$\mathbf{J}E_Z\lvert_{\theta}$$ is the $$k\cdot n_m \times D$$ matrix obtained by stacking these matrices.

**Definition 2.** Fix an architecture $$(n_0, n_1, \ldots, n_d)$$ and let $$\Omega$$ be its corresponding parameter space. Suppose that $$\theta\in\Omega$$ satisfies nice conditions. [^3] The **functional dimension** at $$\theta$$ is defined to be

$$\dim_{fun}(\theta) = \underset{Z \text{ is finite and smooth for } \theta}{\sup}\operatorname{rank} \mathbf{J}E_Z\lvert_{\theta}.$$

In Definition 2, the condition that $$Z$$ be smooth for $$\theta$$ (or, _parametrically smooth_ in \[GLMW22\]) refers to the function $$F:\Omega\times\mathbb R^{n_0} \to \mathbb R^{n_d}$$, given by $$F(\theta, x) = F_\theta(x)$$, being smooth at $$(\theta, z_i)$$, for every $$z_i\in Z$$. One can also define the so-called **batch functional dimension** at $$\theta$$ for a batch $$Z$$, which is equal simply to the rank of $$\mathbf{J}E_Z\lvert_{\theta}$$. Clearly, the batch functional dimension is less than or equal to the functional dimension.

In Section 5 of \[GLMW22\], for a given network $$\mathcal N$$ with parameter $$\theta$$, the authors describe a type of finite set in $$\mathbb R^{n_0}$$ for which the batch functional dimension for that set is equal to $$\dim_{fun}(\theta)$$. To confirm that you have such a set, called a **decisive set** for $$\mathcal N(\theta)$$, requires knowledge of the regions in $$\mathbb R^{n_0}$$ on which $$F_{\theta}$$ is affine-linear; more precisely, for each top-dimensional cell of the canonical polyhedral complex of $$F_\theta$$, a decisive set contains the vertices of some $$n_0$$-dimensional simplex contained in that cell. Provided that $$\theta$$ is "nice" and $$Z\subset\mathbb R^{n_0}$$ is a decisive set, we have $$\dim_{fun}(\theta) = \operatorname{rank}\mathbf{J}E_Z\lvert_{\theta}$$.

Finally, we will be interested in a certain restricted notion of functional dimension, when the set $$Z$$ is restricted to the interior of the positive orthant $$\mathbb R_{>0}^{n_0} = \{(x_1,\ldots,x_{n_0})\ \lvert\ x_i > 0 \text{ for all } 1\le i\le n_0\}$$. Write this as

$$\dim_{fun+}(\theta) = \underset{Z\subset\mathbb R_{>0}^{n_0} \text{ is finite and smooth for } \theta}{\sup}\operatorname{rank} \mathbf{J}E_Z\lvert_{\theta}.$$

After a small change to the notation, the following is the statement of Lemma 8.2 in \[GLMW22\].

##### Lemma 8.2
Fix $$n_0\in\mathbb N$$ and let $$\Omega$$ be the parameter space for architecture $$(n_1,n_2,\ldots,n_d)$$. Let $$\theta\in\Omega$$ be such that there is a smooth point in the strictly positive orthant of $$\mathbb R^{n_1}$$. Let $$A:\mathbb R^{n_0}\to\mathbb R^{n_1}$$ be affine-linear such that every row of the associated matrix has at least one non-zero entry. Use $$(A, \theta)$$ to denote the parameter for architecture $$(n_0,n_1,\ldots,n_d)$$ that precomposes with $$A$$, i.e., $$\Phi\circ A$$ is the first layer map of the network function $$F_{(A,\theta)}$$. <br> 
If $$(A, \theta)$$ satisfies nice conditions, then $$\dim_{fun}(A,\theta) \le n_0n_1 + \dim_{fun+}(\theta)$$. Furthermore, for this inequality to be equality it is necessary that $$\dim_{fun+}(\theta)$$ be realized on a smooth set $$Z^* \subset \operatorname{Im}(\Phi\circ A)$$. 

> ##### Proof sketch
> The proof of this lemma uses the scaling-inverse scaling invariance of $$F_{(A,\theta)}$$, in the first hidden layer. The authors use this to identify the parameter space with coordinates in the product $$\mathbb R_{>0}^{n_1} \times (\mathbb R^{n_0})^{n_1} \times \Omega$$. They then argue that, for $$Z\subset\mathbb R^{n_0}$$, the columns of $$\mathbf{J}E_Z\lvert_{(A,\theta)}$$ that correspond to parameters in $$\mathbb R_{>0}^{n_1}$$ will be zero; that the columns corresponding to parameters in $$(\mathbb R^{n_0})^{n_1}$$ can contribute at most $$n_0n_1$$ to the rank; finally, that the contribution to the rank from columns corresponding to parameters in $$\Omega$$ will at most be the $$\sup$$ of the rank of $$\mathbf{J}$$ of the evaluation map on subsets in $$\mathbb R_{>0}^{n_1}$$,[^4] evaluated at $$\theta$$, which equals $$\dim_{fun+}(\theta).$$ $$\blacksquare$$
{: .block-tip }

Note that all of the assumptions in Lemma 8.2 that occur before the word "Furthermore" will be true of a full measure subset in the parameter space for $$(A, \theta)$$. This is proven in \[GLMW22\] and remarked upon at the beginning of subsection 8.1.  Moreover, if $$n_0 \ge n_1$$ then there is a full measure subset of parameters so that $$A$$ is surjective, in which case $$\operatorname{Im}(\Phi\circ A) = \mathbb R_{\ge 0}^{n_1}$$. By definition, this contains any subset $$Z^*$$ that realizes $$\dim_{fun+}(\theta)$$. So, if $$n_0 \ge n_1$$ then a full measure set of parameters satisfies that necessary condition.

However, if $$n_0 < n_1$$ then $$\operatorname{Im}(\Phi\circ A)$$ cannot contain a set of points in $$\mathbb R_{>0}^{n_1}$$ that are vertices of an $$n_1$$-simplex – since $$\operatorname{Im}(\Phi\circ A) \cap \mathbb R_{>0}^{n_1}$$ is contained in an $$(n_1-1)$$-dimensional affine subspace. This means that any decisive set that realizes $$\dim_{fun+}(\theta)$$ is not in the image. However, perhaps it is possible yet to satisfy this condition sometimes.

**Question.** How do we understand the difference between $$\dim_{fun+}(\theta)$$ and $$\dim_{fun}(\theta)$$?

## Technical Lemma 8.5
---

Let's discuss Lemma 8.5. We'll need, at least, the notion of the ternary labeling at $$x \in \mathbb R^{n_0}$$, determined by $$\theta$$. At this juncture, a (minor) reckoning has arrived. As is common for parameterized classes of functions, and in mathematical modeling, we used the notation $$\theta$$ for our parameter vector in $$\mathbb R^{D}$$, with $$D$$ being the total number of weights and biases in network architecture $$(n_0,n_1,\ldots,n_d)$$. However, the character $$\theta$$ is used in the literature to denote ternary labelings, which are functions $$\mathbb R^{n_0} \to \{-1, 0, 1\}^{N}$$ with $$N = n_1+n_2+\ldots+n_d$$.

For this, we will try out the character $$\tau$$. So, a definition. For this definition, recall for each neuron $$(\ell, j)$$ the pre-activation function $$z^{\ell}_j:\mathbb R^{n_0}\to\mathbb R$$, when $$1\le \ell\le n_d$$ and $$1\le j\le n_\ell$$.

**Definition 3.** Let $$\theta\in\Omega$$ be the parameter for a network with architecture $$(n_0,n_1,\ldots,n_d)$$. For each neuron $$(\ell, j)$$ of the network, define $$\tau^\ell_j:\mathbb R^{n_0} \to \{-1, 0, 1\}$$ by setting $$\tau^\ell_j(x) = \text{sign}(z^\ell_j(x))$$. (Here, the function $$\text{sign}$$ returns 0 if the input is zero, 1 if it is positive, and -1 if it is negative.)
The (full) **ternary labeling** for the network is the function $$\tau: \mathbb R^{n_0} \to \{-1, 0, 1\}^{N}$$ which has a coordinate function for each neuron $$(\ell, j)$$, namely the function $$\tau^\ell_j$$.

Intuitively speaking, the value of $$\tau^\ell_j(x)$$ is positive if and only if the neuron $$(\ell, j)$$ is "on" or "activated" at the point $$x$$. It is zero at $$x$$ if and only if $$F_{\ell-1}\circ\ldots\circ F_1(x)$$ lies inside of the hyperplane $$\{y\ \lvert\ W^\ell y + b^\ell = 0\}$$.

For Lemma 8.5, we have a similar setup to Lemma 8.2. There is a parameter $$\theta\in\Omega$$, for a network with architecture $$(n_1,\ldots, n_d)$$, an affine-linear map $$A:\mathbb R^{n_0}\to\mathbb R^{n_1}$$ (use $$A$$ for the $$n_1\times(n_0+1)$$ matrix that determines it). We are interested in precomposing with $$A$$ to get a neural network with architecture $$(n_0,n_1,\ldots,n_d)$$ and with parameter $$(A, \theta)$$.  Now, before precomposing there are coordinate ternary labelings $$\tau^\ell_j$$ for every  $$1\le \ell\le n_d-1$$ and $$1\le j\le n_{\ell+1}$$[^5]

##### Lemma 8.5
Suppose that $$\theta$$ and $$A$$ are as above and that $$\theta$$ satisfies "nice conditions."  For $$1\le j\le n_1$$ and $$x\in \mathbb R^{n_0}$$, use $$\tau^A_j(x)$$ to denote the ternary label (at $$x$$) with respect to the $$j$$th row of $$A$$.  Assume that $$A$$ is non-degenerate, in the sense that for $$1\le j\le n_1$$ the set where $$\tau^A_j(x) = 0$$ is a hyperplane in $$\mathbb R^{n_0}$$.
We suppose that for every $$1\le k\le n_1$$, there exists $$y_k\in\mathbb R^{n_0}$$ such that <br>
&nbsp;&nbsp;&nbsp;&nbsp; (i) $$\tau^A_k(y_k) = 0$$, <br>
&nbsp;&nbsp;&nbsp;&nbsp; (ii) $$\tau^A_j(y_k) \ne 0$$ for all $$j \ne k$$, <br>
&nbsp;&nbsp;&nbsp;&nbsp; (iii) for all $$(\ell, j)$$ with $$1\le \ell\le n_d-1$$ and $$1\le j\le n_{\ell+1}$$, we have that $$\tau^\ell_j( \Phi\circ A(y_k) ) \ne 0$$, and <br>
&nbsp;&nbsp;&nbsp;&nbsp; (iv) the $$k$$th column of $$\mathbf{J}F_{\theta}\lvert_{\Phi\circ A(y_k)}$$ is not the zero vector.[^6] <br>
Then there is a finite set $$Z$$ in $$\mathbb R^{n_0}$$ such that (v) up to scaling rows of $$A$$ by positive numbers, each entry of $$A$$ is given by a unique affine-linear combination of the coordinates of the vector $$E_Z(A, \theta)$$; and (vi) the ternary labeling for $$(A, \theta)$$ of every point in $$Z$$, at every neuron, is non-zero. <br>
Finally, the lemma also states that if $$y_k \in \mathbb R_{>0}^{n_0}$$ for every $$k$$ then (vii) the set $$Z$$ can be chosen to be in $$R_{>0}^{n_0}$$.

<br> 

Now, finding a set $$y_1, \ldots, y_{n_1}$$ that satisfy conditions (i) - (iii) is possible generically – the conditions (i) and (ii) can be guaranteed as long as the hyperplane arrangement in $$\mathbb R^{n_0}$$ associated to $$A$$ is generic; condition (iii) says that each of the $$n_1$$ points $$\Phi\circ A(y_k)$$ is contained in a top-dimensional cell of the canonical polyhedral decomposition for $$\mathcal N(\theta)$$ and this can be achieved for a choice of $$y_1,\ldots,y_{n_1}$$ by a perturbation of $$\theta$$.

However, there is a positive measure subset of parameters for which condition (iv) will be impossible to satisfy. For example, there is a positive measure subset such that $$F_\theta$$ is a constant function on all of $$\mathbb R^{n_1}$$. Using $$H_k$$ to denote the hyperplane where $$\tau^A_k$$ is zero, there is a larger subset on which $$F_\theta$$ is constant on the set $$\Phi\circ A(H_k)$$ (and in any "nearby" choice of $$A$$ too).

> ##### Proof sketch of Lemma 8.5
> To prove Lemma 8.5, the authors of \[GLMW22\] first note that, as a consequence of the assumptions (i) - (iii), for each $$1\le k\le n_1$$, there is an open neighborhood $$U_k$$ of $$y_k$$ so that the ternary labelings of all neurons in $$\mathcal N(A, \theta)$$ are constant on $$U_k$$ except for $$\tau^A_k$$. Furthermore, letting $$U_k^+$$ and $$U_k^-$$ denote the connected components of $$U_k \setminus \{x\ \lvert\ \tau^A_k(x) = 0\}$$, with sign of $$\tau^A_k$$ on each component matching the superscript, we have that the ternary labeling on every neuron of $$(A, \theta)$$ is constant on $$U_k^+$$ and on $$U_k^-$$. As a consequence, $$F_{(A,\theta)}$$ is affine-linear when restricted to either $$U_k^+$$ or $$U_k^-$$, and so $$\mathbf{J}F_{(A,\theta)}$$ is constant on each of $$U_k^{\pm}$$. Furthermore, $$\mathbf{J}F_\theta$$ is constant when restricted to $$\Phi\circ A(U_k)$$.
> 
> Next, they show that $$\mathbf{J}F_{(A,\theta)}\lvert_{U_k^+} \ne \mathbf{J}F_{(A,\theta)}\lvert_{U_k^-}$$. To do so, they use the chain rule and that $$F_{A,\theta} = F_\theta\circ (\Phi\circ A)$$. Then, since $$\mathbf{J}(\Phi\circ A)\lvert_{U_k^+}$$ contains a non-zero element in the $$k$$th row, and $$\mathbf{J}(\Phi\circ A)\lvert_{U_k^-}$$ has a zero $$k$$th row, assumption (iv) guarantees a non-zero difference in one of the entries of $$\mathbf{J}F_{(A,\theta)}\lvert_{U_k^+} - \mathbf{J}F_{(A,\theta)}\lvert_{U_k^-}$$.
> 
> Having determined that $$\mathbf{J}F_{(A,\theta)}\lvert_{U_k^+} \ne \mathbf{J}F_{(A,\theta)}\lvert_{U_k^-}$$, they have a lemma (Lemma 8.3) that gives the conclusion (v). This lemma produces a set $$Z \subset U_k^+ \cup U_k^-$$, which means that (vi) holds (by construction of $$U_k^{\pm}$$) and that (vii) must hold – making the neighborhood $$U_k$$ smaller if needed. $$\blacksquare$$
{: .block-tip }

##### Lemma 8.3. 
Let $$M$$ be a polyhedral complex embedded in $$\mathbb R^d$$, $$d\ge 1$$, and let $$F:\mathbb R^d \to \mathbb R^{n}$$ be a continous map that is affine-linear on cells of $$M$$. Let $$X, Y$$ be two $$d$$-dimensional cells of $$M$$ that share a $$(d-1)$$-dimensional facet, and denote the hyperplane containing the shared facet by $$H$$. Assume that $$\mathbf{J}F\lvert_X \ne \mathbf{J}F\lvert_Y$$. Then, for any decisive sets, $$S_X\subset X$$ for $$F\lvert_X$$ and $$S_Y\subset Y$$ for $$F\lvert_Y$$, $$H$$ is the solution set to an affine-linear equation $$\{x\ \lvert\ c + Ax = \mathbf{0}\}$$ where every entry of $$A$$ is an affine linear expression in the coordinates of $$E_{S_X\cup S_Y}(F)$$. The matrix $$A$$ is unique up to rescaling rows by constants.

> ##### Proof sketch 
> To prove Lemma 8.3, write the points $$S_X = \{z_0,z_1,\ldots,z_d\}$$ (which are vertices of a $$d$$-dimensional simplex in $$X$$, owing to the fact that $$F\lvert_X$$ is affine-linear). Now, since the vectors $$u_i := z_i - z_0$$, with $$1\le i\le d$$, make a basis of $$\mathbb R^d$$, each partial derivative $$\partial F/\partial x_i$$ is a linear combination of the directional derivatives $$D_{u_i}F(z_0)$$. Additionally, $$\lvert z_i-z_0\rvert D_{u_i}F(z_0)$$ is the difference between two coordinates of $$E_{S_X}(F)$$. And so, for every $$x\in X$$, each entry of $$\mathbf{J}F\lvert_x$$ is a linear combination of coordinates of $$E_{S_X}(F)$$.  This is similarly true for $$\mathbf{J}F\lvert_y$$, $$y\in Y$$, and $$E_{S_Y}(F)$$.
> 
> Note that the extension to $$\mathbb R^d$$ of the affine-linear map $$F\lvert_X$$ can be expressed as $$\mathbf{x} \mapsto c_X + \mathbf{J}F\lvert_X\mathbf{x}$$, for some constant vector $$c_X$$ (and an analogous statement is true for $$Y$$). Since the hyperplane $$H$$ that $$X$$ and $$Y$$ share consists of those $$\mathbf{x}$$ where the extension of $$F\lvert_X$$ and the extension of $$F\lvert_Y$$ agree, we have
> 
> $$H = \{\mathbf{x} | c_X - c_Y + (\mathbf{J}F\lvert_X - \mathbf{J}F\lvert_Y)\mathbf{x} = \mathbf{0} \},$$
> 
> proving the statement. $$\blacksquare$$
{: .block-tip }

We now discuss Theorem 8.7, which provides sufficient conditions to have equality: $$\dim_{fun}(A, \theta) = n_0n_1 + \dim_{fun}(\theta)$$.

##### Theorem 8.7. 
Fix a parameter $$\theta\in\Omega$$ which is "nice" and suppose that $$Z_1 \subset \mathbb R_{>0}^{n_1}$$ is a finite set whose ternary labels with respect to every neuron of $$\mathcal N(\theta)$$ are nonzero, and so that $$\dim_{fun}(\theta) = \operatorname{rank} \mathbf{J}E_{Z_1}\lvert_\theta$$. Suppose that $$A:\mathbb R^{n_0}\to\mathbb R^{n_1}$$ is a surjective affine-linear map that satisfies all the assumptions of Lemma 8.5 (including that every $$y_k$$ is in $$\mathbb R_{>0}^{n_0}$$). Then there is a finite set $$Z \subset \mathbb R_{>0}^{n_0}$$ such that the ternary labeling, for all $$z\in Z$$ and every neuron of $$\mathcal N(A,\theta)$$, is nonzero, and

$$\dim_{fun}(A, \theta) = \operatorname{rank}\mathbf{J}E_Z\lvert_{(A,\theta)} = n_0n_1 + \dim_{fun}(\theta).$$

The proof of Theorem 8.7 is a bit more involved than the proofs of the lemmas above. However, let us remark on the assumptions being made to get the conclusion of this theorem.

First, the existence of a set $$Z_1$$, as in the theorem statement, requires that $$\dim_{fun+}(\theta) = \dim_{fun}(\theta)$$. It would be valuable to understand what causes this to occur.

Second, the surjectivity assumption on $$A$$ requires that $$n_0 \ge n_1$$. It also includes some restrictive assumptions (not full measure) in order to guarantee assumption (_iv_) of Lemma 8.5, as we discussed above, as well as guaranteeing that $$y_k$$ can be chosen from $$\mathbb R_{>0}^{n_0}$$ for each $$k$$. (For example, this is impossible if one of the hyperplanes/neurons of $$\Phi\circ A$$ does not cut through the positive orthant.)

## The Question of $$\dim_{fun+}(\theta)$$ vs. $$\dim_{fun}(\theta)$$
---

As a start for understanding when $$\dim_{fun+}(\theta)$$ and $$\dim_{fun}(\theta)$$ are the same, let us make a simple observation. Let $$x_0, x_1, \ldots, x_{n_0}$$ be affinely independent points in $$\mathbb R^{n_0}$$ and let $$y_0,y_1,\ldots, y_{n_0} \in\mathbb R$$. There is a unique affine linear function $$F:\mathbb R^{n_0} \to \mathbb R$$ with the property that $$F(x_i) = y_i$$ for every $$0\le i\le n_0$$. This falls out of linear algebra.

Indeed, since the points are affinely independent, $$\{x_i - x_0\ \lvert\ 1\le i\le n_0\}$$ is a basis of $$\mathbb R^{n_0}$$. The function $$F$$ is determined by $$n_0+1$$ scalars $$a_0,a_1,\ldots, a_{n_0}$$, so that, writing $$\mathbf{a}$$ for $$(a_1,a_2,\ldots,a_{n_0})$$, we have $$F(x) = a_0 + \mathbf{a}\cdot x$$ for all $$x\in\mathbb R^{n_0}$$. Note that $$y_i - y_0 = \mathbf{a}\cdot (x_i - x_0)$$. Therefore, since for any $$x\in \mathbb R^{n_0}$$, we have scalars $$c_1,\ldots, c_{n_0}$$ so that $$x - x_0 = \sum_{i=1}^{n_0} c_i(x_i - x_0)$$, we see that

$$F(x) - F(x_0) = \mathbf{a}\cdot (x - x_0) = \sum_{i=1}^{n_0} c_i\mathbf{a}\cdot(x_i - x_0) = \sum_{i=1}^{n_0} c_i(y_i - y_0).$$

Therefore, $$F(x) = y_0 + \sum_{i=1}^{n_0} c_i(y_i - y_0)$$.

Let's consider a scenario.  Given an architecture of a ReLU network and a parameter $$\theta\in\Omega$$ that is "nice," say that we have a point $$p$$ in the interior of a top-dimensional cell of the canonical polyhedral complex $$\mathcal C = \mathcal C(\theta)$$. Write $$X$$ for this cell containing $$p$$. Further, suppose that $$Z\subset \mathbb R^{n_0}$$ is a finite set so that:
1. $$Z$$ is the union of points that are in the interior of top-dimensional cells of $$\mathcal C$$;
2. there is a vertex $$v$$ of $$X$$ such that for every top-dimensional cell $$A \ne X$$ which has $$v$$ as one of its vertices, $$int(A) \cap Z \ne\emptyset$$; and
3. if $$A$$ is a top-dimensional cell and $$int(A)\cap Z\ne \emptyset$$, then $$Z$$ contains a decisive set (in $$A$$) for $$F_\theta\lvert_A$$.[^7]

We _hope_ that this will mean that $$\operatorname{rank}\mathbf{J}E_Z\lvert_\theta = \operatorname{rank}\mathbf{J}E_{Z^*}\lvert_\theta$$, where $$Z^* = Z \cup \{p\}$$.

The intuition for this would be that, first, since $$Z$$ has a decisive set on each cell $$A$$ that it intersects, $$F_\theta\lvert_A$$ is determined by values at points in $$Z\cap A$$. This means that $$F_\theta$$ is determined on the boundary of $$X$$ on an affinely independent set of points.  Thus, it would seem that our set $$Z$$ completely determines $$F_\theta\lvert_{X}$$. That is, we should be able to "witness" the partial derivatives in $$E_{\{p\}}\lvert_\theta$$ through rows of $$E_{Z}\lvert_\theta$$.

Let $$A$$ be a top-dimensional cell of $$\mathcal C(\theta)$$. The first thing to observe is that, related to the fact that an affine-linear function is determined by its values on $$n_0+1$$ affinely independent points, if $$x_0,x_1,\ldots,x_{n_0}$$ in $$int(A)$$ is a set of affinely independent points then, for any $$x\in int(A)$$, we can get $$\mathbf{J}E_x\lvert_{\theta}$$ as a linear combination of $$\mathbf{J}E_{x_i}\lvert_{\theta}, 0\le i\le n_0$$. Indeed, there is a unique set of scalars $$c_1,\ldots, c_{n_0}$$ so that $$x - x_0 = \sum_{i=1}^{n_0} c_i (x_i - x_0)$$.  Note that each element of $$\mathbf{J}E_{x}\lvert_\theta$$ is either linear or constant in $$x$$ (when restricting to $$int(A)$$). This means that

\begin{equation}
\label{eq:vertex-loop}
\mathbf{J}E_{x}\lvert_\theta - \mathbf{J}E_{x_0}\lvert_\theta = \sum_{i=1}^{n_0}c_i (\mathbf{J}E_{x_i}\lvert_\theta - \mathbf{J}E_{x_0}\lvert_\theta),
\end{equation}

which expresses $$\mathbf{J}E_{x}\lvert_{\theta}$$ in the desired way as a linear combination.

Now, the vector $$\mathbf{J}E_{x}\lvert_{\theta}$$ is not defined if $$x$$ is contained in a facet of $$A$$. [^8] However, suppose that we use $$(*)$$ to determine such a vector – this would be the limit of $$\mathbf{J}E_{p_i}\lvert_{\theta}$$ for a sequence $$\{p_i\} \subset int(A)$$, where $$ p_i \to x $$. Since it depends on "converging from $$int(A)$$", say that we call this vector $$\mathbf{J}^AE_x\lvert_{\theta}$$.  In fact, from here on we will drop the notation that indicates evaluation at $$\theta$$, considering that as understood; hence, call this vector simply $$\mathbf{J}^AE_x$$.

Taking the above construction a step farther, since $$\{x_i - x_0\ \lvert\ 1\le i\le n_0\}$$ is a basis of $$\mathbb R^{n_0}$$ we could determine $$\mathbf{J}^AE_x$$ from (\eqref{eq:vertex-loop}), for any $$x \in \mathbb R^{n_0}$$.  Another perspective on this would be to consider how the parameters in $$\theta$$ express an affine linear function $$\mathbb R^{n_0} \to \mathbb R^{n_d}$$ which has a restriction to $$A$$ that agrees with the restriction of $$F_\theta$$. Then $$\mathbf{J}^AE_x$$ is the Jacobian of the evaluation map, at $$x$$, corresponding to that affine linear function.

Now, if $$A \ne X$$ is one of the cells having non-empty intersection with $$Z$$ in conditions 1, 2, and 3 above, then for any $$x\in\mathbb R^{n_0}$$, $$\mathbf{J}^AE_x$$ is a linear combination of rows of $$\mathbf{J}E_Z$$. While we are still figuring out how it works in general, let's consider a special case.

**How it works when $$n_0 = 2$$ and all bent hyperplanes at the vertex come from one layer.** Under our "nice" assumptions when $$n_0=2$$, for any vertex $$v$$ of $$X$$, there will be 4 top-dimensional cells of $$\mathcal C(\theta)$$ for which $$v$$ is a vertex (including $$X$$) with two (bent) hyperplanes intersecting at $$v$$. The ternary labeling for each of the hyperplanes will be positive in exactly two of these 4 cells (in their interior), in such a manner that we may associate in one-to-one manner these cells to elements of $$\{+, -\}^4$$. Let $$C$$ and $$\bar C$$ be the two cells which, from ternary labels, we associate to $$(+,-)$$ and $$(-,+)$$. Let $$D$$ and $$\bar D$$ be the cells which we associate to $$(+,+), (-, -)$$.
**Claim.** $$\mathbf{J}^{C}E_v + \mathbf{J}^{\bar C}E_v - \mathbf{J}^{D}E_v - \mathbf{J}^{\bar D}E_v = \mathbf{0}$$.

> To prove this, note that for each of $$C, \bar C, D$$, and $$\bar D$$, the function  in each column of $$\mathbf{J}E_v$$ restricts in the interiors to a polynomial, expressible so that every monomial in this polynomial is degree at most 1 in the parameter coordinates, and is degree 1 or less in the coordinates of $$v$$, as well.  Suppose that such a monomial is degree 0 in every parameter coordinate corresponding to these two hyperplanes – that is, it is degree 0 in every one of the $$2(n_{\ell-1}+1)$$ coordinates for the rows of $$(W^{\ell} | b^{\ell})$$ that correspond to these neurons, and it is also degree 0 in all $$2n_{\ell+1}$$ coordinates appearing in the columns of $$W^{\ell+1}$$ that correspond to these two neurons. Then this monomial appears in a column of $$\mathbf{J}^{C}E_v$$ if and only if it appears in the same column of $$\mathbf{J}^{\bar C}E_v, \mathbf{J}^{D}E_v$$, and $$\mathbf{J}^{\bar D}E_v$$. Note, since these hyperplanes come from a "hidden layer", any monomial of $$E_x$$, $$x$$ in the interior of one of these cells, that is positive degree in one of the coordinates corresponding to the two hyperplanes must be degree 1 in _two_ such coordinates -- one from $$(W^{\ell} | b^{\ell})$$ and one from $$W^{\ell+1}$$. Thus, in the $$2(n_{\ell-1}+1) + 2n_{\ell+1}$$ columns for partials with respect to such coordinates every non-zero monomial is degree 1 in one of those coordinates.  Hence, all monomials in every column that have degree 0 in those coordinates will vanish in the summation $$\mathbf{J}^{C}E_v + \mathbf{J}^{\bar C}E_v - \mathbf{J}^{D}E_v - \mathbf{J}^{\bar D}E_v$$.
> Now, suppose that some monomial in $$\mathbf{J}^{A}E_v$$, for $$A = C, \bar C, D,$$ or $$\bar D$$, has degree 1 in one of these coordinates. It cannot be that $$A = \bar D$$ since both of the neurons in question are unactivated in $$\bar D$$.  Moreover, in each column, such a monomial occuring in $$\mathbf{J}^DE_v$$ must be precisely the sum of monomials that separately occur in $$\mathbf{J}^CE_v$$ and $$\mathbf{J}^{\bar C}E_v$$.
> Since no monomial in $$\mathbf{J}^{A}E_v$$ can be larger than degree 1 in these coordinates, this shows the equation holds.
{: .block-tip }

While the discussion of the claim discusses the Jacobian of the evaluation map at $$v$$, it would appear that it holds at _any_ point. Using that $$X$$ is one of $$C, \bar C, D$$, or $$\bar D$$, the claim tells us that $$\mathbf{J}^XE_{v}$$ (resp. $$\mathbf{J}^XE_{p}$$) is a linear combination of vectors $$\mathbf{J}^AE_{v}$$ (resp. $$\mathbf{J}^AE_{p}$$), where $$A$$ takes on the other three cells (in each of which we have a subset of $$Z$$ that is affinely independent. Since, for each $$A\in \{C, \bar C, D, \bar D\} \setminus \{X\}$$, we may write $$\mathbf{J}^AE_v$$ and $$\mathbf{J}^AE_p$$ as a linear combination of rows of $$\mathbf{J}E_Z$$, this tells us that $$\mathbf{J}^XE_p = \mathbf{J}E_p$$ can be expressed as a linear combination of rows of $$\mathbf{J}E_Z$$.

---

[^1]: I could have written $$z^{n_d}_j$$ instead for the $$j$$-th coordinate function. However, this choice will make for simpler notation below.

[^2]: In other words, think of the function $$f_{j,z_i}:\Omega\to\mathbb R$$ which is given by $$f_{j,z_i}(\theta) = F_{\theta,j}(z_i)$$ and take partial derivatives of $$f_{j,z_i}$$.

[^3]: In terminology of \[GLMW22\], it is an _ordinary point_.

[^4]: Which fact is related to $$\Phi\circ A(Z)$$ being a subset of  $$\mathbb R_{\ge 0}^{n_1}$$.

[^5]: Shifting the index here, since the first hidden layer of the network of $$\theta$$ has $$n_2$$ neurons, and so on.

[^6]: The partial derivatives in $$\mathbf{J}F_{\theta}\lvert_{\Phi\circ A(y_k)}$$ does not involve partials with respect to parameters, but with respect to spatial coordinates in $$\mathbb R^{n_1}$$; i.e., with respect to coordinates $$(x_1,x_2,\ldots,x_{n_1})$$ in $$\mathbb R^{n_1}$$.

[^7]: We also assume that the polyhedral complex is generic and transversal so, in particular, any subset of the supporting hyperplanes that determine the facets $$X\cap X_i$$ (which has cardinality $$\le n_0$$), has a non-empty intersection that is some face of $$X$$.

[^8]: The facet is associated to the bent hyperplane for one neuron, $$\{x\ \lvert\ \tau^{\ell}_j(x) = 0\}$$; for any weight or bias "leading to" that neuron, in row $$j$$ of $$W^{\ell}$$ or $$b^{\ell}$$, the partial of $$E_x$$ with respect to that weight or bias will be undefined.

