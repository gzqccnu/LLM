# Math mechanism of RoPE 

It's now the most popular LLMs' embedding method.

Here we also present its math machanism and implement it in code.

> [!NOTE]
> 
> Below is all prerequisite knowledge and math proof, if you are not interested in it, you can skip it. However in my advice, you can simply read it, cause I write it very very simple for you. No intermediate steps are omitted. So you can just read it.
>
> About this part of math mechanism, you can also read it here [Original Paper of RoPE](https://arxiv.org/pdf/2104.09864). The following proof can be seen as presenting some omitted intermediate steps.

0. Background:
    <br>

    RoPE is relative to a module named transformer. That we will introduce after this in [transformer](../transformer/transformer.ipynb). And here we simply talk about it. In transformer, it receives the text, actually after the embedding i.e. word vector with the pos embedding. It will use three matrix: $ Q $, $ K $, $ V $ multiply every input text generating three new vector named: $ q $, $ k $, $ v $. Then compute the dot mutiplication of $ q $ and $ k $. Remember that we have a the pos embedding on each word of the text. So the $ q $, $ k $ of the word also have it. Then we could begin derivation.

Before that, we should have some more math base. Below we will introduce.

1. Complex number:
    <br>

    Note: $ i = \sqrt{-1} $, $ i^{2} = -1 $.
    And the complex number's base format is: $ a + b i$, in which $ a $ and $ b $ all are real number. In which we call $ a $ the real part. $ b $ the imaginative part.
    We also have another notation of complex number. Do you know the god's formula? It like this: $ e^{i \pi} + 1 = 0 $. It comes from 
    $$
    e^{i \theta} = \cos \theta + i \sin \theta \tag{3} 
    $$
    which is called complex number's **complex exponential form** or called **Euler's formula**. If you let $ \theta $ equals to $ \pi $.
    $$
    e^{i \pi} = \cos \pi + i \sin \pi
    $$
    We know $ \cos \pi = -1 $, $ \sin \pi = 0 $, so we get $ e^{i \pi} + 1 = 0 $.
    Here, we take a example or deep into it. Like real number, we have Cartesian coordinate system. With it we can decide a point by its `x` axis coordinate and `y` axis coordinate. Also we can use this point to (0, 0) distance $ r $, and $ \theta $ the angle with the x-axis to decide its position. That's the same in complex number. But the Cartesian coordinate system of complex number we call it **Complex plane**. Its x axis is the same with Cartesian coordinate system. However the y axis's coordinate unit is $ i $. So the complex number $ 3 + 4 i $ in the complex plane like this:

    <center><img src=https://github.com/gzqccnu/img/blob/main/complex_plane.png?raw=true)></center>

    In which, $ r = \sqrt{3^{2} + 4^{2}} = 5 $. Generally, given a complex number $ a + b i $, its $ r $ equals to $ \sqrt{a^{2} + b^{2}} $. 

2. Conjugate complex number:
    <br>

    $ a + b i $'s conjugate complex number equals to $ a - b i $. That's to say: one complex number and its conjugate complex number they have the same real part but the opposite imaginary part. If we note $ x = a + b i $, then we have $ x^{*} = a - b i $.  We expand to **complex exponential form**. If we have a complex number $ a + bi $, 
    we can convert it to(assume $ \theta $ stands for its angle in complex plane ): $ \cos \theta + i \sin \theta $. And its conjugate complex number can expressed in: $ \cos \theta - i \sin \theta $.
3. Dot of complex numbers:
    <br>

    Consider we have two complex numbers: $ x = a + b i $ and $ y = c + d i $. Then we want to compute the dot mutiplication of $ x $ and $ y $.
    We have the principle:
    $$
    x \cdot y = \langle x, y* \rangle = (a + b i) \cdot (c - d i) = ac + b(-d)i^{2} = ac + bd \tag{4}
    $$

Note we begin with a $ \textbf{2D} $ case.
Consider we already have the $ \boldsymbol{x}_{q} $ and $ \boldsymbol{x}_{k} $. Their positions are respectly $ m $ and $ n $. Now we must to find a way or function:
$$
\boldsymbol{f} (\boldsymbol{vec}, pos)
$$
So we have 
$$
\boldsymbol{q}_{m} = \boldsymbol{f}_{q}(\boldsymbol{x}_{q}, m) \tag{5}
$$

$$
\boldsymbol{k}_{n} = \boldsymbol{f}_{k}(\boldsymbol{x}_{k}. n) \tag{6}
$$

for vector $ \boldsymbol{q}_{m} \cdot \boldsymbol{k}_{n} $ only depend on relative position $ m-n $. And now consider this case: we have vectors $ \boldsymbol{A} $ and vector $ \boldsymbol{B} $ are parallel to x-axis. Then we rotate $ \boldsymbol{A} $ by $ +30 $ degrees with $ \boldsymbol{B} $ by $ +40 $ degrees. And the angle between them is 10 degrees. Expressed in math is:
$$
\cos \langle \boldsymbol{A}, \boldsymbol{B} \rangle = \frac{\boldsymbol{A} \cdot \boldsymbol{B}}{||A|| \ ||B||}
$$
Analogy to position encoding, we see position $ m $ as a rotation angle $ m \theta $. So, the dot multiplication of $ \boldsymbol{q}_{m} $ and $ \boldsymbol{k}_{n} $ only depends on $ (m - n) \theta $ i.e. the relative position $ m - n $.

That's it. Now we formally entry proof.

Note the function only depends on relative position, that's to say, it has the property:
$$
\boldsymbol{q}_{m}^{\boldsymbol{T}} \boldsymbol{k}_{n} = \boldsymbol{f}(\boldsymbol{x}_{q}, m) \cdot \boldsymbol{f}(\boldsymbol{x}_{k}, n) = \langle \boldsymbol{f}(\boldsymbol{x}_{q}, m), \boldsymbol{f}(\boldsymbol{x}_{k}, n) \rangle = g(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, m - n) \tag{7}
$$
$ g $ is for better reading experience.

To solve the identity equation, we must to have some initial conditions:
$$
\boldsymbol{f}_{q} (\boldsymbol{x}_{q}, 0) = \boldsymbol{q} \tag{8}
$$
$$
\boldsymbol{f}_{k} (\boldsymbol{x}_{k}, 0) = \boldsymbol{k} \tag{9}
$$

Based on the **complex exponential form** and take advantage of the geometric meaning of vector in $ \textbf{2D} $ and its complex counter part,
decompose functions in Equations (5) and (6) into
$$
\boldsymbol{f}_{q} (\boldsymbol{q}, m) = R_{q}(\boldsymbol{x}_{q}, m) e^{i \Theta_{q}(\boldsymbol{x}_{q}, m)} \tag{9}
$$

$$
\boldsymbol{f}_{k}(\boldsymbol{x}_{k}, n) = R_{k}(\boldsymbol{x}_{k}, n) e^{i \Theta_{k}(\boldsymbol{x}_{k}, n)} \tag{10}
$$
(6) into
$$
g(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m) = R_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m) e^{i \Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m)}
$$

where $ R_{f} $ , $ R_{g} $ and $ \Theta_{f} $ , $ \Theta_{g} $ are the radical and angular components for $ \boldsymbol{f}_{\{q, k\}} $ and $ g $, respectively. Plug them into Equation (7), we get the relation:
$$
R_{q}(\boldsymbol{x}_{q}, m)R_{k}(\boldsymbol{x}_{k}, n) = R_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m) \tag{11}
$$
$$
\Theta_{k}(\boldsymbol{x}_{k}, n) - \Theta_{q}(\boldsymbol{x}_{q}, m) = \Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m) \tag{12}
$$

<hr style="border: 3px dashed #ccc; width: 100%;">

Here, I want to explain why in 
$$
\Theta_{k}(\boldsymbol{x}_{k}, n) - \Theta_{q}(\boldsymbol{x}_{q}, m) 
$$
genrate `negative sign`. Do you remember Equation (4). We say two complex number's dot product is equal to the product of the conjugate of one complex number and another complex number. Using **complex exponential form**, we have
$$
x = a + bi = \sqrt{a^{2} + b^{2}} (\cos \theta + i \sin \theta) = \sqrt{a^{2} + b^{2}} e^{i \theta}
$$
$$
y = c + di = \sqrt{c^{2} + d^{2}} (\cos \theta + i \sin \theta) = \sqrt{c^{2} + d^{2}} e^{i \theta}
$$
So we have
$$
y^{*} = c - di = \sqrt{c^{2} + d^{2}} (\cos \theta - i \sin \theta) \tag{13}
$$
And how its complex exponential form like? Do you remember formula (3)?
We could Solving by substitution method. 
We know for a complex number: $ a + b i$, its complex exponential form is: 
$$
r e^{i \theta} = r ( \cos \theta + i \sin \theta ) 
$$
It's conjugate form like:  
$$
r ( \cos \theta - i \sin \theta ) 
$$
If we use $ i(-\theta) $ replace $ \theta $ in that form, we get
$$
e^{i (-\theta)} = \cos (-\theta) + i \sin (-\theta) \tag{14}
$$
We know
$$
\cos (-\theta) = \cos \theta
$$
$$
\sin (-\theta) = - \sin \theta
$$
So equation (12) like
$$
e^{i (-\theta)} = \cos \theta - i \sin \theta
$$
So equation (13) equals to
$$
y^{*} = c - di = \sqrt{c^{2} + d^{2}} (\cos \theta - i \sin \theta) = \sqrt{c^{2} + d^{2}} e^{ - i \theta}
$$
According to the law of exponentiation, we obtain equation (11) and (12)
<hr style="border: 3px dashed #ccc; width: 100%;">
with the corresponding initial condition as:

$$
\boldsymbol{q} = ||\boldsymbol{q}|| e^{i \theta_{q}} = R_{q}(\boldsymbol{x}_{q}, 0) e^{i \Theta_{q}(\boldsymbol{x}_{q}, 0)}
$$
$$
\boldsymbol{k} = ||\boldsymbol{k}|| e^{i \theta_{k}} = R_{k}(\boldsymbol{x}_{k}, 0) e^{i \Theta_{k}(\boldsymbol{x}_{k}, 0)}
$$
where $ ||\boldsymbol{q}|| $, $ ||\boldsymbol{k}|| $ and $ \theta_{q} $, $ \theta_{k} $ are the radial and angular part of $ \boldsymbol{q} $ and $ \boldsymbol{k} $ on the $ \textbf{2D} $ plane.
Next, we set $ m = n $ in Equation (11) (12) and take into account initial conditions in Equation (8) (9):
$$
R_{q}(\boldsymbol{x}_{q}, m)R_{k}(\boldsymbol{x}_{k}, m) = R_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 0) = R_{q}(\boldsymbol{x}_{q}, 0)R_{k}(\boldsymbol{x}_{k}, 0) = ||\boldsymbol{q}|| \ ||\boldsymbol{k}|| \tag{15}
$$
$$
\Theta_{k}(\boldsymbol{x}_{k}, m) - \Theta_{q}(\boldsymbol{x}_{q}, m) = \Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 0) = \Theta_{k}(\boldsymbol{x}_{k}, 0) - \Theta_{q}(\boldsymbol{x}_{q}, 0) = \theta_{k} - \theta_{q} \tag{16}
$$
for we have
$$
R_{g}(\boldsymbol{x}_{q}, m) = R_{q}(\boldsymbol{x}_{q}, 0) = ||\boldsymbol{q}||
$$
$$
R_{k}(\boldsymbol{x}_{k}, n) = R_{k}(\boldsymbol{x}_{k}, 0) = ||\boldsymbol{k}||
$$
$$
R_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m) = R_{g} (\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 0) = ||\boldsymbol{q}|| \ ||\boldsymbol{k}||
$$
interprets that the radial functions $ R_{q} $ ,$ R_{k} $ and $ R_{g} $ are independent from the position information.

For Equation (16), we do a transposition then get:
$$
\Theta_{q}(\boldsymbol{x}_{}, m) - \theta_{q} = \Theta_{k}(\boldsymbol{x}_{k}, n) - \theta_{k}
$$
indicates that the angular functions does not dependent on $ \boldsymbol{x} $
so the term 
$$
\Theta_{f}(\boldsymbol{x}_{\{q, k\}}, m ) - \theta_{\{q, k\}}
$$
is a function of $ m $ and we denote it as $ \phi(m) $, yielding:
$$
\Theta_{f}(\boldsymbol{x}_{\{q, k\}}, m ) = \phi(m) + \theta_{\{q, k\}} \tag{17}
$$
Further, by plugging $ n = m + 1$ to Equation (11) (12) we get
$$
\Theta_{k}(\boldsymbol{x}_{k}, m + 1) - \Theta_{q}(\boldsymbol{x}_{q}, m) = \Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 1)
$$
and consider the above Equation (17) we have
$$
(\phi(m + 1) + \theta_{k}) - (\phi(m) + \theta_{q}) = \Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 1)
$$
so we get
$$
\phi(m + 1) - \phi(m) = \Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 1) + \theta_{q} - \theta_{k}
$$
Since RHS is a constant irrelevanttom $ \phi(m) $ with continuous integer inputs produce an arithmetic progression, we set 
$$
\theta = \Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 1) + \theta_{q} - \theta_{k}
$$
then set $ m $ to $(1, 2, 3..) $, we get:
$$
\phi(1) - \phi(0) = \theta
$$
$$
\phi(2) - \phi(1) = \theta
$$
$$
\phi(3) - \phi(2) = \theta
$$
$$
\vdots
$$
$$
\phi(m) - \phi(m - 1) = \theta
$$
So we get:
$$
\phi(m) - \phi(0) = m \theta
$$
we set
$$
\phi(0) = \gamma
$$
we plus all above differential terms regarding $ \phi $
$$
\phi(m) - \phi(m - 1) = \theta
$$
$$
+
$$
$$
\phi(m - 1) - \phi(m - 2) = \theta
$$
$$
+
$$
$$
\vdots
$$
$$
\phi(3) - \phi(2) = \theta
$$
$$
+
$$
$$
\phi(2) - \phi(1) = \theta
$$
$$
+
$$
$$
\phi(1) - \phi(0) = \theta
$$
finally we get
$$
\phi(m) = m \theta + \gamma \tag{18}
$$
where $ \theta $, $ \gamma \in \mathbb{R}$ are constants and $ \theta $ is non-zero.

Equation (18) reveal:
$$
\phi(m) = \Theta_{f}(\boldsymbol{x}_{\{q, k\}}, m ) - \theta_{\{q, k\}} = m \theta + \gamma
$$
so
$$
\Theta_{f}(\boldsymbol{x}_{\{q, k\}}, m ) = \theta_{\{q, k\}} + m \theta + \gamma
$$
so we have equation (9) and (10)
$$
\boldsymbol{f}_{q}(\boldsymbol{x}_{q}, m) = R_{q}(\boldsymbol{x}_{q}, m) e^{i \Theta_{q}(\boldsymbol{x}_{q}, m)} = R_{q}(\boldsymbol{x}_{q}, m) e^{i (\theta_{q} + m \theta + \gamma)} = ||\boldsymbol{q}|| e^{i (\theta_{q} + m \theta + \gamma)} \tag{19}
$$
$$
\boldsymbol{f}_{k}(\boldsymbol{x}_{k}, n) = R_{k}(\boldsymbol{x}_{k}, n) e^{i \Theta_{k}(\boldsymbol{x}_{k}, n)} = R_{k}(\boldsymbol{x}_{k}, n) e^{i (\theta_{k} + n \theta + \gamma)} = ||\boldsymbol{k}|| ^{i (\theta_{k} + n \theta + \gamma)} \tag{20}
$$
in view of
$$
||\boldsymbol{q}|| e^{i \theta_{q}} = \boldsymbol{q}
$$
$$
||\boldsymbol{k}||e^{i \theta_{k}} = \boldsymbol{k}
$$
we plug them into Equation (19) and (20), we get:
$$
\boldsymbol{f}_{q}(\boldsymbol{x}_{q}, m) = \boldsymbol{q} e^{i(m \theta + \gamma)}
$$
$$
\boldsymbol{f}_{k}(\boldsymbol{x}_{k}, n) = \boldsymbol{k} e^{i (n \theta + \gamma)}
$$
we set $ \gamma = 0 $,
$$
\boldsymbol{f}_{q}(\boldsymbol{x}_{q}, m) = \boldsymbol{q} e^{i m \theta} \tag{21}
$$
$$
\boldsymbol{f}_{k}(\boldsymbol{x}_{k}, n) = \boldsymbol{k} e^{i n \theta} \tag{22}
$$
we define 
$$
\boldsymbol{q} = \boldsymbol{f}(\boldsymbol{x}_{m}, 0) = \boldsymbol{W}_{q} \boldsymbol{x}_{m}
$$
$$
\boldsymbol{k} = \boldsymbol{f}(\boldsymbol{x}_{n}, 0) =\boldsymbol{W}_{k} \boldsymbol{x}_{n}
$$
where $ W_{q} $, $ W_{k} $ are all 2D matrix. Plug them into Equation (21) (22) we get:
$$
\boldsymbol{f}_{q}(\boldsymbol{x}_{q}, m) = (\boldsymbol{W}_{q} \boldsymbol{x}_{m}) e^{i m \theta} \tag{23}
$$
$$
\boldsymbol{f}_{k}(\boldsymbol{x}_{k}, n) = (\boldsymbol{W}_{k} \boldsymbol{x}_{n}) e^{i n \theta} \tag{24}
$$
So finally we get the position embedding function in $ \textbf{2D} $
<hr style="border: 3px dashed #ccc; width: 100%;">

Here we will introduce you a new theory a complex number can be expressed in **matrix** like:
$$
a + bi = \sqrt{a^{2} + b^{2}} e^{i \theta} = \sqrt{a^{2} + b^{2}} \begin{pmatrix}
             \cos \theta & - \sin \theta \newline
             \sin \theta & \cos \theta \newline
             \end{pmatrix} = \begin{pmatrix}
                             a & -b \newline
                             b & a \newline
                             \end{pmatrix} \tag{25}
$$
where
$$
\theta = \arctan (\frac{b}{a})
$$
Here we do not present strict mathematical derivation proofs, rather conduct validating proofs from algebraic operations perspective.

Assume we have two complex number:
$$
z_{1} = a + bi \tag{26}
$$
$$
z_{2} = c + di \tag{27}
$$
according to Equation (25) we get the equivalent form of Equation (26)
$$
z_{1} = \begin{pmatrix}
        a & - b \newline
        b & a \newline
        \end{pmatrix} \tag{28}
$$
and Equation (27)
$$
z_{2} = \begin{pmatrix}
        c & -d \newline
        d & c \newline
        \end{pmatrix} \tag{29}
$$
their product(not dot product) in Equation (26) (27) form, we get
$$
z_{1} z_{2} = (a + bi)(c + di) = ac + ad i + bc i + bd(i^{2}) = (ac - bd) + (ad + bc)i \tag{30}
$$
in Equation (28) (29) form, we get
$$
z_{1} z_{2} = 
        \begin{pmatrix}
        a & - b \newline
        b & a \newline
        \end{pmatrix} 
        \begin{pmatrix}
        c & -d \newline
        d & c \newline
        \end{pmatrix} = 
        \begin{pmatrix}
        ac - bd & - (ad + bc) \newline
        ad + bc & ac - bd \newline
        \end{pmatrix}
$$
and return it to complex number form, it equals to Equation (30)
<hr style="border: 3px dashed #ccc; width: 100%;">

According to Equation (25), we have new version of Equation (23) (24)
$$
\boldsymbol{f}_{q}(\boldsymbol{x}_{q}, m) = (\boldsymbol{W}_{q} \boldsymbol{x}_{m}) \begin{pmatrix}
                                                                                    \cos m \theta & - \sin m \theta \newline
                                                                                    \sin m \theta & \cos m \theta
                                                                                    \end{pmatrix}
$$
$$
\boldsymbol{f}_{k}(\boldsymbol{x}_{k}, n) = (\boldsymbol{W}_{k} \boldsymbol{x}_{n}) \begin{pmatrix}
                                                                                    \cos n \theta & - \sin n \theta \newline
                                                                                    \sin n \theta & \cos n \theta
                                                                                    \end{pmatrix}
$$
We can move $ e^{\{m, n\} \theta} $ forward, and the rotation matrix can also be moved forward.
$$
\boldsymbol{f}_{q}(\boldsymbol{x}_{q}, m) = \begin{pmatrix}
                                            \cos m \theta & - \sin m \theta \newline
                                            \sin m \theta & \cos m \theta
                                            \end{pmatrix} (\boldsymbol{W}_{q} \boldsymbol{x}_{m}) \tag{31}
$$
$$
\boldsymbol{f}_{k}(\boldsymbol{x}_{k}, n) = \begin{pmatrix}
                                            \cos n \theta & - \sin n \theta \newline
                                            \sin n \theta & \cos n \theta
                                            \end{pmatrix} (\boldsymbol{W}_{k} \boldsymbol{x}_{n}) \tag{32}
$$
cause all the above are in 2D dimension. So we can write Equation (31) (32) in this unified format
$$
\boldsymbol{f}_{\{q, k\}}(\boldsymbol{x}_{m}, m) = \begin{pmatrix}
                                                          \cos m \theta & - \sin m \theta \newline
                                                          \sin m \theta  & \cos m \theta
                                                          \end{pmatrix}
                                                          \begin{pmatrix}
                                                          W_{\{q, k\}}^{11} & W_{\{q, k\}}^ {12} \newline
                                                          W_{\{q, k\}}^{21} & W_{\{q, k\}}^{22}
                                                          \end{pmatrix}
                                                          \begin{pmatrix}
                                                          x_{m}^{(1)} \newline
                                                          x_{m}^{2}
                                                          \end{pmatrix} \tag{33}
$$
where $ (x_{m}^{(1)}, x_{m}^{(2)}) $ is $ x_{m} $ expressed in the 2D coordinate.
Then we promoted Equation (33) to a general form
$$
f_{\{q,k\}}(\boldsymbol{x}_m, m) = R_{\Theta, m}^d \, W_{\{q,k\}} \, \boldsymbol{x}_m
$$

where

$$
R_{\Theta, m}^d = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \newline
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \newline
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \newline
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \newline
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \newline
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \newline
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix} \tag{34}
$$
$$
x = \begin{bmatrix}
    x_{1} \newline
    x_{2} \newline
    x_{3} \newline
    \vdots \newline
    x_{d} \newline
    \end{bmatrix} \tag{35}
$$
and we split matrix (34) to block matrix format
$$
R_{\Theta, m}^d = \begin{pmatrix}
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 \newline
\sin m\theta_1 & \cos m\theta_1
\end{pmatrix}
& \begin{pmatrix} 0 & 0 \newline 0 & 0 \end{pmatrix}
& \cdots
& \begin{pmatrix} 0 & 0 \newline 0 & 0 \end{pmatrix} \newline[6pt]

\begin{pmatrix} 0 & 0 \newline 0 & 0 \end{pmatrix}
& \begin{pmatrix}
\cos m\theta_2 & -\sin m\theta_2 \newline
\sin m\theta_2 & \cos m\theta_2
\end{pmatrix}
& \cdots
& \begin{pmatrix} 0 & 0 \newline 0 & 0 \end{pmatrix} \newline[6pt]

\vdots & \vdots & \ddots & \vdots \newline[6pt]

\begin{pmatrix} 0 & 0 \newline 0 & 0 \end{pmatrix}
& \begin{pmatrix} 0 & 0 \newline 0 & 0 \end{pmatrix}
& \cdots
& \begin{pmatrix}
\cos m\theta_{d/2} & -\sin m\theta_{d/2} \newline
\sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
\end{pmatrix} \tag{36}
$$
with vector (35)
$$
\boldsymbol{x} = \begin{bmatrix}
    \begin{bmatrix}
    x_{1} \newline
    x'_{1} \newline 
    \end{bmatrix} \newline
    \begin{bmatrix}
    x_{2} \newline
    x'_{2} \newline 
    \end{bmatrix} \newline
    \vdots \newline
    \begin{bmatrix}
    x_{d / 2} \newline
    x'_{d / 2} \newline 
    \end{bmatrix}
    \end{bmatrix} \tag{37}
$$
we take a submatrix from matrix (36) except all zero matrix
$$
\boldsymbol{M}_{i} = \begin{pmatrix}
                     \cos m \theta_{i} & - \sin m \theta_{i} \newline
                     \sin m \theta_{i} & \sin m \theta_{i} \newline
                     \end{pmatrix} \ \text{or}
                     \begin{pmatrix}
                     0 & 0 \newline
                     0 & 0 \newline
                     \end{pmatrix}
$$
we see $ \boldsymbol{M}_{i} $ as a basic element of matrix (36)

we take a subvector from vector (37)
$$
\boldsymbol{x}_{i} = \begin{bmatrix}
                     x_{i} \newline
                     x'_{i} \newline
                     \end{bmatrix}
$$
denote it to $ \boldsymbol{x}_{i} $

we see $ \boldsymbol{x}_{i} $ as a basic element of vector $ \boldsymbol{x} $.
When matrix (36) mutiply vector (37), the computing algorithm of them is: Multiply each element in each row of the matrix by each element in x one by one. That's to say we see a row of the matrix as a vector. and what we do is let the row vector of the matrix multiply the vector $ \boldsymbol{x} $ and the result is still the row vector of the matrix. we can see it as below
$$
\boldsymbol{row\_vec}_{i} = \begin{bmatrix}
                 \boldsymbol{M}_{i, 1} & \boldsymbol{M}_{i, 2} & \cdots & \boldsymbol{M}_{i, d / 2}
                 \end{bmatrix}
$$
where $ i $ is row index of matrix, $ i \in \{1, 2, \cdots, d / 2\}$. 
$$
\boldsymbol{x} = \begin{bmatrix}
                 \boldsymbol{x}_{1} & \boldsymbol{x}_{2} & \cdots & \boldsymbol{x}_{d / 2}
                 \end{bmatrix}
$$
so the new row vector is
$$
\boldsymbol{new\_row\_vec}_{i} = \begin{bmatrix}
                           \boldsymbol{M}_{i, 1} * \boldsymbol{x}_{1} & \boldsymbol{M}_{i, 2} * \boldsymbol{x}_{2} & \cdots & \boldsymbol{M}_{i, d / 2} \boldsymbol{x}_{d / 2}
                           \end{bmatrix}
$$
so when matrix (36) multiply vector (37), the final result like
$$
R_{\Theta, m}^d \boldsymbol{x} = \begin{pmatrix}

                                    \begin{pmatrix}
                                    \cos m \theta_{1} & - \sin m \theta_{1} \newline
                                    \sin m \theta_{1} & \cos m \theta_{1} \newline
                                    \end{pmatrix}
                                    \begin{bmatrix}
                                    x_{1} \newline
                                    x'_{1} \newline
                                    \end{bmatrix} & \begin{pmatrix}
                                                    0 \newline
                                                    0
                                                    \end{pmatrix} & \cdots & \begin{pmatrix}
                                                                             0 \newline
                                                                             0
                                                                             \end{pmatrix} 

                                    \newline

                                    \begin{pmatrix}
                                    0 \newline
                                    0
                                    \end{pmatrix} & \begin{pmatrix}
                                                    \cos m \theta_{2} & - \sin m \theta_{2} \newline
                                                    \sin m \theta_{2} & \cos m \theta_{2} \newline
                                                    \end{pmatrix}
                                                    \begin{bmatrix}
                                                    x_{2} \newline
                                                    x'_{2} \newline
                                                    \end{bmatrix} & \cdots & \begin{pmatrix}
                                                                            0 \newline
                                                                            0
                                                                             \end{pmatrix}

                                    \newline

                                    \vdots & \vdots & \ddots & \vdots

                                    \newline

                                    \begin{pmatrix}
                                    0 \newline
                                    0
                                    \end{pmatrix} & \begin{pmatrix}
                                                    0 \newline
                                                    0
                                                    \end{pmatrix} & \cdots & \begin{pmatrix}
                                                                             \cos m \theta_{d / 2} & - \sin m \theta_{d / 2} \newline
                                                                             \sin m \theta_{d / 2} & \cos m \theta_{d / 2} \newline
                                                                             \end{pmatrix}
                                                                             \begin{bmatrix}
                                                                             x_{d / 2} \newline
                                                                             x'_{d / 2} \newline
                                                                             \end{bmatrix}

                                 \end{pmatrix}
$$
and we can see many zero-submatrix in it, and when computing we actually don't need to compute them, we just to compute the **diagonal part**, that will speed computation up. And we can also not storage the hole matrix with lots of zero-submatrix. That will free a lot of memory.

Now we will summarize above 