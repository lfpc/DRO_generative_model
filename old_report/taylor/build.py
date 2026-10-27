"""Assemble muon_shield_robustness_taylor.tex: the model-independent sections of the DeepONet report
(../muon_shield_robustness.tex: setup, training data, method, parameters, quadrature) and the Taylor
surrogate's own ones below. Run after editing either:

    python build.py && latexmk -pdf muon_shield_robustness_taylor.tex
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = '%' + '=' * 78 + '\n'


def shared_sections():
    """The sections of the DeepONet report that do not depend on the surrogate."""
    src = open(os.path.join(HERE, '..', 'muon_shield_robustness.tex')).read()

    def seg(a, b):
        i = src.index(a)
        return src[i:src.index(b, i)]

    setup = seg(RULE + '\\section{Setup}', RULE + '\\section{The surrogate}')
    i = setup.index('sensitive plane per $10^9$ muons. At the nominal design')
    j = setup.index('the surrogate gives', i)
    setup = (setup[:i] + 'sensitive plane per $10^9$ muons. At the nominal design, simulator runs on the muons of the\n'
             'analysis give $793$--$852$ (each $\\pm29$; the transport is stochastic) and independent muons\n'
             '$820\\pm14$, a hit rate of $8\\times10^{-7}$; ' + setup[j:setup.index('.', j)].replace('$810$', '$837$')
             + setup[setup.index('.', j):])
    train = seg('\\paragraph{Training data.}', '\\subsection{Validation against the simulator}').replace(
        'the members of a first ensemble, trained on the other six pools',
        'the members of a first (DeepONet) ensemble, trained on the other six\n        pools')
    quad = seg('\\section{Integrating the surrogate over the muons}', '\\begin{figure}[h]')
    quad = quad.replace(' (Fig.~\\ref{fig:quadrature})', '').replace(
        'The grid is converged to $10^{-5}$', 'For the DeepONet surrogate the grid is converged to $10^{-5}$')
    return dict(pre=src[:src.index('\\title{')], setup=setup, train=train,
                method=seg(RULE + '\\section{Method}', RULE + '\\section{Results}'),
                params=seg('\\section{Free parameters}', '\\section{Integrating the surrogate over the muons}'),
                quad=quad)


TITLE = r"""\title{Robustness of the SHiP muon shield to parameter tolerances\\[0.3ex]
\large with a Taylor surrogate of the per-muon hit probability}
\author{}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
How much do the muon hits on the SHiP sensitive plane change when the 43 free parameters of the
optimised muon shield deviate from their nominal values? Here the per-muon hit probability is
modelled by a Taylor surrogate: a second-order expansion in the design parameters around the
nominal design, whose gradient and Hessian are explicit coefficients, plus a residual network
that is switched off at the nominal design and takes over far from it. Every conclusion is
checked with the full simulation on $10^9$ muons. If every parameter may be off by up to
$\varepsilon$, the worst design found within these tolerances (searched on the surrogate, then
simulated) multiplies the hits by $1.67\pm0.07$ at $\varepsilon=1\%$, $3.6\pm0.1$ at $2\%$,
$8.9\pm0.3$ at $3\%$, $103\pm4$ at $5\%$ and $5090\pm170$ at $10\%$. A single parameter at its
$\pm10\%$ limit gives at most $\times2.3\pm0.1$ (the field of M7). Random independent deviations
are much milder: harmless at $\pm2\%$, mostly harmless at $\pm5\%$, with a heavy tail at
$\pm10\%$. Bounding the worst case takes sub-percent tolerances on a few parameters, mostly of
M7 and M2: the allocation found for at most $\times2$ (simulated: $\times1.64$) holds M2's entrance
width, M7's field and length and M7's inner yoke to $\pm0.5\%$. Near the nominal design every
harmful direction acts through muons of $30$--$75$\,GeV, while the muons above $175$\,GeV respond
the opposite way. The surrogate gets the directions and the worst cases right, and
underestimates the magnitudes along the harmful directions by $13$--$20\%$.
\end{abstract}

\tableofcontents

"""

MODEL = r"""%==============================================================================
\section{The surrogate}
%==============================================================================

\paragraph{Model.}
The surrogate models the hit probability of each muon $x$ for a design $u$ as
\begin{equation}
  p(\text{hit}\mid x,u) = \sigma\!\Big(c(x) + g(x)\cdot u + \tfrac12\textstyle\sum_{k=1}^{8} h_k(x)\,(q_k\cdot u)^2
  + \gamma(u)\, r(u)\cdot t(x) + c_0\Big),
\end{equation}
a per-muon Taylor expansion of the logit to second order in $u$: the constant $c(x)$, the full
gradient $g(x)\in\mathbb{R}^{43}$ and a curvature of rank 8 in a learned orthonormal basis
$q_1,\dots,q_8$ are outputs of one network of the muon ($p_x$, $p_y$, $p_z$, charge). The residual
$r(u)\cdot t(x)$ (32 features, a DeepONet) captures what a quadratic cannot, and is switched on
by the gate
\begin{equation}
  \gamma(u) = \frac{\langle u_i^4\rangle}{\rho^4},
\end{equation}
with $\langle\cdot\rangle$ the mean over the parameters and $\rho$ the RMS radius of the training
designs. The gate is zero with zero gradient and Hessian at $u=0$, so the derivatives at the
nominal design are exactly the Taylor coefficients; and it opens along a single axis (one
parameter at its limit: $\gamma=0.9$) as well as for spread-out changes, so that one-sided
single-parameter responses can be represented.

\paragraph{Training.}
The Taylor coefficients are fitted to the designs near the nominal one and the residual to all:
in the loss, each row's gradient reaches the Taylor part with the weight
$w(u)=\exp\!\big(-\langle u_i^4\rangle^{1/2}/0.3^2\big)$ and the residual with weight 1 (the
predictions are unchanged; only what each part learns from differs). Four such surrogates are
trained with different seeds on the same data as the DeepONet surrogate and their logits
averaged; the average is itself of the same form. A seed whose final validation loss is more
than $10\%$ above the median of its siblings is replaced by a new one (this happened once:
$0.0189$ against $0.0157$--$0.0162$); the rule uses only the held-out training designs. Errors on
surrogate quantities are the standard error of the four members.

"""

VALIDATION = r"""\subsection{Validation against the simulator}
\label{sec:validation}

Reference designs were simulated on stored muons, on which the surrogate also predicts: $10^9$
muons of the physical flux (the nominal design four times, 16 designs in the half box
$|u_i|\le\tfrac12$, 16 in the box), and a high-momentum check sample of $4\times10^8$ muons drawn
like the $T=1.5$ pools (the same designs plus every parameter alone at $\pm10\%$).

\begin{table}[t]
\centering
\caption{Surrogate vs.\ simulator on the reference designs: median $|\log(\text{pred}/\text{sim})|$
(signed for the nominal design), and the relative vector errors of the 43 one-at-a-time (OAT)
secants $\tfrac12[H(+e_i)-H(-e_i)]$ and curvatures $H(+e_i)+H(-e_i)-2H_0$ on the check sample.
``Noise'': what a perfect model would score, from the simulator's Poisson noise.}
\label{tab:val}
\begin{tabular}{lcc}
\toprule
 & noise & surrogate \\
\midrule
Nominal design                   & $1.7\%$  & $+1.3\%$ \\
Designs in the half box          & $2.1\%$  & $8.5\%$ \\
Designs in the box               & $1.6\%$  & $12\%$ \\
Check sample: nominal design     & $0.2\%$  & $-1.5\%$ \\
Check sample: designs in the box & $0.25\%$ & $6.0\%$ \\
OAT secants (cosine)             & $1.6\%$  & $6.4\%$ (0.999) \\
OAT curvatures (cosine)          & $7.6\%$  & $15\%$ (0.992) \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{refs_tr4n30r_oat.png}\\[1ex]
  \includegraphics[width=0.75\linewidth]{refs_tr4n30r_calibration.png}
  \caption{Top: every parameter alone at $\pm10\%$, secant $\tfrac12[H(+10\%)-H(-10\%)]$ on the
  check sample: simulator (squares) and surrogate (circles). Bottom: predicted vs.\ simulated hits
  of the reference designs.}
  \label{fig:val}
\end{figure}

On the nominal muons the surrogate predicts the nominal design to $1\%$ and the box designs to
$12\%$; the one-at-a-time effects on the check sample are reproduced to $6\%$ (Table~\ref{tab:val},
Fig.~\ref{fig:val}). Within one tolerance unit the walks of Sec.~\ref{sec:walks} agree in
direction, but along the harmful directions the surrogate is $13$--$20\%$ low (see there).

The derivatives are well determined: the four members' gradients at $u=0$ agree to
$\cos=0.90$--$0.95$, and their top Hessian eigenvectors to $\geq0.98$ with the ensemble's.

"""

RESULTS = r"""%==============================================================================
\section{Results}
%==============================================================================

\subsection{Single parameters}

31 of the 43 one-at-a-time effects are resolved beyond three times the members' scatter
(Fig.~\ref{fig:sensitivity}, Table~\ref{tab:oat}). The most sensitive parameter is the
\emph{field of M7}: lowering $|B|$ by $10\%$ raises the hits by $82\%$ (simulated: $\times2.3$).
Next come the exit width and the length of M7, the entrance width of M2 and the inner yoke of M7,
then the field of M2, the length of M4 and the inner yoke of M3. Most responses are one-sided.
Per magnet (Fig.~\ref{fig:magnets}), M7 carries by far the largest share of the gradient ($84\%$
of its norm), then M2 ($39\%$), M6 ($24\%$) and M4 ($19\%$).

\paragraph{Simulated.} The eight largest were simulated at their worse limit
(Table~\ref{tab:concrete}): all eight rise significantly. The surrogate underestimates the
largest ones: the field of M7 ($\times2.29\pm0.10$ simulated, $1.82$ predicted), the entrance
width of M2 ($2.02\pm0.09$ vs.\ $1.59$), the exit width of M7 ($1.83\pm0.08$ vs.\ $1.64$) and the
inner yoke of M3 ($1.49\pm0.07$ vs.\ $1.28$); the length of M7, the inner yoke of M7, the field of
M2 and the length of M4 agree within $1\sigma$. The scans of seven of these parameters were
simulated at $\pm5\%$ and $\pm10\%$ (Fig.~\ref{fig:scans}; the length of M4 was not): 20 of 28 points
agree within $2\sigma$. The misses are the narrowing side of the steepest responses (the field of
M7, the entrance width of M2, the exit width and inner yoke of M7, the inner yoke of M3), where the
simulated rise is steeper than the surrogate's: $2.22$, $1.93$, $1.91$, $1.25$, $1.44$ at $-10\%$
against $1.81$, $1.59$, $1.64$, $1.06$, $1.28$.

\begin{table}[t]
\centering
\caption{The most sensitive parameters (surrogate): first-order sensitivity $g_i/H_0$ per full
tolerance, and the exact change at $\pm10\%$ (errors: the members' standard error).}
\label{tab:oat}
\begin{tabular}{lrrrr}
\toprule
parameter & nominal & $g_i/H_0$ & $H/H_0-1$ at $+10\%$ & at $-10\%$ \\
\midrule
M7:$B$                 & $-1.9$ T & $-0.480$ & $-0.30\pm0.01$ & $+0.82\pm0.04$ \\
M7:\texttt{dXOut}      & 67.81 cm & $-0.340$ & $-0.19\pm0.01$ & $+0.64\pm0.05$ \\
M7:\texttt{Z\_len}     & 193.1 cm & $-0.393$ & $-0.26\pm0.02$ & $+0.62\pm0.05$ \\
M2:\texttt{dXIn}       & 68.04 cm & $-0.252$ & $+0.05\pm0.01$ & $+0.59\pm0.02$ \\
M7:\texttt{dXYokeIn}   & 124.8 cm & $+0.223$ & $+0.56\pm0.02$ & $+0.06\pm0.04$ \\
M2:$B$                 & 1.9 T    & $-0.224$ & $-0.13\pm0.02$ & $+0.34\pm0.02$ \\
M4:\texttt{Z\_len}     & 224.0 cm & $-0.138$ & $-0.04\pm0.03$ & $+0.28\pm0.04$ \\
M3:\texttt{dXYokeIn}   & 85.88 cm & $-0.094$ & $+0.10\pm0.03$ & $+0.28\pm0.02$ \\
M6:\texttt{Z\_len}     & 168.0 cm & $-0.173$ & $-0.12\pm0.02$ & $+0.23\pm0.03$ \\
M7:\texttt{dXYokeOut}  & 94.92 cm & $+0.079$ & $+0.22\pm0.04$ & $+0.10\pm0.04$ \\
M3:\texttt{dXIn}       & 64.81 cm & $+0.111$ & $+0.21\pm0.04$ & $-0.03\pm0.02$ \\
M3:\texttt{Z\_len}     & 225.0 cm & $+0.029$ & $+0.20\pm0.02$ & $+0.16\pm0.02$ \\
M3:$B$                 & 1.9 T    & $-0.015$ & $+0.17\pm0.03$ & $+0.19\pm0.03$ \\
M2:\texttt{Z\_len}     & 224.9 cm & $+0.006$ & $+0.17\pm0.02$ & $+0.18\pm0.01$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{robustness_sensitivity.png}
  \caption{Sensitivity of each parameter. Left: gradient per full tolerance. Right: exact
  change with the parameter at $+10\%$ (orange) and $-10\%$ (blue).}
  \label{fig:sensitivity}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{robustness_magnets.png}
  \caption{Per magnet: share of the gradient norm and largest single-parameter effect.}
  \label{fig:magnets}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{robustness_scans.png}
  \caption{One-at-a-time scans of the eight most sensitive parameters: surrogate (blue) and
  simulator at $\pm5\%$ and $\pm10\%$ (orange; the length of M4 was not simulated).}
  \label{fig:scans}
\end{figure}

\subsection{Curvature and active subspace}

Two Hessian eigenvalues stand out (Fig.~\ref{fig:spectra}): $2.39\pm0.04\,H_0$ per unit$^2$ and
$1.40\,H_0$, then $0.58$ and $\lesssim0.25\,H_0$; 8 are negative, none below $-0.04\,H_0$. Along the
gradient the second-order term is $0.72\,H_0$ at one tolerance unit against $0.89\,H_0$ for the
first-order one: the response is strongly non-linear at that scale.

The active subspace is effectively one-dimensional ($99.3\%$ of $\mathbb{E}[\|\nabla_u p\|^2]$).
Its direction \texttt{act1} shortens M1--M4 and lowers their fields together, i.e.\ it changes the
upstream bending power (Table~\ref{tab:dirs}), and it is orthogonal to the gradient
($\cos=0.02$): individual muons are deflected most along it, but the changes cancel in the total.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{robustness_spectra.png}
  \caption{Left: Hessian eigenvalues in units of $H_0$. Right: active-subspace eigenvalues
  (blue), their cumulative share (red) and the fraction of the gradient they contain (grey).}
  \label{fig:spectra}
\end{figure}

\subsection{Worst directions}
\label{sec:walks}

\begin{table}[t]
\centering
\caption{Hits one tolerance unit along each direction, surrogate and simulator (nominal: $808$
simulated hits). ``max / RMS'': the largest and the RMS change over the 43 parameters.}
\label{tab:walks}
\begin{tabular}{lcccrr}
\toprule
direction & max / RMS change & surrogate $\HH$ & simulator $\HH$ & sim.\ hits & at $t=-1$ (sim.) \\
\midrule
\texttt{worst}  & 4.5\% / 1.5\% & $4.54\pm0.06$ & $5.20\pm0.20$ & 4204 & $1.05\pm0.05$ \\
\texttt{hess1}  & 3.7\% / 1.5\% & $4.35\pm0.13$ & $5.03\pm0.19$ & 4064 & $1.33\pm0.06$ \\
\texttt{ascent} & 5.4\% / 1.5\% & $3.34\pm0.08$ & $4.09\pm0.16$ & 3304 & $0.68\pm0.04$ \\
\texttt{act1}   & 4.0\% / 1.5\% & $2.50\pm0.06$ & $2.47\pm0.10$ & 1993 & $1.60\pm0.07$ \\
\texttt{corner} & 1.5\% / 1.5\% & $2.08\pm0.05$ & $2.24\pm0.09$ & 1808 & $0.80\pm0.04$ \\
\texttt{act2}   & 8.5\% / 1.5\% & $1.42\pm0.01$ & $1.67\pm0.07$ & 1352 & $0.87\pm0.04$ \\
\texttt{random} & ---  / 1.5\% & $0.90\pm0.03$ & $0.93\pm0.05$ &  753 & $1.09\pm0.05$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Small coordinated changes are far more harmful than a large single one.}
Moving every parameter by at most $4.5\%$ (RMS $1.5\%$) along \texttt{worst} multiplies the hits
by $5.2\pm0.2$ (Table~\ref{tab:walks}). The simplest such pattern, \texttt{corner}, moves
\emph{every} parameter by exactly $1.5\%$ with the sign of its gradient and doubles the hits. The
worst single parameter needs a $10\%$ error for a similar effect. A random direction of the same
size changes nothing ($0.93\pm0.05$): the damage comes from the pattern of signs, not the size of
the step.

\paragraph{The surrogate gets the directions; it underestimates the sizes.}
Its two best directions (\texttt{worst} and \texttt{hess1}) are the two the simulator ranks
highest, then \texttt{ascent}. Along these harmful directions the surrogate is $13$--$20\%$ low at
$t=1$ ($3$--$4$ combined standard deviations), and on the other side ($t=-1$) it is within $2\sigma$
except for \texttt{ascent} ($0.56$ vs.\ $0.68$); the active direction and the random one agree
(Fig.~\ref{fig:calibration}).

\paragraph{The quadratic model is not enough.}
The Taylor part alone, built from the gradient and Hessian, gives $2.9$ instead of $5.2$ along
\texttt{worst} (Fig.~\ref{fig:walks}): the hits grow roughly exponentially, which the residual
partly captures. The derivatives give the directions; magnitudes come from the simulator.

\paragraph{The active subspace is a direction of curvature, not of slope.}
\texttt{act1} gives $\times2.47$ at $t=+1$ and $\times1.60$ at $t=-1$: shortening or lengthening
magnets 1--4 together, with their fields, is harmful either way, and the nominal design sits at
the balance point (Sec.~\ref{sec:momentum}). In the plane of \texttt{worst} and \texttt{act1}
(Fig.~\ref{fig:plane}) the nominal design lies in a shallow valley.

\begin{table}[t]
\centering
\caption{What each direction changes at one tolerance unit: its largest parameter changes,
relative and in physical units (cm, or T for $B$).}
\label{tab:dirs}
\small
\begin{tabular}{l p{0.55\linewidth} p{0.25\linewidth}}
\toprule
direction & largest changes & in words \\
\midrule
\texttt{worst}  & M7:$B$ $-4.5\%$ ($|B|$ $-0.09$ T), M7:\texttt{Z\_len} $-3.7\%$ ($-7.2$),
                  M2:\texttt{dXIn} $-3.7\%$ ($-2.5$), M7:\texttt{dXYokeIn} $+3.1\%$ ($+3.9$),
                  M7:\texttt{dXOut} $-2.7\%$ ($-1.8$), M3:\texttt{dXYokeIn} $-2.2\%$ ($-1.9$),
                  M3:$B$ $+2.2\%$ ($+0.04$ T)
                & weaker and shorter M7 with wider yokes, narrower M2 entrance \\
\midrule
\texttt{hess1}  & M2:\texttt{dXIn} $-3.7\%$ ($-2.5$), M7:$B$ $-3.6\%$ ($|B|$ $-0.07$ T),
                  M3:$B$ $+3.1\%$ ($+0.06$ T), M7:\texttt{dXYokeIn} $+3.1\%$ ($+3.8$),
                  M7:\texttt{Z\_len} $-3.0\%$ ($-5.8$), M2:\texttt{Z\_len} $+2.8\%$ ($+6.3$),
                  M3:\texttt{dXYokeIn} $-2.5\%$ ($-2.2$)
                & as \texttt{worst} \\
\midrule
\texttt{ascent} & M7:$B$ $-5.4\%$ ($|B|$ $-0.10$ T), M7:\texttt{Z\_len} $-4.4\%$ ($-8.5$),
                  M7:\texttt{dXOut} $-3.8\%$ ($-2.6$), M2:\texttt{dXIn} $-2.8\%$ ($-1.9$),
                  M2:$B$ $-2.5\%$ ($-0.05$ T), M7:\texttt{dXYokeIn} $+2.5\%$ ($+3.1$),
                  M6:\texttt{Z\_len} $-1.9\%$ ($-3.3$)
                & the gradient: M7 weaker, shorter and narrower \\
\midrule
\texttt{corner} & all 43 parameters by exactly $\pm1.5\%$, each with the sign of its gradient
                & the simplest uniform pattern \\
\midrule
\texttt{act1}   & M2, M3, M4:\texttt{Z\_len} $-3.8$ to $-4.0\%$ ($-8.5$ to $-8.9$), M2:$B$ $-3.4\%$,
                  M3:$B$ $-3.2\%$, M4:$B$ $-2.8\%$ ($-0.05$ to $-0.07$ T), M1:\texttt{Z\_len}
                  $-2.6\%$ ($-3.9$)
                & magnets 1--4 shorter and weaker: less bending upstream \\
\midrule
\texttt{act2}   & M7:\texttt{dXOut} $-8.5\%$ ($-5.7$), M6:\texttt{Z\_len} $-2.0\%$ ($-3.3$), the
                  rest below $1.6\%$
                & the exit width of M7 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{robustness_walks.png}
  \caption{Walks along each direction: surrogate (blue; error bars: the members), simulator
  (orange) and the quadratic model (dotted).}
  \label{fig:walks}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.48\linewidth]{robustness_summary.png}\hfill
  \includegraphics[width=0.42\linewidth]{robustness_calibration.png}
  \caption{Left: $\HH$ at one tolerance unit along each direction. Right: surrogate vs.\
  simulator for all 33 walked designs; the outliers are the box designs of Sec.~\ref{sec:box}.}
  \label{fig:calibration}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.6\linewidth]{robustness_plane.png}
  \caption{The surrogate in the plane of \texttt{worst} ($x$) and the part of \texttt{act1}
  orthogonal to it ($y$); markers: simulated designs with their $\HH$.}
  \label{fig:plane}
\end{figure}

\subsection{Concrete designs along the worst direction}
\label{sec:concrete}

Table~\ref{tab:concrete} lists concrete designs, all simulated. ``Worst, max $\varepsilon$'' is
the worst direction scaled so that no parameter moves by more than $\varepsilon$. It lies within
the $\pm\varepsilon$ tolerances, so it bounds their worst case from below; most of its parameters
stay well inside them, and the worst design within the tolerances is much worse
(Sec.~\ref{sec:tolerance}).

\begin{table}[t]
\centering
\caption{Concrete designs, simulated (nominal $793\pm28$ hits) and predicted. At
$\varepsilon=10\%$ the worst direction changes $|B|$ of M7 from $1.9$ to $1.71$ T, M7:\texttt{Z\_len}
by $-15.9$ cm, M2:\texttt{dXIn} by $-5.6$ cm, M7:\texttt{dXYokeIn} by $+8.6$ cm, M7:\texttt{dXOut} by
$-4.0$ cm, M3:\texttt{dXYokeIn} by $-4.2$ cm, M3:$B$ by $+0.09$ T, M2:\texttt{Z\_len} by $+9.8$ cm,
M7:\texttt{dXYokeOut} by $+3.6$ cm, M4:$B$ by $+0.07$ T, and 33 more parameters by less than $4\%$.}
\label{tab:concrete}
\small
\begin{tabular}{llrcc}
\toprule
design & what changes & hits & sim.\ $\HH$ & surr.\ $\HH$ \\
\midrule
worst, max 1\%   & all 43, RMS 0.33\% & 1032  & $1.30\pm0.06$ & 1.23 \\
worst, max 2\%   & all 43, RMS 0.67\% & 1462  & $1.84\pm0.08$ & 1.64 \\
worst, max 3\%   & all 43, RMS 1.0\% & 2211  & $2.79\pm0.12$ & 2.33 \\
worst, max 5\%   & all 43, RMS 1.7\% & 5407  & $6.82\pm0.26$ & 5.65 \\
worst, max 10\%  & all 43, RMS 3.3\% & 44871 & $56.6\pm2.0$  & 104 \\
opposite, max 5\% & all 43, signs reversed & 917 & $1.16\pm0.06$ & 1.09 \\
opposite, max 10\% & all 43, signs reversed & 3408 & $4.30\pm0.17$ & 4.73 \\
\midrule
worst, top 1 only  & M7:$B$ $-10\%$ ($|B|$ $1.9\to1.71$ T) & 1716 & $2.16\pm0.09$ & 1.82 \\
worst, top 3 only  & + M7:\texttt{Z\_len}, M2:\texttt{dXIn} $-8.2\%$ & 5055 & $6.38\pm0.24$ & 5.15 \\
worst, top 5 only  & + M7 inner yoke $+6.9\%$, exit $-5.9\%$ & 10495 & $13.2\pm0.5$ & 11.2 \\
worst, top 10 only & the ten largest changes & 23066 & $29.1\pm1.1$ & 46.3 \\
\midrule
single & M7:$B$ $-10\%$ ($|B|$ $1.9\to1.71$ T) & 1815 & $2.29\pm0.10$ & 1.82 \\
single & M7:\texttt{dXOut} $-10\%$ ($-6.8$ cm)  & 1452 & $1.83\pm0.08$ & 1.64 \\
single & M7:\texttt{Z\_len} $-10\%$ ($-19.3$ cm) & 1356 & $1.71\pm0.08$ & 1.62 \\
single & M2:\texttt{dXIn} $-10\%$ ($-6.8$ cm)   & 1600 & $2.02\pm0.09$ & 1.59 \\
single & M7:\texttt{dXYokeIn} $+10\%$ ($+12.5$ cm) & 1284 & $1.62\pm0.07$ & 1.56 \\
single & M2:$B$ $-10\%$ ($1.9\to1.71$ T) & 1052 & $1.33\pm0.06$ & 1.34 \\
single & M4:\texttt{Z\_len} $-10\%$ ($-22.4$ cm) & 1036 & $1.31\pm0.06$ & 1.28 \\
single & M3:\texttt{dXYokeIn} $-10\%$ ($-8.6$ cm) & 1181 & $1.49\pm0.07$ & 1.28 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Exponential growth along the direction.}
Along the worst direction (Fig.~\ref{fig:scaled}) the hits grow by $\times1.3$ at a maximum change
of $1\%$, $\times1.8$ at $2\%$, $\times2.8$ at $3\%$, $\times6.8$ at $5\%$ and $\times57$ at $10\%$:
a factor $e$ per $\sim2.5\%$ of tolerance. The surrogate is $5$--$17\%$ low up to $5\%$ and nearly
twice too high at $10\%$, beyond its reliable range. The opposite direction is harmless at $5\%$
($\times1.2$) but not at $10\%$ ($\times4.3$), through other muons (Sec.~\ref{sec:momentum}).

\paragraph{The rise is collective.}
The largest component alone (the field of M7 lowered by $10\%$) gives $\times2.2$; the top three
(adding a shorter M7 and a narrower M2 entrance) $\times6.4$; the top five $\times13$; the top ten
$\times29$; all 43 $\times57$ (Fig.~\ref{fig:attribution}). Each parameter contributes a moderate
factor, and the factors multiply.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.7\linewidth]{worst_direction_scaled.png}
  \caption{Hits along the worst direction against the largest change of any parameter:
  simulator and surrogate.}
  \label{fig:scaled}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{worst_parameters.png}
  \caption{The worst direction at a maximum change of $10\%$, truncated to its largest 1, 3, 5,
  10 changes, and single parameters at their worse limit.}
  \label{fig:attribution}
\end{figure}

\FloatBarrier
\subsection{The worst case for a given tolerance}
\label{sec:tolerance}
\label{sec:box}

\paragraph{Method.} If every parameter may be off by up to $\varepsilon$, the designs allowed
form the box $|u_i|\le\varepsilon/10\%$. Its worst design is found on the surrogate, by
projected-gradient ascent of $\log H$ over the box from 16 starting points (the sign of the
gradient, the scaled worst direction and random corners; 300 Adam steps each, batched), and then
simulated on the $10^9$ muons of the walks. The simulated values are real designs within the
tolerances, so the true worst case is at least as bad; the search is as good as the surrogate.
The worst direction of Sec.~\ref{sec:walks} maximises the hits on a sphere ($\|u\|=1$); on a box
the maximum is at a corner, every parameter at a limit with the harmful sign, so scaling the
direction down underestimates it.

\begin{table}[!htb]
\centering
\caption{The worst design within $\pm\varepsilon$ on every parameter: found on the surrogate,
simulated ($10^9$ muons; nominal $847\pm29$). Last column: the worst direction scaled to the same
$\varepsilon$ (Table~\ref{tab:concrete}).}
\label{tab:tolerance}
\begin{tabular}{lrrccc}
\toprule
$\varepsilon$ & at a limit & sim.\ hits & sim.\ $\HH$ & surr.\ $\HH$ & scaled direction (sim.) \\
\midrule
$1\%$  & 43 & 1413      & $1.67\pm0.07$ & $1.64\pm0.02$ & $1.30\pm0.06$ \\
$2\%$  & 43 & 3062      & $3.62\pm0.14$ & $3.65\pm0.07$ & $1.84\pm0.08$ \\
$3\%$  & 43 & 7550      & $8.9\pm0.3$   & $10.5\pm0.5$  & $2.79\pm0.12$ \\
$5\%$  & 37 & 86\,878   & $103\pm4$     & $240\pm60$    & $6.82\pm0.26$ \\
$10\%$ & 37 & 4\,311\,598 & $5090\pm170$ & $(5.2\pm0.3)\times10^5$ & $56.6\pm2.0$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htb]
  \centering
  \includegraphics[width=0.75\linewidth]{tolerance_boxes.png}
  \caption{Worst case for a given tolerance: the worst design within $\pm\varepsilon$ (filled;
  surrogate and simulator) and the worst direction scaled to $\varepsilon$ (open).}
  \label{fig:tolerance}
\end{figure}

\paragraph{Results} (Table~\ref{tab:tolerance}, Fig.~\ref{fig:tolerance}). Already at $\pm1\%$ the
worst design gives $\times1.7$, at $\pm2\%$ $\times3.6$ and at $\pm3\%$ $\times8.9$, $1.3$ to three times
the scaled direction. Up to $\pm3\%$ these designs act through the mechanism of the worst
direction (M2's entrance narrower, M7 weaker and shorter, M7's inner yoke wider, M3's inner yoke
narrower), with every parameter at its limit ($\cos=0.65$ with the worst direction), and the
surrogate predicts them to $1$--$18\%$. At $\pm5\%$ and $\pm10\%$ the search finds another
mechanism, M7's yokes narrower with M2's entrance and M3's inner yoke wider ($\cos\approx-0.1$
with the worst direction), which the local derivatives do not show. There the surrogate
overestimates by $\times2.4$ and $\times100$, but the designs it finds are real: $\times103$ at
$\pm5\%$ and $\times5090$ at $\pm10\%$. The cruder box search of the walks (\texttt{pga}) ends at a
corner with $\times1820\pm60$, almost all from muons of $40$--$60$\,GeV; halfway to it the simulator
gives $\times55\pm2$, and its opposite corner $\times54\pm2$.

\subsection{Tolerance allocation}
\label{sec:allocation}

\paragraph{Method.} Which tolerances keep the worst case below a target? Greedily: start with
every parameter at $\pm10\%$; find the box's worst design $u^*$ as above; tighten the two
parameters that contribute most to it (the largest first-order shares of $\log H$,
$u^*_i\,\partial\log H/\partial u_i$) by a factor $0.8$, down to a floor of $\pm0.5\%$; repeat
until the worst case is below the target. The first box that meets each target is its
allocation, and its worst design is simulated. This is a heuristic, not an optimum.

\begin{table}[!htb]
\centering
\caption{Tolerance allocations (greedy, surrogate) for a worst case of at most $3$, $2$ and $1.5$
times the nominal hits: the sixteen tightest parameters ($\pm\%$), and a summary of all 43. The
worst design of each allocation is simulated in the last row.}
\label{tab:allocation}
\small
\begin{tabular}{lrccc}
\toprule
parameter & nominal & $\le3H_0$ & $\le2H_0$ & $\le1.5H_0$ \\
\midrule
M2:\texttt{dXIn}         & 68.04 cm & 0.55 & 0.50 & 0.50 \\
M7:$B$                   & 1.9 T    & 0.55 & 0.50 & 0.50 \\
M7:\texttt{Z\_len}       & 193.1 cm & 0.69 & 0.50 & 0.50 \\
M7:\texttt{dXYokeIn}     & 124.8 cm & 0.69 & 0.50 & 0.50 \\
M3:\texttt{dXYokeIn}     & 85.88 cm & 0.86 & 0.55 & 0.50 \\
M7:\texttt{dXOut}        & 67.81 cm & 1.07 & 0.55 & 0.50 \\
M7:\texttt{dXYokeOut}    & 94.92 cm & 1.07 & 0.69 & 0.50 \\
M6:\texttt{Z\_len}       & 168.0 cm & 1.34 & 0.86 & 0.50 \\
M3:$B$                   & 1.9 T    & 0.86 & 0.69 & 0.55 \\
M4:$B$                   & 1.9 T    & 1.34 & 0.86 & 0.69 \\
M3:\texttt{dXIn}         & 64.81 cm & 1.68 & 0.86 & 0.69 \\
M2:\texttt{Z\_len}       & 224.9 cm & 1.34 & 1.07 & 0.69 \\
M4:\texttt{dXIn}         & 39.02 cm & 1.68 & 1.07 & 0.69 \\
M2:\texttt{dXYokeIn}     & 70.62 cm & 1.68 & 1.07 & 0.69 \\
M2:$B$                   & 1.9 T    & 3.28 & 1.34 & 0.69 \\
M1:\texttt{Z\_len}       & 151.5 cm & 1.68 & 1.34 & 0.86 \\
\midrule
parameters below $\pm1\%$ / $\pm2\%$ / $\pm5\%$ & & 6 / 16 / 26 & 11 / 19 / 33 & 17 / 26 / 43 \\
median tolerance & & $3.3\%$ & $2.1\%$ & $1.3\%$ \\
\midrule
worst design: surrogate & & $2.97\pm0.12$ & $1.99\pm0.08$ & $1.50\pm0.02$ \\
worst design: simulator & & $2.25\pm0.09$ & $1.64\pm0.07$ & $1.38\pm0.06$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{tolerance_allocation.png}
  \caption{Left: the tolerance allocations for a worst case of at most $3$, $2$, $1.5$ and
  $1.25\,H_0$ (surrogate; simulated worst designs in the legend). Right: the greedy path, the
  worst case of the box against the number of parameters tightened.}
  \label{fig:allocation}
\end{figure}

\paragraph{Results} (Table~\ref{tab:allocation}, Fig.~\ref{fig:allocation}). A guarantee against
the worst case is demanding. For at most $\times3$, the allocation holds M2's entrance width and
M7's field to $\pm0.55\%$ ($0.37$ cm and $0.01$ T), M7's length and inner yoke to $\pm0.7\%$, six
parameters below $\pm1\%$ and 16 below $\pm2\%$, with a median tolerance of $3.3\%$. For at most
$\times2$ it puts eleven parameters below $\pm1\%$, for $\times1.5$ seventeen, and for $\times1.25$
32 (27 at the $0.5\%$ floor, simulated $\times1.20\pm0.06$); with every parameter at $\pm0.5\%$ the
worst case is $\times1.25$. The order follows the sensitivities: M2's entrance, M7 (field, length,
yokes, exit width) and M3's inner yoke first, the gaps between magnets last. The simulated worst
designs stay below every target ($\times2.25$, $1.64$, $1.38$, $1.20$): at these small tolerances
the surrogate is conservative.

\subsection{Random tolerances}

Designs with every parameter independently uniform in $\pm2\%$, $\pm5\%$ and $\pm10\%$: eight of
each simulated, and 4096 per level from the surrogate (Fig.~\ref{fig:random}).
\begin{center}
\begin{tabular}{lcccc}
\toprule
tolerance & simulated $\HH$ (8 designs) & surr.\ median & surr.\ 5--95\% & $P(H>H_0)$ \\
\midrule
$\pm2\%$  & 0.85, 0.89, 0.97, 0.98, 1.01, 1.07, 1.11, 1.20 & 1.03 & 0.87--1.23 & $60\%$ \\
$\pm5\%$  & 0.92, 1.01, 1.09, 1.15, 1.33, 1.45, 1.59, 4.26 & 1.20 & 0.77--2.01 & $74\%$ \\
$\pm10\%$ & 0.70, 1.01, 1.12, 1.35, 1.99, 3.52, 5.35, 7.99 & 2.03 & 0.79--10.1 & $87\%$ \\
\bottomrule
\end{tabular}
\end{center}
At $\pm2\%$ the hits stay within about $\pm20\%$. At $\pm5\%$ most designs stay within $\pm60\%$,
but one of the eight simulated quadruples the hits. At $\pm10\%$ the median design is twice as bad
and the tail reaches $\times8$ among the simulated designs ($\times34$ at the surrogate's 99th
percentile). The surrogate agrees with the simulator design by design (median ratio
$1.04$--$1.12$).

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{random_tolerances.png}
  \caption{Random independent tolerances: surrogate distribution of 4096 designs per level
  (histogram), and the eight simulated designs (solid: simulator, dotted: surrogate).}
  \label{fig:random}
\end{figure}

\subsection{Which muons respond}
\label{sec:momentum}

The muons are split into bins of $p_z$ (and separately $p_t$ and charge), and each bin's hits
$H_b$ get their own gradient, one-at-a-time effects and worst direction (Fig.~\ref{fig:pz_overview});
the simulated hits per $p_z$ range (Fig.~\ref{fig:simpz}) check them.

\paragraph{Three populations.}
Half of the muons have $p_z<30$\,GeV, but at the nominal design the hits come from two groups
(Table~\ref{tab:pz}; simulated shares in parentheses): \emph{intermediate} momenta,
$30$--$75$\,GeV, with $67\%$ of the hits ($65\%$; $42\%$ at $40$--$50$\,GeV alone) at a hit rate
of $10^{-6}$; and \emph{high} momenta above $125$\,GeV, $1.2\%$ of the muons but $25\%$ of the hits
($24\%$ above $150$\,GeV), with hit rates up to $10^{-3}$ above $300$\,GeV. A third group,
\emph{soft} muons of $20$--$30$\,GeV, gives only $4\%$ of the hits but responds strongly to some
changes.

\begin{table}[t]
\centering
\caption{Per $p_z$ bin at the nominal design (surrogate): share of the muons and of the hits, hit
rate, relative sensitivity $|\nabla H_b|/H_b$ per tolerance unit, and the cosine between the
bin's gradient and that of the total.}
\label{tab:pz}
\begin{tabular}{lrrrrr}
\toprule
$p_z$ [GeV] & muons & share of hits & hit rate & $|\nabla H_b|/H_b$ & $\cos(g_b, g)$ \\
\midrule
10--20   & $1.7\%$   & $0.0\%$  & $4.3\times10^{-9}$ & 4.05 & $-0.37$ \\
20--30   & $48.4\%$  & $4.4\%$  & $7.5\times10^{-8}$ & 2.01 & $-0.12$ \\
30--45   & $28.8\%$  & $45.6\%$ & $1.3\times10^{-6}$ & 1.38 & $+0.90$ \\
45--75   & $15.3\%$  & $21.4\%$ & $1.2\times10^{-6}$ & 1.73 & $+0.83$ \\
75--125  & $4.7\%$   & $4.0\%$  & $7.3\times10^{-7}$ & 0.77 & $+0.64$ \\
125--175 & $0.89\%$  & $2.6\%$  & $2.4\times10^{-6}$ & 0.62 & $+0.36$ \\
175--225 & $0.23\%$  & $3.0\%$  & $1.1\times10^{-5}$ & 1.44 & $+0.05$ \\
225--300 & $0.08\%$  & $9.9\%$  & $1.1\times10^{-4}$ & 1.32 & $+0.03$ \\
$>300$   & $0.006\%$ & $9.2\%$  & $1.2\times10^{-3}$ & 2.90 & $-0.01$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{The populations want opposite changes.}
The worst directions of the $30$--$45$ and $45$--$75$\,GeV bins agree ($\cos=0.92$), as do all
bins above $175$\,GeV ($0.99$--$1.00$), but the two groups are anti-aligned ($\cos=-0.39$ to
$-0.52$), and the soft bins are anti-aligned with the intermediate ones too (down to $-0.67$,
Fig.~\ref{fig:pz_alignment}). The total gradient, and every harmful direction of
Table~\ref{tab:walks}, follows the intermediate group ($\cos=0.83$--$0.90$). Simulated
(Fig.~\ref{fig:simpz}):
\begin{itemize}
  \item worst direction at $10\%$: the $40$--$50$\,GeV hits go from $332$ to $33\,443$ and the
        $50$--$60$\,GeV ones from $55$ to $9553$, while those above $150$\,GeV \emph{fall} from
        $194$ to $49$ and those below $30$\,GeV from $43$ to $9$;
  \item opposite direction: the $40$--$50$\,GeV hits fall from $332$ to $84$, but those of
        $20$--$30$\,GeV rise from $43$ to $1437$ and those above $150$\,GeV from $194$ to $1334$;
  \item the box corner of the walks: only $30$--$75$\,GeV ($516\to1\,442\,724$), the
        high-momentum hits fall ($194\to56$);
  \item random $\pm10\%$ designs can open any group: one raises the $40$--$50$\,GeV hits from $332$
        to $3046$, another those above $150$\,GeV from $194$ to $1066$, and the worst of the eight
        ($\times8.0$) both the soft ($43\to2054$) and the high-momentum ($194\to3319$) ones.
\end{itemize}

\paragraph{Which parameters act on which muons} (Fig.~\ref{fig:pz_sensitivity}).
The high-momentum hits are controlled by the \emph{upstream} magnets: the lengths of M1--M3 and
the field of M3, each changing the hits above $175$\,GeV by $40$--$200\%$ at its worse limit. The
intermediate hits are controlled by the entrance width of M2, the yokes of M3 and M7, and above
all the \emph{downstream} magnet M7. The soft hits respond most to the entrance widths of M2 and
M3 and the yoke of M3 (up to $\times4$). This explains \texttt{act1}: changing M1--M4 together shifts
hits between the high-momentum group and the others, and the nominal design balances them.

\paragraph{Transverse momentum and charge.}
All $p_t$ bins below $1.5$\,GeV behave alike (worst directions at $\cos=0.92$--$1.00$). The muons
above $1.5$\,GeV ($1.6\%$ of the muons, $22\%$ of the hits) are less sensitive ($0.78$ vs.\
$1.0$ per tolerance unit) and only partly aligned ($\cos=0.45$--$0.58$, Fig.~\ref{fig:pt}).
Positive muons give $64\%$ of the hits; both charges respond to the same changes ($\cos=0.96$),
the $\mu^-$ somewhat more strongly ($1.00$ vs.\ $0.84$).

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{sim_pz_profiles.png}
  \caption{Simulated hits per $p_z$ range for the nominal design and some designs of
  Table~\ref{tab:concrete}.}
  \label{fig:simpz}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{pz_bins_overview.png}
  \caption{Per $p_z$ bin (surrogate): share of the muons and of the hits, relative sensitivity
  and curvature along the bin's own worst direction.}
  \label{fig:pz_overview}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.62\linewidth]{pz_bins_alignment.png}
  \caption{Cosine between the worst directions of the $p_z$ bins.}
  \label{fig:pz_alignment}
\end{figure}

\begin{figure}[p]
  \centering
  \includegraphics[width=\linewidth]{pz_bins_sensitivity.png}
  \caption{Sensitivity of each $p_z$ bin to each parameter, relative to the bin's own hits.
  Left: first order. Right: exact one-at-a-time change at the worse of $\pm10\%$.}
  \label{fig:pz_sensitivity}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.49\linewidth]{pt_bins_overview.png}\hfill
  \includegraphics[width=0.42\linewidth]{pt_bins_alignment.png}
  \caption{Per $p_t$ bin: overview (left) and alignment of the bins' worst directions (right).}
  \label{fig:pt}
\end{figure}

\clearpage
%==============================================================================
\section{Conclusions}
%==============================================================================

\begin{enumerate}
  \item \textbf{The design is fragile to coordinated deviations.} With every parameter off by up
        to $\varepsilon$, the worst design found (on the surrogate, then simulated) gives
        $\times1.7$ at $\varepsilon=1\%$, $\times3.6$ at $2\%$, $\times8.9$ at $3\%$, $\times103$ at
        $5\%$ and $\times5090$ at $10\%$; these are lower bounds on the true worst case. Moving every
        parameter by $1.5\%$ with the sign of its gradient already doubles the hits.
  \item \textbf{Bounding the worst case needs sub-percent tolerances on a few parameters and a
        few percent on most.} The allocation found for at most $\times3$ holds M2's entrance width
        and M7's field to $\pm0.55\%$ and M7's length and inner yoke to $\pm0.7\%$, 16 parameters
        below $\pm2\%$, median $3.3\%$; for at most $\times2$, eleven below $\pm1\%$ (simulated
        $\times2.25$ and $\times1.64$). M7, M2 and M3 need the tightest tolerances, the gaps between
        magnets the loosest.
  \item \textbf{Single parameters underestimate the risk.} The worst one at its $\pm10\%$ limit
        (the field of M7) gives $\times2.3$; the entrance width of M2 and the exit width and length
        of M7 give $\times1.7$--$2.0$. The top three changes of the worst direction together give
        $\times6.4$, the top ten $\times29$: the factors multiply.
  \item \textbf{Random deviations are much milder than the worst case.} Independent errors of
        $\pm2\%$ keep the hits within about $\pm20\%$; at $\pm5\%$ most designs stay within
        $\pm60\%$ but some quadruple the hits; at $\pm10\%$ the median design is twice as bad and
        the tail reaches $\times8$--$34$.
  \item \textbf{Different muons pull in opposite directions.} The hits come from muons of
        $30$--$75$\,GeV ($65\%$) and above $150$\,GeV ($24\%$). Every harmful direction near the
        nominal design acts through the first group, while the high-momentum hits fall; the
        reverse direction raises the soft and high-momentum hits instead. The dominant
        active-subspace direction (the lengths and fields of M1--M4) trades hits between the
        groups and is harmful on both sides.
  \item \textbf{The Taylor surrogate gets the directions and the worst cases, not all the sizes.}
        Its worst directions are confirmed by the simulator ($\times5.2$ at one tolerance unit), and
        its worst designs within $\pm1$--$3\%$ are predicted to $1$--$18\%$, but along the harmful
        directions it is $13$--$20\%$ low and it underestimates the steepest single-parameter
        responses; far from the nominal design it overestimates. Every number used for a decision
        here is simulated.
\end{enumerate}

\section*{Limitations}
\begin{itemize}
  \item Uniform fields per magnet block, no field maps.
  \item Muons come from the generative flow, so the absolute numbers depend on how well the flow
        reproduces the muon spectrum.
  \item The surrogate's errors are the scatter of its members: they miss a bias the members
        share, such as the underestimate along the harmful directions (Sec.~\ref{sec:walks}).
  \item The random-tolerance distributions rest on eight simulated designs per level and on the
        surrogate for their tails.
  \item The worst designs within a tolerance are found on the surrogate: their simulated values
        bound the true worst case from below, possibly loosely at $\pm5\%$ and $\pm10\%$, where the
        surrogate is unreliable. The tolerance allocation is a greedy heuristic, not an optimum.
\end{itemize}

\clearpage
\appendix
"""

APP_TAYLOR = r"""
\section{The Taylor surrogate on problems with a known answer}
\label{app:toys}

On four toy problems with a binary outcome per sample and known probabilities (10--43 parameters),
the exact gradient and Hessian at $u=0$ are available. Trained on increasing numbers of simulated
samples, the Taylor surrogate's errors on both fall steadily up to $10^8$ samples, without the
plateau of a general-purpose network; with the weighted loss used here it needs 2--12 times more
samples than an unweighted fit for a given Hessian accuracy, and stays somewhat less accurate. The details, and
the comparison with the DeepONet surrogate, are in the companion comparison report.

"""


def build():
    S = shared_sections()
    s = S['pre'] + TITLE + S['setup'] + MODEL + S['train'] + VALIDATION + S['method'] + RESULTS
    s += S['params'] + S['quad'] + APP_TAYLOR + '\\end{document}\n'
    with open(os.path.join(HERE, 'muon_shield_robustness_taylor.tex'), 'w') as f:
        f.write(s)


if __name__ == '__main__':
    build()
