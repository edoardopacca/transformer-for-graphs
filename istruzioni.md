# Istruzioni del progetto — handoff per Claude

Documento di riferimento per chiunque (incluso Claude in una nuova chat) riprenda
questo progetto. Leggere **tutto** prima di agire.

> **⚠️ REGOLA DI LETTURA NON NEGOZIABILE (prima di toccare qualsiasi cosa).** Leggere
> **attentamente OGNI riga** di questo `istruzioni.md`. Se si sta per lavorare su un report
> specifico, leggere anche **OGNI riga del suo `.tex`** (i report sono lunghi e densi; gli
> errori più gravi in questo progetto sono sempre nati dal *non aver letto tutto*). Non
> saltare righe perché "sembrano un dettaglio": le caption, i commenti `%` nel `.tex`, e le
> righe della sezione Errori qui sotto contengono i vincoli che fanno fallire chi non li
> legge. Non rispondere/agire da una vista parziale di un file: se un `Read` è troncato,
> continuare fino in fondo.

---

> **🟨 PIVOT DI PROGETTO (dopo il Report 9) — l'obiettivo ora è IL PAPER, non più i report.**
> Tutti i report da 1 a 9 restano chiusi/congelati (§12 ne è l'indice), ma da qui in avanti la
> prof non vuole più singoli report come consegna finale: vuole **finalizzare il paper**. Vedi
> **§0 subito sotto** per l'obiettivo attuale, la scaletta del paper (citata parola per parola),
> e le regole su come Claude può/non può contribuirci. Un eventuale Report 10+ resta comunque
> utile come **documento di lavoro personale dell'utente** (per capire/validare un esperimento
> prima di deciderne l'esito per il paper), non più come sorgente da cui l'utente estrae slide
> per la prof — vedi §11 per il compito corrente.
>
> **Materiale non ancora promosso da nessuna parte**: `report/9/scratch_prof_experiments/
> scratch_prof_experiments.tex` contiene sei esperimenti di follow-up su domande sollevate dal
> Report 9 (griglia memorizzazione-vs-meccanismo per path e per cicli; softmax al posto di
> normalized-ReLU; nodi isolati mai visti in training; fine-tuning del solo margine del
> read-out) — vedi §11 per lo stato preciso di ciascuno.

---

## 0. Obiettivo attuale: il paper

**Da qui in avanti il lavoro è finalizzare il paper, non produrre altri report per la prof.**
La scaletta e i contenuti del paper vivono in due file LaTeX nella root del repo, **editati
dalla prof su Overleaf**:

- `idea_paper_by_prof.tex` — titolo, scaletta, teorema ed esperimenti del paper (titolo
  provvisorio: *"Connectivity"*).
- `setting.tex` — solo pacchetti/macro/environment dei teoremi (`\input{setting}`), zero
  contenuto scientifico. Dai comandi `\enote`/`\anote` definiti lì risultano coautori/revisori
  **Elisabetta** e **Aryo**.

> **⚠️ REGOLA NON NEGOZIABILE: Claude non modifica MAI direttamente `idea_paper_by_prof.tex` né
> `setting.tex`.** La prof scrive e sincronizza questi file su Overleaf; una modifica diretta di
> Claude in locale andrebbe fuori sincrono e creerebbe conflitti. Claude contribuisce scrivendo
> analisi/esperimenti/bozze di testo o figure altrove (es. `report/10/...`, script, risposte in
> chat), che l'utente poi riporta a mano su Overleaf nei modi che deciderà di volta in volta — non
> assumere un formato fisso per questo, va chiesto/proposto caso per caso.
>
> **Se il file è cambiato dall'ultima lettura, rileggerlo per intero prima di agire** (vale la
> stessa regola non negoziabile di lettura in cima a questo documento): la citazione qui sotto è
> uno snapshot, non la fonte di verità.

### Contenuto di `idea_paper_by_prof.tex`, citato parola per parola

(Snapshot preso in questa sessione; il preambolo LaTeX puramente tipografico — `\documentclass`,
`\usepackage`, `\title`, `\author`, `\date`, `\begin{document}`/`\end{document}` — è omesso perché
non è contenuto, il resto è riportato integralmente, incluso il teorema e la sua dimostrazione
completa in appendice.)

```latex
\section{Introduction}
\begin{itemize}
    \item Motivation of studying how Transformers learn connectivity: Connectivity as a testbed for algorithmic reasoning. References to Ye et al.,2026 and Sanford et al.,2024, Abbe et al., 2024.
    \item Central question: \textit{How do Transformers learn connectivity? Which architectural features are needed for a given graph structure? }
    \item Previous works study relationship between depth of Transformer and graph diameter for expressivity: Sanford: study depth needed worst cases over graphs and show that depth $O(\log(n))$ is needed, where $n$ is number of nodes. Ye et al. show that depth $ 3^L $ is sufficient and needed, where $L$ is graph diameter.
    % They consider generalization after training on ER graphs.
    \item We study whether graph diameter is indeed a bottleneck. We restrict to specific graph families: paths, ...
\end{itemize}

\paragraph{Our Contributions.}

\section{Related Work}

\begin{itemize}
    \item \href{https://arxiv.org/pdf/2405.18512}{Understanding Transformer Reasoning Capabilities
via Graph Algorithms}.
    \item \href{https://arxiv.org/pdf/1307.4884}{Smoothed analysis on connected graphs}: random perturbation of the edge set of an arbitrary (adversarial) graph, makes the diameter to be $O(\log(n))$ with high probability.
\end{itemize}

\section{Setting}
\begin{itemize}
    \item description of architecture.
\end{itemize}


\section{Expressivity with Similarity Readout}

\begin{theorem}
\label{thm:similarity_readout_doubles_radius}
Let $G=(V,E)$ be an undirected graph on $n$ vertices and let
$A\in\{0,1\}^{n\times n}$ be its self-loop-augmented adjacency matrix,
i.e.\ $A_{ij}=1$ if and only if $i=j$ or $\{i,j\}\in E$. Let $d_G$ denote
shortest-path distance in $G$, with $d_G(i,j)=\infty$ if $i,j$ lie in
distinct connected components. Fix $R\geq 1$ and suppose
\[
H(A)=p_R(A)=\sum_{r=0}^{R}\alpha_r A^r,
\qquad \alpha_r\geq 0 \text{ for all } r,\quad \alpha_R>0 .
\]
Let $H_i(A)$ denote the $i$-th row of $H(A)$ and define
\[
s_{ij}(A)=\frac{\langle H_i(A),H_j(A)\rangle}
{\|H_i(A)\|_2\,\|H_j(A)\|_2}.
\]
Then $s_{ij}(A)$ is well defined for all $i,j$, and
\[
s_{ij}(A)>0 \quad\Longleftrightarrow\quad d_G(i,j)\leq 2R .
\]
Consequently, thresholding $s_{ij}(A)$ at $0$ computes all-pairs
connectivity exactly on every graph whose connected components have
diameter at most $2R$.
\end{theorem}


\section{Experimental Results}

\subsection{Path graphs: connectivity can be learned beyond the $2*3^L$ threshold}

\begin{itemize}
    \item odd training. Unbalanced case (exact match perfect), balanced case (exact match not perfect).
\end{itemize}


\subsection{Difference between paths and cycles}
\begin{itemize}
    \item Motivation: In the attention score, extreme points are important. We compare with cycles, where diameter is the same but no extreme points.
    \item Paths tend to make far points disconnected, cycles tend to connect disconnected points.
\end{itemize}

\subsection{Generalization to three paths, finetuning??}


\subsection{Other graph structures: multi-path, barbel}
\begin{itemize}
    \item If somehow helps making the point that diameter is not the right measure.
    \item If experimental setting is consistent with the previous sections.
\end{itemize}


\section{Conclusion}


\appendix

\section{Proof of Theorem~\ref{thm:similarity_readout_doubles_radius}}

\begin{proof}
\emph{Step 1: walk characterization of powers of $A$.}
We claim that for every $k\geq 0$ and all $i,j\in V$,
\begin{equation}\label{eq:power_support}
(A^k)_{ij}>0 \quad\Longleftrightarrow\quad d_G(i,j)\leq k .
\end{equation}
The entry $(A^k)_{ij}$ counts walks of length $k$ from $i$ to $j$ in the
augmented graph $G^{\circ}$ obtained from $G$ by adding a self-loop at
every vertex; since all entries of $A$ are nonnegative, $(A^k)_{ij}>0$
if and only if such a walk exists.

($\Leftarrow$) If $d_G(i,j)=\ell\leq k$, take a shortest path
$i=v_0,v_1,\dots,v_\ell=j$ in $G$ and prepend $k-\ell$ traversals of the
self-loop at $i$; this yields a walk of length exactly $k$ in
$G^{\circ}$ from $i$ to $j$.

($\Rightarrow$) Conversely, given a walk of length $k$ from $i$ to $j$
in $G^{\circ}$, delete every self-loop step; the result is a walk in $G$
from $i$ to $j$ of length at most $k$, whence $d_G(i,j)\leq k$.

\emph{Step 2: support of the embeddings.}
Since $\alpha_r\geq 0$ and every $A^r$ is entrywise nonnegative, $H(A)$
is entrywise nonnegative, and
\[
H_{im}(A)=\sum_{r=0}^{R}\alpha_r (A^r)_{im}>0
\quad\Longleftrightarrow\quad
\exists\, r\in\{0,\dots,R\}:\ \alpha_r>0 \text{ and } (A^r)_{im}>0 .
\]
We show this holds if and only if $d_G(i,m)\leq R$. If $H_{im}(A)>0$,
then $(A^r)_{im}>0$ for some $r\leq R$, so by \eqref{eq:power_support}
we get $d_G(i,m)\leq r\leq R$. Conversely, if $d_G(i,m)\leq R$, then
$(A^R)_{im}>0$ by \eqref{eq:power_support}, and since $\alpha_R>0$ the
corresponding term is strictly positive. Hence
\begin{equation}\label{eq:ball_support}
\operatorname{supp}\bigl(H_i(A)\bigr)
= B_R(i) := \{m\in V:\ d_G(i,m)\leq R\}.
\end{equation}
Note that $\alpha_R>0$ alone suffices: no positivity is required of the
lower-order coefficients.

\emph{Step 3: well-posedness.}
For every $i$ we have $d_G(i,i)=0\leq R$, so $i\in B_R(i)$ and
$H_{ii}(A)>0$ by \eqref{eq:ball_support}. Hence $\|H_i(A)\|_2>0$ for all
$i$ and $s_{ij}(A)$ is well defined. Moreover, since the denominator is
strictly positive, $s_{ij}(A)$ and $\langle H_i(A),H_j(A)\rangle$ have
the same sign.

\emph{Step 4: sign of the inner product.}
By nonnegativity of $H(A)$,
\[
\langle H_i(A),H_j(A)\rangle
= \sum_{m\in V} H_{im}(A)\,H_{jm}(A) > 0
\quad\Longleftrightarrow\quad
\exists\, m\in V:\ H_{im}(A)>0 \text{ and } H_{jm}(A)>0,
\]
i.e., by \eqref{eq:ball_support},
\begin{equation}\label{eq:ball_intersect}
\langle H_i(A),H_j(A)\rangle>0
\quad\Longleftrightarrow\quad
B_R(i)\cap B_R(j)\neq\emptyset .
\end{equation}

\emph{Step 5: ball intersection iff distance at most $2R$.}
If $m\in B_R(i)\cap B_R(j)$, then $i$ and $j$ lie in the same connected
component and, by the triangle inequality,
\[
d_G(i,j)\leq d_G(i,m)+d_G(m,j)\leq 2R .
\]
Conversely, suppose $d_G(i,j)=\ell\leq 2R$ and let
$i=v_0,v_1,\dots,v_\ell=j$ be a shortest path. Set
$m=v_{\lceil \ell/2\rceil}$. Along a shortest path,
$d_G(v_0,v_t)=t$ and $d_G(v_t,v_\ell)=\ell-t$ for every $t$, so
\[
d_G(i,m)=\lceil \ell/2\rceil\leq \lceil 2R/2\rceil = R,
\qquad
d_G(m,j)=\ell-\lceil \ell/2\rceil=\lfloor \ell/2\rfloor\leq R,
\]
whence $m\in B_R(i)\cap B_R(j)$. Therefore
\begin{equation}\label{eq:distance_iff}
B_R(i)\cap B_R(j)\neq\emptyset
\quad\Longleftrightarrow\quad
d_G(i,j)\leq 2R .
\end{equation}

\emph{Conclusion.}
Combining Steps~3--5,
\[
s_{ij}(A)>0
\quad\Longleftrightarrow\quad
\langle H_i(A),H_j(A)\rangle>0
\quad\Longleftrightarrow\quad
d_G(i,j)\leq 2R .
\]
If every connected component of $G$ has diameter at most $2R$, then
$d_G(i,j)\leq 2R$ if and only if $d_G(i,j)<\infty$, i.e.\ if and only if
$i$ and $j$ are connected; disconnected pairs have $d_G(i,j)=\infty>2R$
and hence $s_{ij}(A)=0$. Thus $\mathbf{1}\{s_{ij}(A)>0\}$ is exactly the
connectivity matrix of $G$.
\end{proof}
```

### Come leggere questa scaletta rispetto ai Report 1–9 (mapping, non sostituto della lettura)

- Il **Teorema** (§4 del paper) è la versione idealizzata/dimostrata di un fatto trovato
  empiricamente nel **Report IV**: il read-out di similarità raddoppia il raggio a `2R` (nel
  nostro transformer reale `R=3^L`, cioè `2R=2·3^L`, il "doubled wall"). `H(A)=p_R(A)` è un
  polinomio idealizzato in `A`, non il trunk vero — è la controparte teorica pulita, non una
  descrizione letterale della RoBERTa-faithful transformer.
- **§5.1 "Path graphs... beyond the $2\cdot3^L$ threshold"** = il risultato
  sbilanciato-vs-bilanciato di **Report VI (Thread B)**, spiegato meccanicisticamente in
  **Report VII** (endpoint-completion signal) e ripreso su n=46 nel **Report IX**.
- **§5.2 "Difference between paths and cycles"** = **Report VIII** (collasso totale sui due
  cicli mai visti in training) e **Report IX Thread B** (stesso regime sistematico sui cicli:
  seed lottery).
- **§5.3 "Generalization to three paths, finetuning??"** = **Report IX Thread A.3/A.4** (fallimento
  sul terzo componente) più l'esperimento 6 di `scratch_prof_experiments.tex` (fine-tuning del
  solo read-out, non risolutivo su un K≥3 generico) — **è la sezione aperta su cui si sta
  lavorando ora** (vedi §11): un fine-tuning mirato su celle specifiche, non un K≥3 generico.
- **§5.4 "Other graph structures: multi-path, barbell"** = Report IV/V (parallel-paths, barbell)
  e Report VI (multipath).

---

## 1. Obiettivo del progetto

Studiamo se un transformer **standard** impara la soluzione *algoritmica*
(matrix-powering) per la **connettività di grafi**, partendo dal paper:
**Ye, Fu, Jia, Sharan (2026), "Transformers Provably Learn Algorithmic Solutions
for Graph Connectivity, But Only with the Right Data"** (PDF in repo:
`2510.19753v2.pdf`).

Tesi del paper:
- un modello a `L` layer risolve la connettività su grafi di diametro ≤ `3^L`
  (calcola potenze dell'adiacenza `A^k`); `3^L` è la **capacità**.
- il **data lever**: allenare *solo* su grafi entro capacità (diam ≤ `3^L`)
  spinge (dimostrato per il **Disentangled** transformer) verso l'algoritmo
  invece dell'euristica del grado; il paper *sostiene empiricamente* (Fig. 7/11)
  che questo si trasferisce al transformer standard.

**Storia dei report** (ciascuno è autosufficiente nel proprio `.tex`; dettaglio in §12):

- **Report 3** (`report/3/`): il data lever **non regge** per lo standard transformer
  (la generalizzazione OOD è dominata dal seed, non dal filtro di diametro); caratterizza il
  **muro di capacità** `3^L` (≈9 a L=2).
- **Report 4** (`report/4/`): il **read-out di similarità** aiuta il reach a lunga distanza ma
  non il bottleneck (il limite è il *trunk*); apre la domanda diametro-vs-spectral-gap;
  introduce il probe **parallel_paths** (distanza fissa, k cammini) e l'euristica del grado
  come terzo modo di fallire.
- **Report 5** (`report/5/`): matrix-powering **vs** DFS/BFS visit-bounded (bridged-cliques,
  oracoli). Verdetto: il modello è matrix-powering distance-bounded, non una traversata
  bounded; il "node budget" osservato è un'euristica soft, non una seconda capacità.
  **CHIUSO.**
- **Report 6** (`report/6/`) — "path di ragionamento": quali dati servono per imparare un
  ragionamento, lungo gli assi **parallelismo** (multipath) e **struttura iterata**
  (bridged-cliques a catena). Introduce il **principio dati non negoziabile**: ogni
  esperimento allena su UNA distribuzione esplicitamente nominata, mai un mixing opaco
  (vedi §9). **CHIUSO.**
- **Report 7** (`report/7/`) — apre il trunk: perché uno split asimmetrico a due catene
  generalizza ben oltre il muro. Introduce la batteria meccanicistica standard (sweep
  comportamentale + attention scores reali + node-to-node contribution) riusata da ogni
  report successivo. **CHIUSO.**
- **Report 8** (`report/8/`) — dove/come viene combinata l'informazione di connettività nel
  trunk. Introduce il **probe stagewise a 5 stadi** (`H^(0)→H_attn^(1)→H^(1)→H_attn^(2)→H^(2)`)
  che localizza il meccanismo per sotto-blocco (attention layer 2 costruisce il reach lungo,
  MLP2 affila il cut). **CHIUSO.**
- **Report 9** (`report/9/`) — oltre il diametro: generalizzazione OOD su path (training più
  stretto, terzo/quinto/settimo componente mai visto) e lo stesso regime sistematico applicato
  ai **cicli** (niente estremi aperti). Scoperta principale: **seed lottery** sui cicli (3/4
  seed collassano a "tutto connesso" oltre una soglia assoluta, 1/4 risolve sempre).
  **CHIUSO.** Dettaglio finale in §12; materiale di follow-up non ufficiale in
  `scratch_prof_experiments.tex` (§11).

---

## 2. Workflow e REGOLE OPERATIVE (non negoziabili)

- **Git: Claude NON committa e NON pusha mai.** Claude modifica i file in locale e
  *consegna i comandi git all'utente*, che li esegue. (Vedi memoria
  `feedback_no_git_commits`.) Claude **può** eseguire `git pull`.
- **⚠️ REGOLA NON NEGOZIABILE — il progetto ha SEMPRE accesso completo all'HPC Bocconi,
  training incluso.** Il sandbox in cui gira Claude non ha accesso di rete diretto (niente
  `ssh`/`scp` interattivi verso HPC — è un limite tecnico della sessione, non del progetto),
  ma il pattern è identico a quello di git: **Claude prepara lo script e i comandi `sbatch`,
  l'utente li lancia su HPC.** Questo vale per QUALSIASI training, non solo per eval.
  **Non scrivere MAI** frasi tipo "non abbiamo accesso al training", "risorse che Claude non
  ha", "fuori portata da questo sandbox" riferite a HPC/training — sono FALSE. Se un
  esperimento richiede un retrain: prepara lo script/il comando `sbatch` e consegnalo,
  esattamente come per i comandi git; non presentarlo come un limite strutturale. (Vedi
  memoria `feedback_hpc_access_always_available`.)
- **Branch: si lavora su `main`.** Non creare branch nuovi se non richiesto.
- **HPC (Bocconi): MAI calcolo sul nodo di login.** Qualsiasi cosa pesante va via
  `sbatch` (o `srun` su compute node). Sul login solo git/ls/tail/cat.
- **Eval leggeri**: vanno bene **in locale sul Mac** (MPS/CPU) se non toccano checkpoint su
  HPC. Gli eval che toccano i **checkpoint su HPC** (i `.pt`, che NON sono in git) → o si
  `scp`-ano i `.pt` in locale, o si lancia l'eval via `sbatch` su HPC (quest'ultimo preferito
  quando serve GPU o quando il checkpoint è pesante).
- **Cosa va in git**: solo artefatti piccoli — `.json`, `.png`, `.npz`, `.csv`,
  `history.json`, log `out/*.out`, sorgenti, `.tex`. **Mai** i `.pt` (gitignored: `*.pt`) né
  gli artefatti LaTeX (`*.aux *.fls *.fdb_latexmk *.out *.synctex.gz`, e `*.log`).
- Ciclo tipico: *edit in locale → utente push → `git pull` su HPC → `sbatch` →
  a fine job: su HPC committa json/png/npz e push → in locale `git pull`*.
- Ordine push per evitare "divergent branches": se sia locale che HPC hanno
  commit, pushare prima da un lato, poi `git pull` dall'altro.
- **Smoke-test SEMPRE il codice nuovo su un checkpoint/dato giocattolo prima di girarlo su
  pesi reali** (training di poche decine di step in locale, o dati sintetici) — prassi
  seguita per ogni script nuovo dal Report 6 in poi, non saltarla.

---

## 3. Preferenze dell'utente (cosa piace / cosa no)

**Piace:**
- Analisi onesta **da ML researcher**: separare segnale da rumore, ammettere
  risultati negativi/rumorosi, non vendere come pulito ciò che non lo è.
- **Tabelle** con dati utili + **figure chiare**; didascalie informative.
- Figure che **rappresentano fedelmente** i dati (per esiti bimodali/seed-lottery → curve
  per-seed, NON media+banda che nasconde la bimodalità).
- Config **fedele al paper** quando riproduciamo, e **segnalare i conflitti** apertamente.
- Proporre esperimenti, ragionare sul "perché", spiegare in modo semplice quando
  chiede "non capisco".
- Concisione: *"non scrivere più di così"* — non gonfiare.
- **Standing instructions vanno eseguite, non solo riconosciute.** Se questo file (o il
  report stesso) contiene già un'istruzione su cosa fare dopo, e l'utente chiede qualcosa di
  affine con un verbo generico ("analizza", "guarda"), non basta rispondere in chat: va
  anche eseguito il passo scritto (es. scrivere l'analisi nel documento permanente, non solo
  discuterla). Distinguere sempre "analizza/discuti in chat" da "scrivi l'analisi nei
  documenti" — il secondo non va saltato solo perché la richiesta usava un verbo soft.

**Il report NON è più mostrato direttamente alla prof (dal Report 7 in poi):** l'utente
costruisce a mano un PowerPoint per la prof partendo dal report — il report è materiale di
lavoro/riferimento per l'utente, non la consegna finale. Questo NON allenta le regole di
scrittura sotto (§4): sono comunque lo standard da cui l'utente estrarrà le slide. Pattern
ricorrente dal Report 7 in poi: un report si chiude → l'utente costruisce le slide a mano →
eventuali richieste successive arrivano in una NUOVA chat (vedi memoria
`project_report7_audience_change`).

**Non piace:**
- Figure fuorvianti (media+banda su esiti bimodali/seed-lottery; preferisce la **tabella
  per-seed** o le curve per-seed distinte).
- Numeri/figure inventati o messi **senza avere i dati reali** in mano.
- Spiegazioni vaghe; risultati confusi spacciati per chiari.
- Che Claude committi/pushi/lanci sbatch da sé, o dica di non avere accesso a HPC.

---

## 4. Errori già fatti — DA NON RIPETERE

**Operativo / HPC / SLURM:**

1. **Mai calcolo sul nodo login HPC** → sempre `sbatch`.
2. **Mai grafici/numeri senza i dati reali in mano** (lavorare dall'output del terminale) →
   prima `git pull`/estrai i dati reali, poi scrivi.
3. **Time limit SLURM troppo corti**: nodi condivisi possono essere ~3× più lenti del previsto
   → prima di scrivere un nuovo `--time`, cercare il tempo storico REALE della stessa
   identica computazione (grep in questo file / `sacct`) invece di riusare un default per
   inerzia; calibrare con margine.
4. **`history.json` scritto solo a fine training** → i job killati da timeout non la
   scrivono (recuperare via eval del `last.pt`, oppure rilanciare con time limit adeguato).
5. **Partizioni `debug_*` richiedono `--qos=debug`** (altrimenti "QOS not permitted").
6. **`QOSMaxSubmitJobPerUserLimit`**: tetto sui task sottomessi insieme (in coda+run),
   ~25–30 sulla QOS `normal`. Array enormi vengono RIFIUTATI → sottometti a blocchi.
7. **Multi-partizione NON permessa** per questo account (`--partition=a,b,c` fallisce) →
   scegli UNA partizione; sposta un job PD senza scancel con
   `scontrol update JobId=<id> Partition=... TimeLimit=...` (ridurre il time è permesso,
   aumentarlo no).
8. **`QOSMaxGRESPerUser` limita le GPU contemporanee per utente** (storicamente 4): se si
   sottomettono più job insieme, solo alcuni girano subito, gli altri restano PD in coda —
   normale, nessuna azione necessaria.
9. **I nomi delle partizioni possono cambiare dopo una manutenzione cluster** (è già successo
   almeno una volta: una partizione "nuda" è sparita, sostituita dai soli tier `short_*`/
   `medium_*`/`long_*`/`debug_*`). Verificare sempre con `sinfo -o "%P %G %l %D"` prima di
   fidarsi di un `--partition=` scritto in una sessione precedente. Vedi §6 per lo stato
   attuale delle partizioni.
10. **`__pycache__/*.pyc` e i `.png` rigenerati in locale possono bloccare `git pull`** su
    HPC (modifiche locali a file tracciati → "local changes would be overwritten"). Sblocco:
    `git checkout -- __pycache__/`, eventualmente `git stash push -u`, poi `git pull`.
11. **Concatenare job SLURM**: `sbatch --dependency=afterany:JOB1:JOB2 ...` (parte quando i
    precedenti FINISCONO comunque vada). Preferire `afterany` ad `afterok` se un job a valle
    deve girare anche quando uno a monte fallisce/salta checkpoint mancanti.
12. **Uno script di eval può saltare in silenzio i checkpoint mancanti** e finire COMPLETED
    avendo valutato quasi nulla → controllare sempre il `.out` (righe tipo `-- skip ...`) e
    contare gli output prodotti prima di fidarsi.

**Codice / plotting:**

13. **`\le`/`\ge` nei titoli matplotlib** → errore mathtext; usare unicode `≤`/`≥`.
14. **`\texttt{#1}` con underscore in LaTeX** → "Missing $"; evitare di stampare path con `_`
    fuori da `\verb`.
15. **`P^L` proxy: usa `3^L`, non `L`**. Il modello a L layer raggiunge distanza `3^L`
    (matrix-powering); `diffusion_reach(adj, n_steps)` va chiamato con `n_steps = 3**L`.
16. **matplotlib xticks su sweep discreti**: settare `ax.set_xticks(...)` espliciti con i
    valori reali, non fidarsi della griglia automatica.
17. **`DataLoader(prefetch_factor=..., persistent_workers=True)` richiede `num_workers>0`** —
    su debug locale passare `--num_workers 0` disattiva anche quelle opzioni, non solo il
    parallelismo.
18. **Un `\` seguito da spazio fuori da `$...$` in una stringa matplotlib** viene
    renderizzato letteralmente (es. `"vs.\\ stage"`) — usare `"vs. stage"` semplice.
19. **`fig.colorbar(im, ax=lista_di_assi)` insieme a `fig.tight_layout()` e un `suptitle` su
    più righe produce sovrapposizioni** — riservare spazio con
    `fig.tight_layout(rect=[0,0,W,H])` e usare un colorbar-axis dedicato via
    `fig.add_axes([...])`. **Dopo ogni script di plotting nuovo, guardare SEMPRE la figura
    generata (via `Read` sull'immagine)** prima di considerarlo finito — un warning assente
    non basta, e due curve possono sovrapporsi esattamente nascondendosi a vicenda (successo
    quasi a leggere una conclusione invertita da un grafico dove exact=0 e cut=0 coincidevano
    visivamente).
20. **`git restore runs/` (per sbloccare un `git pull`) scarta anche figure tracked
    rigenerate ma non committate** → dopo un `git restore runs/`, rilanciare i plot-script
    prima di fidarsi delle figure locali.

**Scrittura del report (valide per OGNI report):**

21. **MAI la metrica "reach radius" a numero singolo** (es. "massima distanza con acc≥soglia").
    Nasconde cosa succede in mezzo (l'accuracy per-distanza è non-monotona). Mostrare sempre
    l'evoluzione completa (curva/tabella per-distanza).
22. **⚠️ OGNI tabella e OGNI figura deve dichiarare MODELLO + TRAINING SET + TEST SET +
    METRICA nella caption, SEMPRE.** Template: «\emph{Model:} … (arch, L, d_model,
    read-out), trainato su … (training set), N seed, n=…. \emph{Test set:} … (famiglia,
    #grafi). \emph{Metrica:} … (media/mediana su …).» Se manca uno di questi campi, la
    caption è incompleta.
23. **⚠️ La caption NON rifà l'analisi del corpo.** Solo Model + Training + Test + Metric,
    più le legende strettamente necessarie a leggere la figura (colori, dashed/dotted,
    grassetto). Zero interpretazione ("il modello crolla", "segue X") in caption — quella
    frase deve già stare nel corpo che annuncia la figura.
24. **NESSUN riferimento a CODICE/FILE/PATH/FLAG nel testo renderizzato del report.** Via dal
    PDF: nomi di funzioni/generatori, script, path di output, field json, flag CLI. Vanno
    tenuti nei commenti `%` del `.tex` (invisibili) o in questo file. Unica eccezione: i path
    in `\includegraphics`/`\figorbox`. **Errore ripetuto due volte nella stessa sessione
    nonostante fosse già noto** → dopo ogni blocco di prosa nuovo, subito
    `grep -n "texttt\|\.py\b\|\.sbatch" file.tex | grep -v "^[0-9]*:%"`, non aspettare la
    verifica finale.
25. **NIENTE 'advisor'/attribuzioni personali, NIENTE storia/sequenza cronologica degli
    esperimenti.** Non scrivere "suggested by the advisor"; non rivelare che un esperimento è
    stato fatto in due tempi ("we re-ran X with Y **added**") → presentarlo come un unico
    esperimento disegnato così dall'inizio. Al lettore non importa l'ordine cronologico.
26. **⚠️ Ogni grafico che aggrega su più seed deve mostrare error bars (std sui seed), MAI
    solo la media nuda.** Se il fenomeno è bimodale/seed-lottery, preferire linee/punti
    per-seed distinti — ma mai una media senza indicare la dispersione.
27. **⚠️ Quando uno sweep discreto trova qualcosa di interessante in un punto (picco, salto,
    transizione), valutare SEMPRE anche i punti immediatamente vicini**, non saltare al
    prossimo valore già pianificato — un valore isolato non basta a dire che la forma locale
    è quella che sembra.
28. **Non comprimere una tabella copiata/adattata da un report precedente togliendo colonne
    che servono al contesto** (dimensioni, split, condizioni) solo perché sembrano
    ridondanti — la leggibilità stand-alone conta più della compattezza.
29. **La sezione "piano/esperimenti" è un riassunto tematico col verdetto in vista, non un
    elenco arido che poi duplica l'intro dei Results.** Tenere: setup comune, costruzione
    core, una roadmap "in hindsight" raggruppata per macro-aree con i `\ref`.
30. **Titoli-paragrafo piani (non enigmatici)**, simboli definiti alla prima occorrenza anche
    in caption, sweep discreti con xtick espliciti anche in figura.
31. Quando un'analisi si rivela confusa/confondata, **correggerla onestamente** invece di
    tenere una tabella fuorviante — un risultato negativo pulito batte un positivo sporco.
32. **⚠️ Quando un esperimento misura una quantità su TUTTE le famiglie/condizioni,
    analizzarla su TUTTE, non solo sul caso "interessante"** — prima di scrivere, chiedersi
    "questo dato copre più casi di quelli che sto guardando?" e coprirli tutti.
33. **Report chiuso ≠ report finale se contiene ancora riferimenti a lavoro non fatto o
    documenti inesistenti.** Prima di considerare un report davvero finito: rimuovere
    scaffolding di pianificazione (blocchi "DRAFT/PLAN ONLY", "[Data pending]"), aggiornare
    ogni frase che parla al futuro ("da lanciare in una sessione successiva") a passato/
    presente, e verificare che ogni "vedi documento separato" citato esista davvero (grep sul
    contenuto atteso) — altrimenti toglierlo piuttosto che lasciare un riferimento morto.

**Nota su oracoli/traversate bounded (Report 3–5, storico):** i Report 3–5 hanno usato una
metodologia estesa di confronto con oracoli (matrix-power vs bounded-BFS/DFS su bridged-
cliques e barbell) per testare se il modello facesse una traversata bounded invece di
matrix-powering — verdetto: no, è matrix-powering distance-bounded. Se un report futuro
riapre questa domanda, il codice (`dfs_oracle.py`, `eval_oracle_agreement_families.py`,
`eval_bridged_cliques.py`) e il ragionamento completo sono in `report/5/
transformer_for_graphs_5.tex`; non ripetuto qui.

---

## 5. Cosa Claude deve sapere in una nuova chat

- Repo locale: `~/transformer-for-graphs` (Mac). Su HPC: `~/transformer-for-graphs`.
- HPC alias ssh: `hpc`. Utente: `3352759`. Env conda: **`graph_tf`**.
- Report: `report/1/` (solo PDF, niente sorgente), `report/2/` … `report/9/`, ciascuno
  `transformer_for_graphs_N.tex`. **Compilare dalla cartella del report** (i path figura
  usano `\graphicspath{{../../}}`; `\includegraphics{runs/reportN/...}`). Due passate di
  `pdflatex` per i `\ref`. Ogni `.tex` ha l'helper `\figorbox{path}{width}` (fallback grigio
  se la figura non è ancora pullata — utile solo mentre un report è ancora aperto; su un
  report chiuso non dovrebbe mai comparire).
- I checkpoint `.pt` stanno **solo su HPC** (gitignored). I risultati piccoli
  (json/png/npz/csv) sono in `runs/...` e versionati.
- **`runs/` è organizzata in un bucket per report**: `runs/reportN/...`. Ogni nuovo output va
  nel bucket del report corrente. Alcuni script cross-referenziano bucket precedenti quando
  riusano checkpoint già esistenti (es. il Report 9 riusa checkpoint del Report 6 per la
  visualizzazione attention su grafi multipath) — controllare i path espliciti negli script
  stessi piuttosto che assumere. Per trovare un `.pt` (mai bucketato, sempre gitignored):
  `find runs -name '*.pt'` senza scoping, cercare per run-name.
- Memoria persistente: **niente git da Claude**; `git pull` sì.

**File/script chiave, sempre validi:**
- `model.py`: `ModelConfig`, `GraphConnectivityTransformer` ("minimal", pre-LayerNorm,
  read-out simmetrizzato) e `RobertaGraphTransformer` ("RoBERTa-faithful", post-LayerNorm,
  dropout, init 0.02) — vedi §7. `GraphBinaryClassifier` (classificatore binario,
  mean-pool+1 logit). `laplacian_smoothness`.
- `data.py`: tutti i generatori di grafi + le misure strutturali — vedi §8 per l'elenco
  aggiornato.
- `experiments2/train_families_n20.py`: **l'entrypoint di training generico e riusato da
  ogni report dal 6 in poi**, parametrico in canvas (`--n_nodes`/`--p`), famiglia/e
  (`--families`, sempre esplicite dal Report 6, mai `mixed` di default), architettura
  (`--arch roberta|minimal`), read-out (`--readout linear|similarity`, `--sim_fixed` per
  congelare scale/bias), attenzione (`--attn_kind normalized_relu|softmax`), profondità
  (`--n_layers`), seed, step/batch. Nome cartella output:
  `n{n}_{families}_{arch}_{readout}_lam{λ:g}_seed{S}` (più suffissi per `sim_fixed`/
  `attn_kind`/`n_layers` non-default).
- `eval_families.py`/`plot_families.py`: batteria di famiglie generiche (usato soprattutto
  nei Report 3–4).
- **Pattern della batteria meccanicistica standard (dal Report 7 in poi)**: per ogni nuova
  condizione di training si ripete lo stesso schema — uno script `eval_*.py`/
  `mechanistic_*.py` per lo sweep comportamentale + gli attention score reali, uno
  `stagewise_*.py` per il probe a 5 stadi (vedi §7), e un `plot_*.py` gemello che rigenera le
  figure in locale senza GPU. I nomi esatti cambiano per esperimento (es.
  `mechanistic_asym_chains.py`/`plot_mechanistic_asym_chains.py` per gli split a due
  componenti, `stagewise_kway.py` per i K-way, `mechanistic_parallel_paths_isolated.py` per
  il test sui nodi isolati) — usare `ls *.py | grep -i <parola chiave>` per trovare quello
  giusto piuttosto che fidarsi di un elenco statico qui, che andrebbe out-of-date ad ogni
  nuovo esperimento.
- `scripts/*.sbatch`: i lanci SLURM corrispondenti, un file per condizione/seed.

---

## 6. Setup HPC

**Partizioni (stato corrente — verificare comunque con `sinfo -o "%P %G %l %D"` prima di
fidarsi, i nomi possono cambiare dopo una manutenzione cluster, vedi errore 9 in §4):**

Per ogni famiglia di nodi (`gpuh200`/H200, `gpunew`/H100, più eventuali pool temporanei) ci
sono fasce a priorità decrescente per job più lunghi:
- **`short_*`** (cap **1h10**, priorità alta): usare per eval-only veloci (batterie di
  qualche minuto per checkpoint). Partono quasi subito.
- **`medium_*`** (cap **6h10**): training piccoli (canvas piccolo, poche ore).
- **`gpuh200`/`gpunew`** nude (cap **1 giorno**): training standard di questo progetto
  (canvas n=40/n=46, tipicamente 6–16h).
- **`long_gpuh200`/`long_gpunew`** (cap **3 giorni**, solo 2 nodi ciascuna — spesso più
  congestionate delle nude): usare solo se serve davvero >1 giorno, o se le nude sono piene.
- **`debug_*`** (15 min, richiede `--qos=debug`).
- **`gpua100`** (pool temporaneo, nodi A100 riassegnati dalla partizione studenti per
  un periodo limitato — verificare con l'utente se è ancora disponibile prima di
  affidarcisi per training che devono girare oltre la finestra concordata).

**Regole pratiche**: eval-only → `short_*`, `--time=01:00:00`. Training piccolo → `medium_*`.
Training standard (n=40/46) → `gpuh200`/`gpunew` nuda, `--time` calibrato sul tempo storico
reale (§4, errore 3), non un default riusato per inerzia. Spostare un job PD senza scancel:
`scontrol update JobId=<id> Partition=... TimeLimit=...` (ridurre il time è permesso,
aumentarlo no). Mai multi-partizione (`--partition=a,b,c` non è supportato per questo
account).

**Altri vincoli:**
- **`QOSMaxGRESPerUser`** limita le GPU contemporanee per utente (storicamente 4): job in
  eccesso restano PD in coda, nessuna azione necessaria.
- sbatch standard: `--account=3352759`, `--gpus=1`, `--cpus-per-task=16` (8 per eval-only
  CPU-bound), `--mem=40G` (16G per eval-only), env-var `OMP_NUM_THREADS=1` ecc.,
  `source ~/.bashrc; conda activate graph_tf`.
- **Tempi indicativi** (batch 1000, 1M step, RoBERTa L=2/d512): n=20 ~1h40m; n=40/n=46
  ~6–16h a seconda del carico dei nodi condivisi (possono essere ~3× più lenti del previsto —
  sempre controllare `sacct` di un job storico simile prima di fissare `--time`).
- Quota **home** limitata (~180–200G); le cache HF/pip (`~/.cache`) e gli env conda la
  riempiono. Se "Disk quota exceeded": pulire `~/.cache`, `conda clean`.
- **`/scratch` ha retention 30 giorni** (pulizia automatica dei file più vecchi). Questo
  progetto vive sotto `~/` (home), non `/scratch`, quindi non è normalmente toccato — rilevante
  solo se una sessione futura sposta output pesanti lì per qualche motivo.

---

## 7. Architetture (in `model.py`)

Config comune "big": `d_model=512`, `d_ff=2048`, `n_layers=2` (parametrico, `--n_layers`),
`~6.3M` params. GELU FFN; nessun mask causale; nessun positional encoding (l'identità in A+I
fissa i token). Ottimizzatore AdamW, peak lr `1e-4`, weight decay `1e-4`, cosine + warmup,
bf16. Input = `A + I` (self-loop), target = matrice di connettività `R` (`R_ij=1` se i,j
stessa componente).

- **Attenzione**: `attn_kind` ∈ {`"softmax"` (classica), `"normalized_relu"`} — quest'ultima
  è la costruzione `α = (1/n)·ReLU(QKᵀ/√d_h)` del paper Ye et al., diventata la scelta
  standard per il matrix-powering dal Report 6/7 in poi. Verificato (Report 9) che ogni
  script di eval/mechanistic legge `attn_kind` dal checkpoint salvato — nessuna modifica
  necessaria per valutare un checkpoint softmax con gli stessi script.
- **`GraphConnectivityTransformer`** — variante "minimal / A.1-style": pre-LayerNorm,
  read-out lineare simmetrizzato.
- **`RobertaGraphTransformer`** — variante "RoBERTa-faithful" (post-LayerNorm, dropout 0.1,
  init `N(0,0.02)`), **standard dal Report 7 in poi** per ogni esperimento nuovo.
- **Read-out**: `readout="linear"` (default) o `readout="similarity"`
  (`R̂_ij = scale·cos(h_i,h_j) + bias`, scale/bias imparabili) — quest'ultimo standard dal
  Report 8/9 in poi: rende il comportamento molto più robusto al mixing imperfetto del
  trunk, raddoppia il reach a singola rotta (`2·3^L`). `--sim_fixed` congela scale=1/bias=0
  (nessun margine di decisione imparabile, logit = puro cos) — usato come ablazione
  (Report 9): mostra che è proprio il margine imparabile a difendere il cut.
- **`GraphBinaryClassifier`** — stesso trunk + mean-pool + 1 logit, per un task binario
  (es. "1 vs 2 componenti").
- **`laplacian_smoothness(H, A)`** = `Tr(HᵀLH)/#archi`, loss ausiliaria spettrale (usata nel
  Report 4, poi droppata: inerte a convergenza col read-out di similarità).

Numero di teste: single-head nella riproduzione paper e in ogni esperimento dal Report 6 in
poi; 4 teste solo nei vecchi modelli n40big (Report 2/3).

**Il probe stagewise a 5 stadi** (introdotto Report 8, riusato da ogni report successivo):
cattura le embedding a `H^(0)` (read-in), `H_attn^(1)` (dopo attention 1), `H^(1)` (dopo
MLP1), `H_attn^(2)` (dopo attention 2), `H^(2)` (finale, dopo MLP2) e ne misura la geometria
coseno/similarity per categoria di coppia. Convenzione consolidata: attention layer 2 di
solito costruisce il reach a lungo raggio in modo largo/indiscriminato; il feed-forward
successivo (MLP2) affila selettivamente il cut. Vale come punto di partenza per leggere
qualsiasi nuovo probe stagewise su una condizione mai vista prima.

---

## 8. Dati (generatori in `data.py`)

Tutti i generatori producono adiacenza *senza* self-loop; i self-loop si aggiungono con
`add_self_loops`. Target via `compute_connectivity_matrix`; distanze via
`compute_all_pairs_shortest_paths` (APSP, scipy).

**Generatori base:**
`generate_er_graph(n, p, rng)` (Erdős–Rényi) · `generate_one_chain_graph(n)` (un path) ·
`generate_one_cycle_graph(n)` (un ciclo) · `generate_two_chains_graph(n, k)` /
`generate_two_cycles_graph(n, k)` (due path/cicli disgiunti, split bilanciato n=2k) ·
`generate_two_cliques_graph(n, k)` · `generate_path_union_graph(n, rng, max_paths=4)`
(unione di k∈{1..4} path disgiunti che partizionano tutti gli n nodi — training "base" dei
Report 6–9) · `generate_blocks_graph(n, rng, kind="er"|"clique")` · `generate_barbell_graph` ·
`generate_random_regular_graph(n, rng, degree=3)` · `generate_chain_plus_graph(n, rng)`.

**Split a due componenti, esplicitamente nominati (Report 8/9):**
`generate_split_chains_graph(n, short_len)` (split casuale in due path, ogni sample sceglie
`short_len`) · `generate_split_cycles_graph(n, short_len)` (stesso, ma chiuso a ciclo —
richiede `short_len≥3`) · `generate_split_cliques_graph`/`generate_bridged_cliques_graph`
(due clique ± un ponte, differiscono di un arco — Report 5) ·
`generate_split_cliques_asym_graph(n, short_len)` (due clique COMPLETE di taglie asimmetriche
— diverso dal precedente) · `generate_chorded_cycles_graph(n, short_len)` (due cicli con un
chord ciascuno — un solo landmark, niente endpoint aperto) ·
`generate_split_regular_graph(n, d, short_len, rng)` (due grafi d-regolari connessi — nessun
landmark, il controllo più pulito) · `generate_bridged_blocks_graph`/`generate_clique_chain_graph`
(blocchi/cricche a catena, Report 6).

**Split a K componenti (Report 9, generalizzazione oltre K=2):**
`generate_three_way_split_graph(n, small_len, large_split=None)` ·
`generate_multi_path_split_graph(n, sizes)` / `generate_multi_cycle_split_graph(n, sizes)`
(K componenti di taglie arbitrarie, path o cicli).

**Percorsi paralleli fra due terminali:**
`generate_parallel_paths_graph(n, n_paths, path_len)` (costruzione SEMPLICE: terminali `0,1`
uniti da `n_paths` path disgiunti di `path_len` archi, il resto dei nodi **isolato** — nessun
padding; usata nel test sui nodi mai visti in training, Report 9) ·
`generate_multipath_graph(n, n_full, path_len, rng, n_trunc=0, term_deg=4, trunc_len=None)`
(costruzione con padding: grado dei terminali fissato con foglie + filler sparso per il resto
del canvas, supporta anche route troncate dead-end — usata per l'esperimento multipath del
Report 6 e per le visualizzazioni di attention del Report 9); `permute_with_meta` applica una
permutazione casuale mantenendo tracciabili terminali/route.

**Misure strutturali:** `compute_graph_diameter` · `compute_spectral_gap(adj)` (Fiedler
norm.) · `effective_resistance(adj)` (via pseudo-inversa Laplaciana; scala non confrontabile
tra grafi diversi, usare il ranking) · `diffusion_reach(adj, n_steps)` (`P^n_steps`
row-stocastica, in [0,1]; usare `n_steps=3^L`, MAI `L`).

---

## 9. Scelte sperimentali e perché

- **Canvas size**: n=20 (riproduzione paper esatta, App. D.1), n=40 (standard Report 2–8),
  n=46 (Report 9 — stesso setup di n=40, solo canvas più grande, per lasciare più margine ai
  test OOD sulle taglie di split). La scelta va sempre dichiarata esplicitamente e motivata,
  non riusata per inerzia.
- **⚠️ Principio dati non negoziabile (dal Report 6 in poi): ogni esperimento allena su UNA
  distribuzione esplicitamente nominata** (o una combinazione esplicita di poche, dichiarata
  e motivata) — mai il mixing opaco uniforme-su-9-famiglie dei Report 1–5. Se si riusa un
  numero che veniva dal mixed, trattarlo da baseline, non da condizione pulita.
- **Similarity read-out standard dal Report 8/9 in poi** (vedi §7): rende il comportamento
  robusto al mixing imperfetto del trunk senza rimuovere il segnale meccanicistico
  sottostante.
- **single-head vs 4-head, minimal vs RoBERTa**: il paper non fissa l'architettura esatta
  (A.1 = idealizzata pre-norm; D.1 = "adopt RoBERTa" = post-norm/dropout/init 0.02). RoBERTa
  single-head è diventato lo standard dal Report 6 in poi.
- **Niente restrizione di diametro nel reach experiment**: serve l'opposto — grafi con
  distanze lunghe per misurare se il reach arriva davvero a `3^L`.
- **Ogni nuova condizione/ablazione va confermata su almeno 2 seed prima di trattarla come
  risultato robusto** (norma consolidata dal Report 7 in poi); un solo seed è accettabile
  solo come esplorazione preliminare, da segnalare esplicitamente come tale.

---

## 10. Lessico del report (usare questi termini)

**Metriche di base:**
- **exact-match accuracy**: frazione di grafi con matrice `R̂` esatta su *tutte* le coppie.
- **pairwise accuracy**: frazione di coppie di nodi predette correttamente.
- **reach**: pairwise accuracy sulle coppie *connesse* (target 1), spesso condizionata alla
  shortest-path distance `d` ("per-distance reach").
- **cut**: pairwise accuracy sulle coppie *disconnesse* (target 0) — tipicamente fra
  componenti diverse.
- **capacity `3^L`**: distanza massima risolvibile da un modello a L layer (architetturale,
  non di copertura dati). **within-capacity / beyond-capacity**: coppie a `d≤3^L` / `d>3^L`.
  **doubled wall `2·3^L`**: il limite raddoppiato che il read-out di similarità sblocca su
  split a due componenti sbilanciati.
- **predicted-positive rate**: frazione di TUTTE le coppie del grafo (non solo quelle
  rilevanti) predette connesse — segnale diagnostico di collasso "tutto connesso".
- **n_active**: numero di nodi non isolati in un grafo strutturato dentro un canvas n×n.
- **matrix-powering** (soluzione locale, `A^{3^L}`) vs **traversata bounded**
  (DFS/BFS visit-bounded, Report 3–5) vs **spectral/Laplacian** (globale, Report 4).

**Vocabolario meccanicistico (dal Report 7/8/9):**
- **endpoint-completion signal**: il segnale extra che un componente sufficientemente piccolo
  da risolversi per intero dà per l'*altro* componente in uno split a due, costruito dalla
  stessa attention che fa il reach ordinario (Report 7).
- **stagewise probe** (5 stadi `H^(0)→H_attn^(1)→H^(1)→H_attn^(2)→H^(2)`, §7): localizza il
  meccanismo per sotto-blocco.
- **broad-then-selective**: il pattern per cui l'attention alza la similarità in modo largo e
  indiscriminato, poi un feed-forward step la corregge selettivamente per blocco.
- **seed lottery**: fenomeno per cui semi diversi producono meccanismi qualitativamente
  diversi (scoperto per split_cycles-only nel Report 9: 3/4 semi collassano, 1/4 risolve
  sempre).
- **memorisation vs. mechanism (griglia sparsa)**: training su un insieme discreto e fisso di
  taglie di split, test sulle taglie interleaved mai prodotte — se il comportamento è
  indistinguibile tra allenate e held-out, è meccanismo genuino, non memorizzazione.
- **broadcasting endpoint**: attention quasi-uniforme dai/verso i due estremi liberi di un
  path (o i due terminali di un grafo multi-route), documentata dal Report 7 in poi.
- **K-way split**: generalizzazione oltre due componenti (K=3, 5, 6, 7…), con la stessa
  distinzione reach (trasferisce quasi sempre) vs cut (fallisce nel distinguere più
  componenti *altri* simultaneamente presenti fra loro).

---

## 11. Stato attuale e compito corrente

**Compito aperto (Report 10, documento di lavoro personale — vedi §0): fine-tuning mirato del
solo read-out su celle a tre componenti specifiche.** Nasce dalla sezione §5.3 del paper
("Generalization to three paths, finetuning??", §0) e da un'osservazione della prof sulle
figure meccanicistiche del Report IX §A.3 (`report/9/transformer_for_graphs_9.tex`,
`stagewise_threeway.py`): sulla cella a tre componenti **(15,15,16)** (split bilanciato) e sulla
**(7,15,24)** (un cut su tre fallisce, più coppie nella componente da 24 nodi oltre il muro
raddoppiato `2·3^L=18`), il checkpoint split-chains-only n=46 seed 1000
(`runs/report9/n46_train/n46_split_chains_roberta_similarity_lam0_seed1000/last.pt`) *sembra*
aver imparato bene (blocchi diagonali molto più scuri/rossi delle celle fuori diagonale nella
geometria coseno a 5 stadi), ma l'exact match resta 0 perché i logit delle coppie
cross-componente restano leggermente **sopra** lo zero. Domanda: un fine-tuning **solo di
scale/bias del read-out di similarità** (resto del trunk congelato), mirato esplicitamente su
queste celle (non un flusso K≥3 generico come l'esperimento 6 di `scratch_prof_experiments.tex`,
che infatti non aveva funzionato), riesce a spostare la soglia di decisione quel poco che serve?
Da verificare: (a) se risolve (15,15,16); (b) se questo si trasferisce a (7,15,24) mai vista nel
fine-tuning, e in particolare se rende connesse anche le coppie della componente da 24 nodi a
distanza >18 (oltre il muro raddoppiato — sarebbe il risultato più interessante); (c) se il
modello ricorda ancora lo split a due componenti (own-family, K=2) dopo il fine-tuning; (d)
quanto sono cambiati scale/bias, in valore assoluto. Fine-tuning molto più lungo del precedente
tentativo (l'esperimento 6 usava 3000 step, non 30k — verificato in
`scripts/r9_n46_pretrain_2path2cycle_finetune_readout_seed*.sbatch`; nessuna run a 30k step
esiste nel repo, quindi il confronto "non cambiava niente" va verificato di nuovo con lo script
dedicato). Script: `finetune_readout_threeway.py` (nuovo, vedi il suo docstring per l'uso
esatto); report di analisi in `report/10/` (in corso).

**Materiale esplorativo non ufficiale, se qualcuno lo riprende**:
`report/9/scratch_prof_experiments/scratch_prof_experiments.tex` (18–26 pagine a seconda
della versione, compila pulito) contiene sei esperimenti di follow-up nati da domande della
prof sul Report 9, non collegati a nessun report ufficiale:

1. **Griglia memorizzazione-vs-meccanismo, chains** (2 seed) — reach generalizza senza cuciture
   ai valori held-out; il cut collassa più duramente del training continuo alla stessa taglia.
2. **La stessa griglia, cicli** (2 seed) — i due seed disaccordano nettamente (uno collassa
   presto dentro la griglia stessa, l'altro dà una storia pulita che rompe solo al break
   abituale): conferma che è la seed lottery del Report 9 a ripresentarsi, non una proprietà
   del training a griglia. **Un terzo seed (idealmente 3000, quello che risolve sempre nella
   condizione continua) chiarirebbe se il training a griglia può mai dare un modello
   pienamente riuscito sui cicli.**
3. **Softmax al posto di normalized-ReLU** (2 seed, n=46) — niente rottura netta
   risolve-o-crolla: il cut resta quasi perfetto per tutto lo sweep, il fallimento è
   interamente nel reach (floor-e-recupero-parziale, non declino monotono).
4. **Margine del read-out congelato** (scale=1/bias=0, 2 seed) — **già promosso nel Report 9
   ufficiale come sezione A.2b**, non solo qui.
5. **Nodi isolati mai visti in training** (1 seed, 4 checkpoint split-chains-only) — il modello
   non manca mai la connessione dei terminali, ma fallisce quasi totalmente nel riconoscere i
   nodi isolati come disconnessi (normalized-ReLU non scala per la dimensione del componente,
   solo per la dimensione del canvas). **Un secondo seed manca ancora.**
6. **Fine-tuning del solo margine del read-out** (2 seed) — ri-allenare solo scale/bias su
   K≥3 non recupera nulla: conferma che il fallimento su K≥3 è un limite rappresentazionale
   del trunk, non una miscalibrazione della soglia.

**Altre domande aperte, mai riprese**: la domanda della prof sul barbell (il modello
generalizza da ER-trained a grafi barbell, o solo se i barbell sono nella distribuzione di
training?) non è mai stata riaffrontata dopo il Report 4.

---

## 12. Archivio: cosa hanno coperto i report precedenti

Ogni report è autosufficiente nel proprio `.tex` (obiettivo, setup, dati, risultati,
verdetto) — questa sezione è solo un indice per orientarsi velocemente, non un sostituto
della lettura. Lo storico sessione-per-sessione (chi ha fatto cosa, quando, con quale job
SLURM) non è più mantenuto qui: vive nella cronologia git (`git log`) e nei commenti `%` dei
singoli `.tex`, se servisse mai recuperarlo.

- **Report 1** (`report/1/`, solo PDF): baseline iniziale, capacity test, dinamiche di
  restrizione del diametro.
- **Report 2** (`report/2/`): retrain su ER a varie taglie canvas (n10/n14/n40), primo
  confronto in/OOD.
- **Report 3** (`report/3/`): Parte I — il data lever non regge per lo standard transformer
  (seed-dominated, non diametro-dominated). Parte II — caratterizza il muro di capacità 3^L.
- **Report 4** (`report/4/`): similarity read-out aiuta il reach ma non il bottleneck;
  diametro-vs-spectral-gap (aperto); probe `parallel_paths` pulito per isolare il gap;
  euristica del grado come terzo modo di fallire.
- **Report 5** (`report/5/`): matrix-powering vs DFS/BFS visit-bounded, via bridged-cliques e
  tre oracoli. Verdetto: matrix-powering distance-bounded, non traversata bounded; "node
  budget" = euristica soft data-prior, non seconda capacità. **CHIUSO, accorciato 31→26 pp.**
- **Report 6** (`report/6/`): "path di ragionamento" — multipath (parallelismo aiuta anche
  mai visto in training), two-chains asimmetriche, bridged-cliques iterate. Introduce il
  principio dati non negoziabile (§9). **CHIUSO.**
- **Report 7** (`report/7/`): apre il trunk — perché uno split asimmetrico a due catene
  generalizza oltre il muro raddoppiato. Introduce la batteria meccanicistica standard
  (sweep + attention scores reali + node-to-node contribution) e il broadcasting-endpoint.
  **CHIUSO.**
- **Report 8** (`report/8/`): dove/come viene combinata l'informazione di connettività nel
  trunk. Introduce il probe stagewise a 5 stadi; scoperta MLP2-affila-il-cut; test dei due
  cicli (OOD totale, mai visto in training) → collasso "tutto connesso". **CHIUSO.**
- **Report 9** (`report/9/`): oltre il diametro — generalizzazione OOD su path (training più
  stretto, K=3/5/6/7 componenti mai visti) e lo stesso regime applicato ai cicli. Thread A:
  il segnale di completamento non dipende dalla ricchezza della distribuzione di training,
  fallisce in modo specifico (un solo cut sbagliato su tre) quando appare un vero terzo
  componente. Thread B: **seed lottery** sui cicli — 3/4 seed collassano a un threshold
  assoluto, 1/4 risolve sempre; un test a tre cicli conferma che anche i semi che collassano
  hanno un meccanismo selettivo genuino sotto quella soglia. A.1 e A.5 non perseguiti nella
  forma originale. **CHIUSO.** Materiale di follow-up non ufficiale: §11.

---

*Nota per chi riprende il progetto in una nuova chat*: questo file è stato riorganizzato per
chiudere il Report 9 e passare la mano al prossimo report. Le regole operative (§2–§10) sono
durature e valgono per qualsiasi report futuro; §11/§12 sono la parte che va aggiornata ad
ogni nuovo report aperto/chiuso.
