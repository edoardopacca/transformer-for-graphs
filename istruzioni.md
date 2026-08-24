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

### Il compito di Claude da qui in avanti: scrivere il paper (non solo supervisionare)

> Aggiunto 2026-08-20. Da questo momento Claude ha due compiti in parallelo: (1) **supervisionare
> gli esperimenti** per capire cosa è abbastanza solido da entrare nel paper, (2) **scrivere
> effettivamente il testo del paper**, sezione per sezione, tenendo sempre a mente il contesto di
> tutto il resto (non scrivere una sezione come se le altre non esistessero — es. non introdurre in
> un esperimento un dettaglio di setting mai definito in §Setting).

**Workflow per il file**: Claude **non modifica mai direttamente** `idea_paper_by_prof.tex` né
`setting.tex` (regola già in §0, ribadita qui). Claude scrive in un **file `.tex` separato**,
copia di partenza identica a `idea_paper_by_prof.tex` (stesso preambolo, stessa `\input{setting}`),
che poi Edoardo legge e copia a mano nell'Overleaf della prof. Nome file:
`paper_draft.tex` (root del repo, accanto a `idea_paper_by_prof.tex`). Ogni volta che si scrive
un pezzo nuovo, va mantenuta la coerenza col resto del documento già scritto in quel file.
**Le immagini**: nel report i path `\includegraphics` puntano a file locali (`runs/reportN/...`);
per il paper su Overleaf le immagini vanno caricate a mano da Edoardo, quindi quando serve una
figura Claude lo segnala esplicitamente e chiede aiuto per il caricamento, non assume che il path
locale funzioni lì.

**Cartella di riferimento per lo stile**: `neurips styles/` (root del repo) contiene
`neurips_2025.tex`/`.sty`/`.pdf` — il template ufficiale della venue a cui il paper è
destinato. Il file `idea_paper_by_prof.tex` attuale **non** usa ancora la classe `neurips_2025`
(usa `article` + `setting.tex`); non cambiare la classe di propria iniziativa, è una decisione
della prof.

**Regole di scrittura (non negoziabili, dette esplicitamente da Edoardo)**:
- **Linguaggio semplicissimo, comprensibile anche a chi non ha fatto il paper con noi.** Mai
  termini composti "AI-ish" per abbreviare (es. mai scrivere qualcosa come
  "constraint-online-notwritten-based method" o "RoBERTa-based model" senza spiegare): se serve
  nominare una cosa tecnica, **scrivere il termine per esteso e poi spiegarlo subito**, non dare
  per scontato che il lettore capisca una sigla o un nome proprio (es. non scrivere "RoBERTa-based"
  e basta — spiegare cosa significa in pratica per l'architettura: quali scelte concrete comporta).
  Prima di scrivere, informarsi (mentalmente/di prassi) su quali sono le espressioni tipicamente
  "AI-ish" nei paper di questo campo ed evitarle il più possibile.
- Il paper ha un **limite di 4 pagine** (single-column), references e appendici escluse (vedi
  bando sotto) — scrivere con questo vincolo sempre in mente, non gonfiare.
- **Confronti sperimentali sempre sullo stesso training set**: la prof ha chiesto esplicitamente
  che gli esperimenti messi a confronto nel paper usino sempre la stessa distribuzione di
  training, altrimenti per lei il confronto non è valido/leggibile.

**Bando della venue (copiato verbatim da Edoardo, per capire i vincoli di formato/contenuto)**:

> We invite submissions that take principled approaches to advancing the understanding of
> generative modeling. This understanding may be pursued from diverse perspectives, including
> mathematical theory, physical modeling, and rigorous empirical analysis. In particular, we
> encourage contributions that address the following foundational questions:
> * Model classes and expressivity: What classes of functions, distributions, or algorithms can
>   modern generative models represent, and how do architectural choices shape this expressive
>   power? What makes transformers, diffusion models, discrete diffusions, and state-space models
>   particularly effective at capturing linguistic or visual structures? How do repeated
>   token-level computations expand the computational capabilities of these models, potentially
>   enabling general reasoning?
> * Learning, generalization and inductive bias: How do generative models acquire high-level
>   structure and capabilities such as in-context learning, compositional generalization, and
>   reasoning from data? What aspects of structure and skill acquisition are governed by unifying
>   principles across architectures and modalities, and which are shaped by the specific
>   inductive biases of data, objective, architecture, and optimization? Are there fundamental
>   limits to current self-supervised learning paradigms, such as next-token prediction?
> * Inference-time computation and adaptation: Modern generative systems increasingly rely on
>   computation performed at inference time, from simple sampling to test-time adaptation. When
>   and why does additional inference-time computation improve generation quality and reasoning?
>   What are the statistical and computational limits of adapting generative models to
>   distribution shifts, and how do these limits depend on model class, data structure, and the
>   nature of the shift?
> * Post-training: Which properties of pretrained models enable successful post-training? Does
>   post-training create new capabilities, reveal latent ones, or reweight behaviors already
>   present in the pretrained model? What are the fundamental distinctions between supervised
>   fine-tuning, reinforcement learning, and distillation, and when does each offer genuine
>   advantages?
>
> Submission instructions
> Submissions are limited to four single-column pages, plus unlimited pages for references and
> appendices. The reviewing process will be double-blind and all submissions must be anonymized.
> Please do not include author names, affiliations, acknowledgements, or any other identifying
> information in your submission. Submissions and reviews of rejected papers will not be made
> public. All submissions must be made through OpenReview at this link. We ask you to use the
> standard LaTeX NeurIPS style files. It is not required to fill the NeurIPS checklist.
> Appendices can be submitted in a the same PDF file as the main text.
> Note: If you are creating a new OpenReview profile, we strongly recommend using your
> institutional email address. Profiles created without an institutional email may require a
> moderation process, which can take up to two weeks.

**Conseguenze pratiche del bando**: 4 pagine per il corpo (no limite per refs/appendici) →
teorema e dimostrazione lunga possono stare in appendice (già così in `idea_paper_by_prof.tex`);
submission **anonima/double-blind** → nessun nome, affiliazione, o riferimento identificativo nel
testo finale (per ora, in fase di bozza, non è un problema urgente, ma va tenuto a mente prima
della submission vera); usare gli style file NeurIPS standard (cartella `neurips styles/`).

**Scaletta dettagliata data da Edoardo per ogni sezione (appunti suoi, copiati alla lettera —
usare come guida quando si scrive quella sezione)**:

*1. Introduction.* Iniziare spiegando perché è importante parlare di connectivity per i
transformer: "Connectivity as a testbed for algorithmic reasoning" / "Neural algorithmic
reasoning is a field of research dedicated to exploring such capabilities. Algorithmic execution
is desirable because models use it to generalize out-of-distribution and scale to larger problem
sizes." Perché connettività per grafi in particolare? Perché simula un ragionamento con
dipendenze di una rete neurale. Dopo questa introduzione, come fa Ye et al., una scritta in
**grassetto** con la domanda fondamentale: "How do Transformers learn connectivity? Which
architectural features are needed for a given graph structure?" Poi, in una frase semplice, cosa
dicevano i previous work su qual era il bottleneck (relazione fra profondità del transformer e
diametro del grafo). Poi diciamo che noi invece abbiamo studiato se il diametro era davvero un
bottleneck. Poi "Our Contributions" — per ora solo l'intestazione, si riempie quando ci sono
tutti i risultati. Bozza di contenuto per dopo (non ancora da mettere nel testo): se cambiamo in
similarity readout abbiamo come reference `2·3^L` anziché solo `3^L`; per i path graphs la
connettività può essere imparata oltre il `2·3^L` bottleneck; grafi tipo split (8,36) sono più
facili da imparare rispetto a un grafo split (23,23).

*2. Related Work.* Il riferimento allo smoothed analysis (`arxiv.org/pdf/1307.4884`) c'entra
poco così com'è — potrebbe agganciarsi meglio a nuovi esperimenti OOD (es. allenare solo su
grafi con aggiunta di nodi e poi testare su altri) o a curriculum learning, ma Edoardo stesso non
è sicuro che valga la pena — rischio di aggiungere esperimenti inutili solo per far quadrare la
related work.

*3. Setting.* Descrivere il setup dell'architettura. **La prof vuole che gli esperimenti
confrontati nel paper usino sempre lo stesso training set** — per lei è l'unico modo per fare un
confronto valido.

*4. "Similarity Read-out Doubles the Capacity".* Va messo come primo risultato sperimentale/
teorico (nota: nel file attuale è già §4, prima della sezione Experimental Results) — ha senso
metterlo lì perché poi tutta la sezione 5 parla di capacità/diametro in termini di `2·3^L`.

*5. Experimental Results — sotto-punti, come pensati da Edoardo:*
  - **(1) Beyond-threshold (§5.1).** La prof vuole mettere direttamente il caso con training
    solo sugli split "odd" (dispari) e poi test su tutti gli split, per mostrare che il caso
    sbilanciato ("unbalanced") viene imparato mentre quello bilanciato ("balanced") no. Mettere
    sicuramente anche l'immagine della geometria di similarità (stagewise/layerwise cosine
    geometry).
  - **(2) Paths vs cycles (§5.2).** Guardando gli attention score si è visto che per i path
    l'attenzione si concentra sugli estremi (endpoint) — da lì l'idea di testare anche i cicli
    per vedere se gli estremi erano fondamentali. Non lo sono, ma anche i cicli imparano
    un'euristica: questa è la prima differenza. La seconda differenza: i path tendono a rendere
    le coppie lontane disconnesse (di default), i cicli tendono a connettere coppie non connesse
    (anche se spesso in realtà "capisce" — rimando a §5.3 dove si parla del fine-tuning). Qui
    mostrare chiaramente le geometry di similarità per il caso bilanciato, con training fatto
    solo sugli "odd". Le stesse cose vanno rifatte in appendice per gli esperimenti allenati su
    tutte le combinazioni (non solo odd).
  - **(3) Generalization to three components / fine-tuning (§5.3).** Mostrare che anche qui
    spesso il modello "capisce" (nel senso della geometria) pur sbagliando la soglia di
    decisione, e quindi si è deciso di fare fine-tuning. **Punto ancora confuso per Edoardo
    stesso**: qui si torna al caso con training su tutte le combinazioni (non solo odd) — va
    capito come rendere l'intera Sezione 5 coerente rispetto a quale training set si usa dove
    (vedi anche il vincolo della prof al punto Setting sopra: stesso training set per poter
    confrontare).
  - **(4) Other structures: multipath, barbell (§5.4).** Da capire come rendere il setup
    sperimentale coerente con le sezioni precedenti (stesso training set/condizioni).

*6. Conclusion.* Non ancora sviluppata.

---

### Stato di avanzamento di `paper_draft.tex` (aggiornato 2026-08-23)

> Questa sotto-sezione va tenuta aggiornata ad ogni sessione che tocca `paper_draft.tex` — è lo
> stato reale del file, non un piano. Prima di scrivere una nuova sezione, leggere per intero
> `paper_draft.tex` da disco (può essere stato modificato da Edoardo in parallelo — è già successo:
> ha riscritto a mano il paragrafo "Model" del Setting con le equazioni esplicite mentre Claude
> lavorava in un'altra parte del file. Nessun conflitto quella volta perché le modifiche erano in
> punti diversi, ma va sempre verificato).

> **⚠️ Correzione 2026-08-23 (stessa sessione): Claude aveva scritto prosa/tabelle/figure incorporate
> per §5.1 e un'Appendice B senza che fosse stato chiesto — Edoardo aveva chiesto SOLO di generare i
> due grafici (curva exact-match vs $a$ + heatmap attention) sulla base della consulenza AI esterna
> che aveva incollato in chat, non di scriverli nel paper. Tutto quel testo è stato rimosso di nuovo:
> `paper_draft.tex` è tornato esattamente allo stato lasciato da Edoardo (Introduction, Related Work,
> Setting riscritta da lui, Teorema 1, il resto ancora placeholder), incluso rimuovere
> `\usepackage{graphicx}`/`\usepackage{float}` che Claude aveva aggiunto al preambolo. **Lezione
> operativa**: quando l'utente chiede "generami i grafici", il deliverable è il file immagine, non
> una sezione di paper — non inferire il permesso di scrivere prosa dal fatto che i dati/l'analisi
> sono già pronti. Aspettare un'istruzione esplicita tipo "scrivi la sezione" prima di toccare il
> `.tex` con contenuto narrativo.**

**Scritto e compilante (0 errori, 0 riferimenti indefiniti) al 2026-08-23 — SOLO questo, il resto è
placeholder originale:**
- **Introduction**: motivazione (Veličković & Blundell 2021 per neural algorithmic reasoning),
  perché la connettività (algoritmo esplicito via matrix powering + composizione di implicazioni
  logiche, con riferimento ad Abbe et al. 2024 sulla syllogism composition), domanda centrale in
  grassetto, Sanford et al. 2024 (profondità logaritmica, worst-case) e Ye et al. 2026
  (Disentangled Transformer, capacità $3^L$ esatta), la nostra domanda (il diametro è davvero il
  collo di bottiglia?). **Regola di stile applicata rigidamente**: mai em-dash (—), termini tecnici
  sempre scritti per esteso e spiegati subito, mai citazioni "di comodo" fuori contesto.
- **Related Work**: due paragrafi (`\paragraph`, non elenco puntato) — "Transformer expressivity and
  graph connectivity" (Sanford et al. 2024, Merrill & Sabharwal 2025) e "From expressivity to
  learnability" (Abbe et al. 2024 sulla globality, Ye et al. 2026 sul gap data-training). Resta un
  `\begin{itemize}` con solo il riferimento allo smoothed analysis (Spielman-Teng), da integrare o
  rimuovere — Edoardo ha detto che così com'è non c'entra molto, forse aggancio a OOD/curriculum
  futuri, ma "aggiungerei esperimenti inutili solo per farlo quadrare".
- **Setting** (`\label{sec:setting}`): **riscritta da Edoardo stesso** con equazioni esplicite per
  $H^{(0)}$, l'attenzione normalized-ReLU, i due residual+LayerNorm — più formale della prima bozza
  di Claude. Definisce Model, Read-out (similarity, con rimando al Teorema 1), Graph families (two-path,
  two-cycle, **setting "odd"** — nome ufficiale ora, non più "sparse-grid": training ristretto a
  $\{3,5,7,9\}$ — e three-path per l'OOD a tre componenti), Evaluation (exact match, reach, cut).
  **Bug minore non ancora corretto** (di Edoardo, da confermare con lui prima di toccarlo): la riga
  `${(3,43),(5,41),(7,39),(9,37)}$` manca degli `\{ \}` letterali (servirebbe `\{(3,43),\dots\}$`)
  — le graffe esterne sono solo raggruppamento LaTeX, quindi in output non compare alcuna parentesi
  graffa attorno alla lista.
- **§4 Expressivity with Similarity Readout**: Teorema 1 invariato (già presente in
  `idea_paper_by_prof.tex`), con la dimostrazione completa in Appendice A.
- **§5.1–§5.4, Conclusion**: ancora i placeholder `\begin{itemize}` originali di
  `idea_paper_by_prof.tex`, non toccati.

**Deliverable reale di questa sessione: due file immagine, NON incorporati nel paper**, in
`paper_figures/` (root del repo), ciascuno in **PDF vettoriale + PNG** di anteprima:
- `fig_oddtrain_exactmatch_vs_a.{pdf,png}` — curva exact-match vs taglia $a$ del training "odd" a
  due path (n=46, similarity read-out, seed 1000/2000). Segue la spec esatta data da Edoardo
  (Opzione 1 della consulenza AI): blu `#0072B2` per dati/media (linea 1.8pt, marker 5pt), grigio
  `#B0B0B0` per le linee dei singoli seed (0.7pt) — **non azzurro chiaro come nel primo tentativo,
  errore corretto**; marker pieno=taglia vista in training, vuoto=OOD; linea verticale tratteggiata
  grigia a $a=11.5$ con etichetta onesta "failure begins at $a=12$" (mai "capacity boundary"); nota
  in alto "All splits satisfy $D_{\mathrm{long}}>18=2\cdot3^L$" con il "18" in vermiglio `#D55E00`;
  nota piccola sotto l'asse x "split $=(a,46-a)$"; dimensione 6.2×2.6in (dentro il range 6.1–6.3 ×
  2.4–2.6 richiesto); solo griglia orizzontale `#DDDDDD`; niente spine alto/destra.
- `fig_oddtrain_attn_layer2.{pdf,png}` — heatmap dei pesi di attention reali, **solo layer 2** (non
  $S$, non layer 0), due pannelli $(8,38)$ risolto vs $(23,23)$ fallito, colormap magma, scala
  colore condivisa clippata al 99.5° percentile congiunto, triangoli arancioni `#E69F00` sui 4
  endpoint, confini di componente come linee bianche tratteggiate, un'unica colorbar condivisa,
  assi etichettati genericamente "node position along path" (non "query/attended-to node" come nel
  primo tentativo), tick solo su 1/confine/46, dimensione 6.1×2.5in con due pannelli quadrati.

**Dati sorgente**: `runs/report9/asym_chains_n46/n46_splitchainsgrid_seed{1000,2000}/asym_chains.json`
(figura 1) e `runs/report9/heatmaps/n46_splitchainsgrid_seed1000/heatmap_data.npz` (figura 2, campi
`a8__alpha1`/`a23__alpha1`). **Nota per chi riprende**: i numeri dietro la figura 1 (es.
$0.998\pm0.002$ ad $a=11$, ricalcolato dai JSON reali) differiscono leggermente dai numeri
arrotondati riportati nello scratch doc (`report/9/scratch_prof_experiments/scratch_prof_experiments.tex`,
che diceva $0.995\pm0.005$) — usare sempre i JSON come fonte di verità, non le tabelle degli scratch
doc. Gli script Python che generano queste due figure vivono solo nella cronologia di questa
sessione (non ancora promossi a script `.py` dedicati nel repo) — se in futuro Edoardo chiede di
rigenerarle o di scriverne di analoghe per §5.2/§5.3/§5.4, ricostruire dagli stessi file sorgente
seguendo le stesse regole di stile sopra, non da un vecchio tentativo con colori diversi.

**Cosa NON è ancora successo**: nessun testo è stato scritto in `paper_draft.tex` oltre a quanto già
c'era. Il paper è ancora Introduction + Related Work + Setting + Teorema 1 + placeholder. Le due
figure sono pronte per essere valutate da Edoardo e, solo su sua richiesta esplicita, incorporate in
una sezione scritta.

### Decisioni di design delle figure (da una consulenza AI esterna di Edoardo, sua trascrizione
### quasi verbatim — vale come riferimento per §5.2/§5.3/§5.4 quando si arriverà a scriverle)

Punto di partenza concettuale: la figura va progettata **attorno al claim**, non riciclando i
grafici dei report. Per la sezione beyond-threshold il claim forte è che il modello, allenato solo
su $\{3,5,7,9\}$, generalizza perfettamente alle taglie interlacciate mai viste, con rottura netta a
$a=12$ — e che **la rottura non coincide con l'attraversamento di $2\cdot3^L$**: tutti gli split
mostrati hanno già $D_{\mathrm{long}}>18$, quindi il fenomeno interessante è sbilanciato-vs-bilanciato,
non un attraversamento di soglia.

**Quattro alternative erano state proposte** (Claude ha implementato solo l'Opzione 1 + la heatmap
di attention, le due che Edoardo/l'AI consulente hanno indicato come prioritarie "per partire";
le altre restano disponibili se in futuro serve un taglio diverso):
1. **Exact match vs short-component size $a$** (quella scelta e implementata): marker pieni/vuoti
   per trained/OOD, linea spessa=media, linee sottili=seed singoli, soglia verticale tratteggiata
   con etichetta onesta ("exact match drops from $a=12$", **mai** "capacity boundary" — sarebbe
   scientificamente sbagliato), annotazione in alto sul fatto che $D_{\mathrm{long}}>2\cdot3^L$.
   Palette: blu `#0072B2` per dati/media, grigio chiaro per i seed, arancio/vermiglio `#D55E00` per
   eventuali soglie teoriche. Preferita perché mostra insieme OOD-generalisation, sbilanciato-vs-
   bilanciato e la rottura netta — e si può riusare identica per i cicli.
2. **Exact match vs diametro della componente lunga** ($x=D_{\mathrm{long}}$, linea verticale alla
   soglia teorica $2\cdot3^L$): più aggressiva nel dimostrare "il diametro non è la misura giusta"
   perché mostra pattern controintuitivo (diametro più grande → più facile in una parte del range),
   ma comunica meno direttamente il punto sbilanciato/bilanciato. Da tenere in mente per una
   figura di §5.4 o come figura alternativa/gemella nella stessa sezione.
3. **Heatmap seed×split** (righe=seed, colonne=$a$, colore=exact match, triangolini sopra le
   colonne allenate): meno immediata alla prima lettura ma **è la scelta giusta per i cicli**, dove
   c'è vera seed lottery (3/4 seed collassano, 1/4 risolve sempre nel continuous training) — una
   media con error bar sarebbe lì fuorviante (viola la regola già in vigore, errore 26 di questo
   file). Da usare quando si scrive §5.2.
4. **Schema illustrativo a due grafi** (es. $(8,38)$ vs $(23,23)$ disegnati esplicitamente con
   l'esito accanto): molto efficace per una talk/intro, ma seleziona solo due esempi — da abbinare
   all'Opzione 1 come inset, non da usare da sola come evidenza principale.

**Per la heatmap di attention (1.b)**: mostrare **solo** $\alpha$ del layer 2 (non $S$, non il layer
0, non entrambi i layer) — nei report c'era troppa carne al fuoco per una figura da paper. Due
pannelli affiancati (split risolto vs split fallito), righe=nodo query $i$, colonne=nodo attenzionato
$j$ (l'enfasi sugli endpoint deve comparire come **bande verticali**, cioè colonne, non righe).
Colormap magma o viridis, **mai jet**. Scala colore condivisa fra i due pannelli (altrimenti non si
può confrontare l'intensità), clippata a un percentile alto comune (99.5°) per non far sparire il
contrasto per pochi outlier — dichiarare sempre il clip in caption. Confini di componente come linee
bianche tratteggiate sottili, endpoint marcati con piccoli triangoli sopra la matrice (non linee
che attraversano tutta la heatmap, sembrerebbero dati).

### Metodologia sui seed: nuova policy per gli esperimenti principali del paper

> Decisione presa 2026-08-23, da applicare da qui in avanti a ogni claim centrale del paper
> (non ai soli quattro esperimenti nominati sotto, che sono il caso concreto discusso finora).

**Regola**: ogni esperimento comportamentale che sostiene un claim principale del paper va portato
a **almeno 5 seed indipendenti** (standard usato anche da Sanford et al. 2024, mean±std su 5 seed).
Per condizioni che mostrano variabilità seed-to-seed sostanziale (finora: i cicli, dove nel
continuous training 3/4 seed collassano e 1/4 risolve sempre — vera "seed lottery", non rumore),
salire a **10 seed** per poter quantificare la frazione di seed che risolve, non solo descriverla
aneddoticamente.

**Come riportare i numeri** (per non nascondere bimodalità dietro una media):
- L'unità di incertezza è il **seed/modello**, non il singolo grafo di test — mai calcolare uno
  std "finto" su tutti i $N_{\mathrm{seed}}\times N_{\mathrm{test}}$ grafi insieme.
- Nei grafici: **linee sottili per seed individuale + una linea spessa per la media**, mai solo la
  media nuda (questa è già la regola 26 di questo file, qui solo ribadita con lo standard preciso).
- Se il fenomeno è bimodale (es. cicli), scriverlo esplicitamente in prosa ("3/10 seed risolvono lo
  split bilanciato, 7/10 collassano") invece di un mean±std che a metà strada sembra un risultato
  intermedio quando invece nessun singolo modello si comporta così.
- Tabelle: mean ± std sui seed in una colonna; l'appendice può portare il dettaglio per-seed se
  interessante (come già fa il progetto per i risultati con seed lottery, es. Report IX Thread B).

**Stato attuale dei seed per i quattro esperimenti principali (verificato 2026-08-23,
`runs/report9/n46_train/`)**:

| Esperimento | Seed esistenti | Mancano per 5 | Script sbatch pronti |
|---|---|---|---|
| Two-path continuous (§app:full-distribution-control) | 1000,2000,3000,4000 | 5000 | `scripts/r9_n46_splitchains_seed5000.sbatch` |
| Two-cycle continuous | 1000,2000,3000,4000 | 5000 | `scripts/r9_n46_splitcycles_seed5000.sbatch` |
| Two-path odd/grid (§sec:beyond-threshold, Figure 1/2) | 1000,2000 | 3000,4000,5000 | `scripts/r9_n46_splitchains_grid_seed{3000,4000,5000}.sbatch` |
| Two-cycle odd/grid | 1000,2000 | 3000,4000,5000 | `scripts/r9_n46_splitcyclesgrid_seed{3000,4000,5000}.sbatch` |

Tutti e 8 gli script sono stati **preparati da Claude per templating** (copiati dagli script già
esistenti per gli altri seed, solo seed/job-name/path sostituiti) ma **non lanciati** — bloccato
dall'outage HPC (vedi §11). Quando l'accesso torna, lanciarli con `sbatch scripts/<nome>.sbatch`
(uno per volta o in blocco, rispettando il tetto job in coda di §6); a fine training rigenerare
tabella/Figura 1/Figura 2 di `paper_draft.tex` con gli script Python usati sopra (vedi i comandi
`python3` inline nella sessione che ha prodotto le figure, non ancora promossi a script `plot_*.py`
dedicati — se questa figura sopravvive nella versione finale del paper vale la pena crearne uno).

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

> **✅ Accesso HPC ripristinato (2026-08-24).** L'outage SSH dal 2026-08-19 è risolto (memoria di
> progetto e questo callout rimossi). I due job già sottomessi prima dell'outage — Slurm **619502**
> (`r10ft3way`, fine-tuning mirato) e **619513** (`r10ft3wayfull`, fine-tuning sull'intera
> distribuzione K=3) — **si analizzano in un secondo momento, su richiesta esplicita di Edoardo**:
> non controllarli/pull-arli di propria iniziativa in una nuova sessione, la checklist per farlo
> resta comunque qui sotto per quando servirà.
>
> **Priorità immediata invece (2026-08-24): portare a 5 seed i quattro esperimenti principali del
> paper** (§0, "Metodologia sui seed"). Otto script sbatch sono già pronti in `scripts/` (creati
> 2026-08-23, mai lanciati):
> ```
> scripts/r9_n46_splitchains_seed5000.sbatch
> scripts/r9_n46_splitcycles_seed5000.sbatch
> scripts/r9_n46_splitchains_grid_seed3000.sbatch
> scripts/r9_n46_splitchains_grid_seed4000.sbatch
> scripts/r9_n46_splitchains_grid_seed5000.sbatch
> scripts/r9_n46_splitcyclesgrid_seed3000.sbatch
> scripts/r9_n46_splitcyclesgrid_seed4000.sbatch
> scripts/r9_n46_splitcyclesgrid_seed5000.sbatch
> ```
> **Come lanciarli** (Edoardo, da terminale HPC dopo `git pull`): sono 8 job GPU indipendenti, ognuno
> ~16h (`long_gpuh200`, vedi §6) — sottometterli tutti insieme è dentro il tetto job-in-coda di §6
> (~25-30 su QOS `normal`), quindi:
> ```bash
> cd ~/transformer-for-graphs && git pull
> for f in scripts/r9_n46_splitchains_seed5000.sbatch \
>          scripts/r9_n46_splitcycles_seed5000.sbatch \
>          scripts/r9_n46_splitchains_grid_seed{3000,4000,5000}.sbatch \
>          scripts/r9_n46_splitcyclesgrid_seed{3000,4000,5000}.sbatch; do
>   sbatch "$f"
> done
> ```
> A fine training ciascuno produce da sé tabelle/figure aggiornate nelle stesse cartelle degli altri
> seed (`runs/report9/asym_chains_n46/`, `heatmaps/`, `mechanistic/`, `stagewise/`) — su HPC
> `git add runs/report9/ out/*.out && git commit ... && git push`, poi in locale `git pull`.
> **Dopo**: (a) rigenerare le tre figure di questa sessione (`paper_figures/fig_oddtrain_*`) con i 5
> seed invece di 2 — stessi script Python descritti sopra, stesse regole di stile/colore, solo
> estendendo i seed usati; (b) fare lo stesso per l'heatmap sui cicli, dove la seed lottery del
> continuous training (3/4 collassano, 1/4 risolve) e la variabilità già vista nell'odd-cycle
> training (2 seed in disaccordo netto) rendono quella heatmap probabilmente il risultato più
> interessante di tutti una volta a 5 seed.

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
esatto); sbatch: `scripts/r10_finetune_threeway_splitchains_seed1000.sbatch`; report di analisi
in `report/10/` (in corso).

**Secondo esperimento collegato, più ampio**: `finetune_readout_threeway_full.py` (stesso
metodo — solo `sim_scale`/`sim_bias`, trunk congelato — stesso checkpoint di partenza) ma allena
sull'**intera distribuzione** degli split a tre componenti
(`generate_path_union_graph(min_paths=3, max_paths=3)`, ogni combinazione di taglie, non solo le
due celle nominate), per **200.000 step** (4× il primo esperimento). Domanda diversa da quella
del fine-tuning mirato: non "riesco a correggere queste due celle specifiche" ma "l'esposizione a
tutto lo spazio K=3 fa imparare al solo read-out una regola generale a tre componenti?" — la
stessa domanda del §5.3 del paper, ma con budget molto più lungo del generico K∈{3..6} già
provato (esperimento 6 di `scratch_prof_experiments.tex`, 3000 step, non aveva funzionato). Traccia
sia le due celle nominate (per confronto diretto con il primo esperimento) sia una metrica
aggregata su grafi K=3 generici ad ogni valutazione. Sbatch:
`scripts/r10_finetune_full3way_splitchains_seed1000.sbatch`. Entrambi gli esperimenti si
graficano con lo stesso script, `plot_finetune_readout_threeway.py --run_subdir {finetune_readout_threeway,finetune_readout_full3way}`.

**Checklist per quando l'accesso HPC torna** (vedi il blocco ⚠️ a inizio sezione): (1) su HPC,
`squeue -u 3352759` per vedere se 619502/619513 sono ancora in coda/running o già finiti; (2)
`cat out/r10ft3way_619502.out` e `out/r10ft3wayfull_619513.out` per leggere l'esito (occhio ai
marker `FAILED:` — ogni step della batteria post-finetuning è indipendente e non blocca gli
altri se fallisce); (3) se completati, su HPC `git add runs/report10/ out/r10ft3way_*.out
out/r10ft3wayfull_*.out && git commit ... && git push`; (4) in locale `git pull`, poi
`python plot_finetune_readout_threeway.py --tag n46_splitchains_seed1000_threeway
--run_subdir finetune_readout_threeway` e lo stesso con `--tag
n46_splitchains_seed1000_full3way --run_subdir finetune_readout_full3way`; (5) riempire i
blocchi `[PENDING]` di `report/10/transformer_for_graphs_10.tex` con l'analisi vera, guardando
sia le curve sia (se interessante quanto visto nei log parziali del job 619502 — vedi cronologia
chat, non ripetuta qui) l'eventuale trade-off fra le due celle target quando un'unica coppia
scale/bias condivisa deve correggerle entrambe.

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
