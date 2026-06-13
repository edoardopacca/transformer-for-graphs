# Istruzioni del progetto — handoff per Claude

Documento di riferimento per chiunque (incluso Claude in una nuova chat) riprenda
questo progetto. Leggere **tutto** prima di agire.

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

**Il nostro report 3 ha due parti, con due conclusioni:**
- **Parte I — il data lever non regge per lo standard transformer.** Riproduciamo
  il setup del paper a n=20: la generalizzazione OOD è **dominata dal seed**, non
  dal filtro di diametro; restricted e unrestricted generalizzano a tassi
  statisticamente indistinguibili. La "transizione di fase" pulita del paper è,
  nelle nostre mani, un evento ad alta varianza (1 seed fortunato su ~3).
- **Parte II — caratterizziamo il muro di capacità** (distanza ≈9 = `3^2`):
  cicli, diagnostica pairwise su 2chain, conteggio componenti, e lo sweep
  profondità→reach (`d* ≈ 3^L`). Più esperimenti spettrali/Laplaciani in corso.

**Report 4** (`report/4/transformer_for_graphs_4.tex`, compilare da `report/4/`):
nuovo filone su **architettura migliore** + **nozione di difficoltà**. Tre thread:
(1) il **read-out di similarità** su Roberta aiuta il *reach a lunga distanza*
(path_union, chain_plus OOD: 0.78→1.00) **ma NON il bottleneck** (barbell 0→0) →
il limite è il *trunk*, non la testa; la **loss Laplaciana è inerte** a
convergenza (il similarity readout porta già la Dirichlet energy a ~0), λ
droppato. (2) **diametro vs spectral gap**: questione APERTA — il muro 3^L è reale
(chain_plus lo mostra anche a n=20: reach 1.0 fino a 9, crollo, recupero), ma
barbell-vs-1chain/expander suggeriscono che il *gap/bottleneck* aggiunga. Ipotesi
di lavoro: "diametro **E** gap", non uno solo. **NON** scrivere "il diametro è la
nozione sbagliata" (prematuro). (3) **spectral gap**: `expander_var` è confuso
dal confound *densità* (grado alto = OOD su ER sparso), non isola; il test pulito
è **parallel_paths** (distanza fissa, k cammini variati, resistenza = ℓ/k).
C'è anche un **terzo modo di fallire** oltre reach/cut: l'**euristica del grado**
(2cliques sbaglia coppie a d=1). Tutto su **1 seed** finora → preliminare, in
attesa dei seed rimanenti + n40 + parallel_paths.

---

## 2. Workflow e REGOLE OPERATIVE (non negoziabili)

- **Git: Claude NON committa e NON pusha mai.** Claude modifica i file in locale e
  *consegna i comandi git all'utente*, che li esegue. (Vedi memoria
  `feedback_no_git_commits`.)
- **Branch: si lavora su `main`.** Non creare branch nuovi se non richiesto.
- **HPC (Bocconi): MAI calcolo sul nodo di login.** Qualsiasi cosa pesante va via
  `sbatch` (o `srun` su compute node). Sul login solo git/ls/tail/cat.
- **Eval leggeri**: vanno bene **in locale sul Mac** (MPS) — es. `eval_cycles_n40.py`,
  `diagnose_2chains_pairwise.py`. Gli eval che toccano i **checkpoint su HPC**
  (i `.pt`, che NON sono in git) → o si `scp`-ano i `.pt` in locale, o si lancia
  l'eval via `sbatch` su HPC.
- **Cosa va in git**: solo artefatti piccoli — `.json`, `.png`, `history.json`,
  log `out/*.out`, sorgenti, `.tex`. **Mai** i `.pt` (gitignored: `*.pt`) né gli
  artefatti LaTeX (`*.aux *.fls *.fdb_latexmk *.out *.synctex.gz`, e `*.log`).
- Ciclo tipico: *edit in locale → utente push → `git pull` su HPC → `sbatch` →
  a fine job: su HPC committa json/png e push → in locale `git pull`*.
- Ordine push per evitare "divergent branches": se sia locale che HPC hanno
  commit, pushare prima da un lato, poi `git pull` dall'altro.

---

## 3. Preferenze dell'utente (cosa piace / cosa no)

**Piace:**
- Analisi onesta **da ML researcher**: separare segnale da rumore, ammettere
  risultati negativi/rumorosi, non vendere come pulito ciò che non lo è.
- **Tabelle** con dati utili + **figure chiare**; didascalie informative.
- Figure che **rappresentano fedelmente** i dati (per esiti bimodali → curve
  per-seed, NON media+banda che nasconde la bimodalità).
- Config **fedele al paper** quando riproduciamo ("tutto uguale a loro"), e
  **segnalare i conflitti** (es. single-head del paper vs 4-head di n40big).
- Proporre esperimenti, ragionare sul "perché", spiegare in modo semplice quando
  chiede "non capisco".
- Concisione: *"non scrivere più di così"* — non gonfiare.

**Non piace:**
- Figure fuorvianti (media+banda su esiti bimodali; il "generalization rate"
  boxplot/bar non lo voleva → preferisce la **tabella per-seed**).
- Numeri/figure inventati o messi **senza avere i dati reali** in mano.
- Spiegazioni vaghe; risultati confusi spacciati per chiari.

---

## 4. Errori già fatti — DA NON RIPETERE

1. **Suggerito `python ...` sul nodo login HPC** (calcolo sul login è vietato).
   → sempre `sbatch`.
2. **Stavo per fare grafici/numeri senza avere i dati** (lavoravo dall'output del
   terminale). → prima `git pull`/estrai i dati reali, poi scrivi.
3. **Creato un branch git** quando voleva `main`.
4. **Figure media±banda** su esiti per-seed bimodali (fuorvianti) → voleva
   per-seed e confronto restricted-vs-unrestricted.
5. **Time limit SLURM troppo corti** (reach L=2/3/4 killati a 6h; servivano ~14h;
   nodi a volte ~3× più lenti → TIMEOUT). Controllare s/step e dimensionare.
6. **`history.json` scritto solo a fine training** → job killati non la scrivono
   (recuperare via eval del `last.pt`, oppure rilanciare con time limit adeguato).
7. **Metrica embedding per-distanza (crossing)** rumorosa: l'ho presentata troppo
   pulita. Il read-out è *lineare*, quindi `‖h_i−h_j‖²` non è la quantità decisiva.
8. **`\le` nei titoli matplotlib** → errore mathtext; usare unicode `≤`.
9. **`\texttt{#1}` con underscore** in LaTeX → "Missing $"; evitare di stampare
   path con `_` fuori da `\verb`.
10. **Partizioni `debug_*` richiedono `--qos=debug`** (altrimenti "QOS not
    permitted"). Tipo: `sbatch -p debug_gpunew --qos=debug ...`.
11. **`QOSMaxSubmitJobPerUserLimit`**: c'è un tetto sui *task sottomessi* insieme
    (in coda+run), ~25–30 sulla QOS `normal`. Array enormi (`--array=11-81`, 71
    task) vengono RIFIUTATI. → sottometti a **blocchi** (per-seed). Il `%N`
    (ArrayTaskThrottle) limita i *running* concorrenti, **non** i sottomessi.
12. **Multi-partizione NON permesso** per questo account: `--partition=a,b,c` dà
    "Multiple partition job request not supported when a partition is set in the
    association". → scegli **una** partizione; trova la migliore con
    `squeue --start -u <id>` (ETA) + `sprio` (priorità) + `sinfo`. Spostare un job
    PD: `scontrol update JobId=<id> Partition=gpunew`.
13. **debug = 15 min**: insufficiente per il reeval (~17 ckpt, ~25 min) → TIMEOUT.
    Usa `gpunew` per i job > 15 min.
14. **png committati bloccano `git pull`** se rigenerati in locale (plot_families
    riscrive i png tracked → conflitto). NON lanciare `plot_families.py` *prima*
    del pull; se bloccato: `git restore runs/` poi `git pull` poi plot. (L'utente
    **NON** vuole gitignorare i png.)
15. **`P^L` proxy: usa `3^L`, non `L`**. Il modello a L layer raggiunge distanza
    `3^L` (matrix-powering), non `L`. `diffusion_reach(adj, n_steps)` va chiamato
    con `n_steps = 3**L` (=9 per L=2), altrimenti `P^2` è ~0 oltre 2 hop.
16. **`expander_var` non isola il gap**: variare il grado cambia gap *e* densità
    → grafi densi sono OOD per modello su ER sparso. Per isolare il gap usa
    **parallel_paths**. La resistenza effettiva grezza non è confrontabile tra
    grafi (dipende dagli archi) → usa il ranking, o `P^L` (probabilità in [0,1]).
17. **matplotlib xticks**: su bar plot con poche barre matplotlib mette tick a
    0.5 — settare `ax.set_xticks(...)` espliciti con le distanze intere.
18. **`long_gpunew` ha solo 2 nodi** (condivisi con gpunew); un job lungo (es.
    `--time=36h`) **non parte MAI** lì (`StartTime=Unknown`): lo scheduler non
    trova una finestra di 36h libera. → usa **`gpunew`** e **non gonfiare il
    `--time`**: chiedere 12h invece di 36h rende il backfill molto più facile (i
    job corti come reeval/parpaste partono subito). n20 ~1h40, n40 ~6-8h: usa
    `--time` 6h / 12h. Spostare un job PD: `scontrol update JobId=<id>
    TimeLimit=12:00:00 Partition=gpunew` (ridurre il time è permesso, aumentarlo no).
19. **MAI la metrica `d*` (reach radius = max distanza con acc ≥ soglia).**
    L'utente la detesta e l'ho bandita: un singolo numero di "massima distanza
    raggiunta" è fuorviante perché **nasconde ciò che succede in mezzo** — un
    modello può azzeccare d=36 ma sbagliare d=14, e l'accuracy per-distanza è
    **non-monotona** (crolla nel mezzo, recupera ai bordi). → Mostrare SEMPRE
    l'**evoluzione completa** dell'accuracy lungo la distanza (curva/tabella
    per-distanza), non collassarla in un solo numero. Quel che conta è *come*
    evolve, non il massimo raggiunto.
20. **Il report è per la PROF: deve VEDERE i risultati, non solo testo.** Ogni
    affermazione importante va con una **tabella/figura**. Le tabelle: **per-seed**,
    etichettate (n20/n40, ER/mixed, linear/similarity), e **solo le tipologie di
    grafo interessanti** (NON quelle con tutti 1.00 / tutti 0 — non significative).
    "Guardale tutte, riporta le interessanti." Stile da **studente al primo progetto**:
    prosa semplice, niente toni da venditore; spiega i meccanismi in modo banale.
21. **Ogni caption deve dire QUALE MODELLO e QUALE TEST SET.** L'utente si lamenta
    se non si capisce: modello = *trainato su cosa* (ER vs mixed), *quanti seed*,
    *quale dimensione* (L=2, d_model=512), e su *quale famiglia/pool* sono fatte le
    medie. E dire se i numeri sono **media o mediana** e di quali seed.
22. **`barbell_var` NON è un esperimento pulito sul gap.** Il gap di un barbell è
    intrinsecamente minuscolo (n20: 0.006–0.019; n40: 0.0008–0.005) ed è **a U** nella
    clique_size; e la clique_size cambia anche **diametro** (3↔19) e **densità** delle
    cricche. Bucketare per gap mischia barbell diversissimi → fuorviante. La difficoltà
    segue **densità cricche + lunghezza ponte**, NON lo scalare gap. → presentarlo
    **per diametro/clique-size**, non per gap.
23. **`expander_var` è confuso dalla densità** (variare il grado alza gap *e* densità →
    OOD): la salita a gap piccolo è reale, la **discesa a gap grande è il confound
    densità** (più netto a n40, trainato più sparso). NON isola il gap.
24. **Il probe `parallel_paths` PULITO funziona (risultato positivo!).** Il vecchio
    (su checkpoint n64 reach) era inconcludente per due confound: **padding isolato**
    (canvas vuoto) e **terminali di grado k** (cambiano col k). Il nuovo
    `eval_parallel_paths_clean.py` li toglie: **distanza fissa ℓ**, **canvas riempito**
    (path sparso, grado medio ~2), **grado terminali fissato a 4** (padding con foglie);
    varia solo k = #route (resistenza ℓ/k). Esito: **oltre la capacità, più route →
    connessione confermata** (k=1 fallisce ~0.55, k=2→0.93, k=3→1.00 a distanza fissa).
    Quindi il bottleneck/resistenza **è isolabile** con un probe costruito (anche se sui
    grafi naturali resta collineare col diametro).
25. **Il muro `3^L=9` è ARCHITETTURALE, non di dati.** 2chains/1cycle **sono nel
    training mixed** (visti ~10⁸ volte, un solo grafo a meno di permutazione) e il
    modello L=2 **comunque** non connette coppie oltre 9 hop → prova pulita che è un
    limite di profondità, non di copertura dati. Buon argomento per il report.
26. **Mappa di difficoltà: due assi SEPARABILI = diametro (capacità) + densità
    (mean degree vs training).** Lo **spectral gap NON è separabile** sui grafi
    naturali: corr(diametro, log-gap) = **−0.92** (un grafo sparso lungo ha diametro
    grande E gap piccolo). Non c'è l'angolo "diametro grande + gap grande" → non si
    distinguono osservativamente. Solo il **barbell** rompe la collinearità, ma essendo
    denso finisce sull'asse densità.
27. **reach vs cut sono fallimenti DIVERSI.** reach = il muro `3^L` (netto). cut
    (over-connect) = un **avvallamento dolce, model-specific, NON a 9** (verificato:
    un modello matrix-power *pulito* taglia perfetto — controllato con un oracolo, vedi
    `eval_near_miss_cut.py`). La similarity sistema **entrambi**; la loss Laplaciana li
    **scambia** (compra reach, spende cut).
28. **Comportamento OLTRE il muro varia per famiglia:** chain_plus/path_union hanno un
    **floor ~0.55 + risalita**; 2chains/1cycle/2cycle **crollano a 0** (niente risalita).
    Stessa capacità 9, code diverse. A **n20** le famiglie strutturate sono troppo corte
    per esporre il muro (il mixed le risolve); il muro **ritorna a n40**.
29. **Due tipi di tabella, da NON confondere:** (a) **per-distanza-di-coppia** =
    accuracy sulle coppie a shortest-path distance d (reach); (b) **per-diametro-del-
    grafo** = exact-match dell'intero grafo al crescere del diametro. Scriverlo esplicito.
30. **Trappole matplotlib/LaTeX ricorrenti:** `\le`/`\ge` nei titoli matplotlib →
    usare unicode ≤ ≥ (già errore #8, ci ricasco). `\multirow` in LaTeX richiede il
    package → usare il formato a blocchi `\multicolumn{n}{l}{\emph{...}}` (come il resto
    del report). Il **degree heuristic** è un *density-OOD su 2cliques*: fixato solo dal
    **mixed training** (i dati), NON dal read-out né dalla loss.
31. **Quando un'analisi si rivela confusa/confondata, CORREGGILA onestamente** (es.
    barbell_var-by-gap) invece di tenere una tabella fuorviante. L'utente apprezza
    l'onestà e nota i confound — preferisce un risultato negativo pulito a uno positivo
    sporco.

---

## 5. Cosa Claude deve sapere in una nuova chat

- Repo locale: `~/transformer-for-graphs` (Mac). Su HPC: `~/transformer-for-graphs`.
- HPC alias ssh: `hpc`. Utente: `3352759`. Env conda: **`graph_tf`**.
- Report: `report/1/` (solo PDF, niente sorgente), `report/2/`, `report/3/`,
  `report/4/transformer_for_graphs_4.tex`. **Compilare dalla cartella del report**
  (i path figure usano `\graphicspath{{../../}}`; `\includegraphics{runs/...}`).
  Due passate di `pdflatex` per i `\ref`/`\part`. Tutti i .tex hanno l'helper
  `\figorbox{path}{width}` (fallback se la figura non è ancora pullata).
- I checkpoint `.pt` stanno **solo su HPC** (gitignored). I risultati piccoli
  (json/png) sono in `runs/...` e versionati.
- Memoria persistente: **niente git da Claude**.

**File/script chiave:**
- `model.py`: `ModelConfig`, `GraphConnectivityTransformer` (read-out `linear` o
  `similarity`, metodi `hidden_states`/`embeddings`/`forward_and_embeddings`),
  `RobertaGraphTransformer`, `GraphBinaryClassifier`, `laplacian_smoothness`.
- `data.py`: generatori (sez. 8).
- `experiments2/`: `reproduce_paper_n20.py` (Parte I, `--arch minimal|roberta`),
  `retrain_and_test_er_n40_big.py` (n40big), `train_reach_depth.py` (reach),
  `train_components_classifier.py` (conteggio componenti), `train_laplacian.py`
  (Esp. B), `ood_evaluation.py`.
- Root: `eval_cycles_n40.py`, `diagnose_2chains_pairwise.py` (`--round 1|2`),
  `eval_reach_checkpoint.py`, `eval_reach_by_distance.py`, `analyze_embeddings.py`,
  `plot_repro_paper_figures.py`, `plot_repro_rate.py`,
  `plot_repro_roberta_perseed.py`, `plot_repro_seed_examples.py`, `plot_reach_law.py`.
- `scripts/*.sbatch`: i lanci SLURM corrispondenti.

**File chiave Report 4 (filone difficoltà/architettura):**
- `experiments2/train_families_n20.py`: training roberta `--families er|mixed`,
  `--readout linear|similarity`, `--lambda_lap` con warmup, **`--n_nodes`/`--p`
  parametrici** (default 20/0.08; per n40 passare `--n_nodes 40 --p 0.05`). Nome
  cartella: `n{n}_{fam}_roberta_{readout}_lam{λ:g}_seed{S}`.
- `eval_families.py`: valuta UN checkpoint su ~14 famiglie; auto-detect
  arch/readout dallo state_dict. Output `families_eval.json` + 3 png
  (`capacity_per_distance`, `by_diameter`, `by_spectral_gap`). Metriche per
  famiglia: `exact, pairwise, reach_acc, disc_acc, per_dist, by_diam, by_gap`.
- `plot_families.py`: rigenera i png dai json (no GPU). Usare in locale dopo pull.
- `eval_difficulty_map.py` + `scripts/eval_difficulty_map.sbatch`: per OGNI grafo di
  un pool che copre il piano dumpa `(diameter, gap, mean_degree, ncomp, exact,
  pairwise)` → `<ckpt>/difficulty_map/difficulty_map.json`. Eval-only sui ckpt linear.
- `eval_near_miss_cut.py` + `scripts/near_miss_cut.sbatch`: test del **cut** (spezza una
  catena togliendo 1 arco; cut-acc vs near-miss distance + reach vs distance). Valida
  l'oracolo matrix-power (taglia perfetto).
- `eval_parallel_paths_clean.py` + `scripts/parallel_paths_clean.sbatch`: probe **gap
  confound-free** (distanza fissa, grado terminali fisso=4 con foglie, canvas riempito,
  varia k=#route). Output `<ckpt>/parallel_paths_clean/parallel_paths_clean.json`.
- `analyze_parallel_paths.py` + `scripts/parallel_paths.sbatch`: la VECCHIA versione (su
  reach n64) — **inconcludente, NON usare** (vedi errore 24); rimossa dal report.
- `scripts/`: `train_families.sbatch` (n20, array ordinato per seed, eval auto),
  `train_families_n40.sbatch`, `reeval_families.sbatch` (ripassa tutti i ckpt).
- `notes/note_difficolta.tex`: spiegazione divulgativa (it) di tutto il filone.
- Checkpoint riusati: roberta n20 unrestricted (8 seed, `runs/report3/repro_paper_n20_roberta/`)
  = baseline `ER/linear`.

---

## 6. Setup HPC

- Partizioni GPU H200: `gpuh200` (cap 24h), `long_gpuh200` (cap 72h),
  `debug_gpuh200` (15 min, serve `--qos=debug`). H100: `gpunew`/`long_gpunew`/
  `debug_gpunew`. **`gpuh200` e `long_gpuh200` condividono gli stessi nodi fisici**
  (gnode09–16); H100 sono un pool separato (gnode05–07 circa).
- **Cap per-utente `QOSMaxGRESPerUser = 4 GPU` contemporanee** (verosimilmente
  *globale* sulla QOS `normal`). Quindi al massimo 4 job in R insieme; gli array
  più grandi accodano (`Resources`/`Priority`/`QOSMaxGRESPerUser`).
- sbatch standard: `--account=3352759`, `--gpus=1`, `--cpus-per-task=16`
  (24 se data-bound, es. filtro diametro = APSP per grafo), `--mem=40G`,
  env-var `OMP_NUM_THREADS=1` ecc., `source ~/.bashrc; conda activate graph_tf`.
- **Tempi indicativi** (batch 1000, 1M step): n=20 unrestricted ~1h40m;
  n=20/n=40 con filtro diametro (CPU-bound APSP) ~3–5h; n=64 reach più layer
  ~5–11h (L=4 ~11h → time limit 14h). Nodi condivisi possono essere ~3× più lenti.
- Quota **home**: limitata (~180–200G); le cache HF/pip (`~/.cache`) e gli env
  conda la riempiono. Se "Disk quota exceeded": pulire `~/.cache`, `conda clean`.

---

## 7. Architetture (in `model.py`)

Config comune "big": `d_model=512`, `d_ff=2048`, `n_layers=2`, `~6.3M` params,
attention **normalized-ReLU** `α = (1/n)·ReLU(QKᵀ/√d_h)` (variante del paper) o
softmax; GELU FFN; nessun mask causale; nessun positional encoding (l'identità in
A+I fissa i token). Ottimizzatore AdamW, peak lr `1e-4`, weight decay `1e-4`,
cosine + warmup, bf16. Input = `A + I` (self-loop), target = matrice di
connettività `R` (`R_ij=1` se i,j stessa componente).

- **`GraphConnectivityTransformer`** — modello principale n×n. Variante
  **"minimal / A.1-style"**: pre-LayerNorm, read-out lineare simmetrizzato.
  Opzione `readout="similarity"`: `R̂_ij = scale·cos(h_i,h_j) + bias` (vista
  spettrale: connessione = similarità embedding).
- **`RobertaGraphTransformer`** — variante "RoBERTa-faithful" (post-LayerNorm,
  dropout 0.1, init `N(0,0.02)`), usata nella riproduzione del paper.
- **`GraphBinaryClassifier`** — stesso trunk + mean-pool + 1 logit, per il task
  binario "1 vs 2 componenti".
- **`laplacian_smoothness(H, A)`** = `Tr(HᵀLH)/#archi` = `Σ_{(i,j)∈E}‖h_i−h_j‖²`,
  loss ausiliaria spettrale (Esp. B).

Numero di teste: **single-head** nella riproduzione paper (App. D.1); **4 teste**
nei modelli n40big.

---

## 8. Dati (generatori in `data.py`)

Tutti i generatori producono adiacenza *senza* self-loop; i self-loop si
aggiungono con `add_self_loops`. Target via `compute_connectivity_matrix`;
distanze via `compute_all_pairs_shortest_paths` (APSP, scipy; serve `scipy`).

**Composizione training `mixed` (Set B), da `train_families_n20.py`:** stream
ONLINE (nessun set fisso) uniforme su **9 famiglie**: `er, er_blocks, clique_blocks,
path_union, 2chains, 2cliques, 1cycle, 2cycle, 1chain`. **Tenute FUORI (eval/OOD):
barbell, expander, chain_plus.** Batch 1000 × 10⁶ step ≈ 10⁹ grafi totali. NB: le
famiglie deterministiche (1chain, 1cycle, 2chains, 2cliques, 2cycle) sono UN solo
grafo a meno di permutazione → viste ~10⁸ volte ripermutate. ER trainato: a n20
`ER(20,0.08)` (grado~1.6), a n40 `ER(40,0.05)` (grado~2.0).

- `generate_er_graph(n, p, rng)` — Erdős–Rényi.
- `generate_two_chains_graph(n, k)` — due path disgiunti da k (n=2k).
- `generate_two_cliques_graph(n, k)` — due cricche disgiunte da k.
- `generate_one_chain_graph(n)` — un path su tutti gli n nodi.
- `generate_one_cycle_graph(n)` — un ciclo `C_n`.
- `generate_two_cycles_graph(n, k)` — due cicli `C_k` disgiunti.
- `generate_path_union_graph(n, rng, max_paths=4)` — unione di `k`∈{1,2,3,4}
  (uniforme) path disgiunti che partizionano gli n nodi; ~25% sono un singolo
  path (distanze fino a n−1). Usato per il reach experiment.
- `generate_blocks_graph(n, rng, kind="er"|"clique")` — k∈{1..4} blocchi internamente
  connessi (ER o clique). `generate_barbell_graph(n, rng, clique_size=None)` — due
  clique + ponte (bottleneck, gap minuscolo). `generate_random_regular_graph(n, rng,
  degree=3)` — expander (gap grande, diam piccolo). `generate_chain_plus_graph(n, rng)`
  — catena lunga + componente staccata (espone il muro 3^L anche a n piccolo).
  `generate_parallel_paths_graph(n, n_paths, path_len)` — 2 terminali + k cammini
  disgiunti (distanza fissa = path_len, resistenza = path_len/k).
- **Misure strutturali** (`data.py`): `compute_spectral_gap(adj)` (Fiedler norm.,
  primo autovalore >0), `effective_resistance(adj)` (R(i,j) via L⁺; scala dipende
  dagli archi → non confrontabile tra grafi), `diffusion_reach(adj, n_steps)` (`P^L`
  row-stoch., probabilità ∈[0,1]; usare `n_steps=3^L`).
- **`model.py`**: `RobertaGraphTransformer` ora supporta `readout="similarity"`;
  `attention_maps(x)` estrae i pesi di attention reali (per il mixing/Jacobiana).

---

## 9. Scelte sperimentali e perché

- **n=40, p=0.05** (data lever, Report 2): a questa densità i cutoff D≤7/9/11
  accettano frazioni ben diverse (~14/55/84%) → confronto sensato. A n=20 i cutoff
  sono quasi vacui.
- **n=20, p=0.08** (riproduzione paper): è la loro config esatta (§3.3). Test su
  2Chain(20,10) e 2Clique(20,10).
- **Niente restrizione di diametro nel reach experiment**: serve l'opposto — grafi
  con distanze *lunghe* (path-union a **n=64**) per misurare se la reach arriva a
  `3^L` (a n=40 le distanze ER si fermano ~20, troppo corte per testare `3^3=27`).
- **Due round (exp2)** a n=40: secondo seed indipendente per esporre la varianza
  run-to-run.
- **Read-out di similarità + loss Laplaciana** (Esp. B): la connettività è
  spettrale (kernel del Laplaciano `L=D−A` = indicatori delle componenti). Testiamo
  se una bias spettrale sposta il muro `3^L` (matrix-powering = locale; spettrale =
  globale, non limitato dal diametro).
- **single-head vs 4-head, minimal vs RoBERTa**: il paper non fissa l'architettura
  esatta (A.1 = idealizzata pre-norm; D.1 = "adopt RoBERTa" = post-norm/dropout/init
  0.02). Quindi proviamo **entrambe** e lo diciamo.

---

## 10. Lessico del report (usare questi termini)

- **exact-match accuracy**: frazione di grafi con matrice `R̂` esatta su *tutte* le
  coppie (metrica del paper, §3.3).
- **pairwise accuracy**: frazione di coppie di nodi predette correttamente.
- **per-distance pairwise / reach(d)**: pairwise condizionata alla shortest-path
  distance `d`; "reach" = su coppie *connesse* (target 1).
- **d\***: exact-reach radius = massima `d` con reach ≥ 0.99.
- **capacity 3^L**: distanza massima risolvibile da un modello a L layer.
- **within-capacity / beyond-capacity**: coppie a `d ≤ 3^L` / `d > 3^L`.
- **data lever**: restringere il training a grafi within-capacity (diam ≤ `3^L`).
- **diameter filter** `D≤7/9/11`, **unfiltered**, **exp2** (secondo round).
- **in-distribution** (ER di training) vs **OOD** (2Chain, 2Clique, 1cycle, 2cycle).
- **n_active**: numero di nodi non isolati in un grafo strutturato dentro un canvas
  n×n (es. 2chain padded).
- **generalisation rate**: frazione di seed che generalizzano (exact > soglia), con
  CI di **Wilson**. (Nel report attuale la tabella per-seed è preferita al rate.)
- **matrix-powering** (soluzione locale, `A^{3^L}`) vs **spectral/Laplacian**
  (globale, componenti); **algorithmic** vs **heuristic** (degree heuristic:
  predice connesso per nodi ad alto grado → fallisce su 2clique).
- **seed-dominated**: l'esito dipende dal seed (init + stream di training), non dal
  filtro. (Nei run, un seed fissa init pesi *e* stream dei grafi; i test set OOD
  sono fissi tra seed.)

---

## 11. Stato attuale (aggiornare quando cambia)

- **Parte I (repro n=20)**: completa, 8 seed/condizione (minimal + RoBERTa).
  Conclusione: data lever non aiuta, seed-dominato. Figure: `fig1_reproduction`,
  `seed_examples` (restricted vs unrestricted per seed), tabella per-seed.
- **Parte II**: cicli, diagnostica 2chain (+exp2), conteggio componenti, e
  **reach depth sweep** (L=1→d\*=3, L=2→9, L=3/4→63): muro = `3^L`, sale con la
  profondità; figura `reach_by_distance.png`. L=2 confermato saturo (loss ~0.13,
  1M step).
- **Esperimento A (embedding geometry)**: fatto; segnale qualitativo (il modello
  codifica un "righello di distanza", non un'etichetta di componente); la metrica
  per-distanza grezza è rumorosa (read-out lineare ≠ distanza embedding).
- **Esperimento B (Laplaciano/similarità, n=64)**: fatto. Il read-out di
  similarità (λ=0) aiuta il reach; la loss Laplaciana a λ=1 collassa il modello
  ("tutto connesso"). → motivò il filone Report 4 con λ piccolo + warmup.

**Stato Report 4 (al 2026-06-13): RISCRITTO E COMPLETO, 23 pagine, 3 thread + takeaway.**
Tutto multi-seed (n20 = 8 seed, n40 = 4 seed); solo depth-sweep e i checkpoint reach
n64 restano single-seed. Compila pulito da `report/4/`. Struttura:

- **Thread 1 — "what makes a graph hard: diameter and density".** Due assi
  SEPARABILI: (1) distanza-vs-capacità (muro `3^L=9`, fig `acc_vs_distance.png`);
  (2) densità-vs-training (grafi più densi del training falliscono, fig
  `acc_vs_density.png`). Lo **spectral gap NON è un terzo asse separabile** sui grafi
  naturali: corr(diametro, log-gap) = −0.92 (fig `diam_gap_collinear.png`, scatter
  colorato per famiglia). + pannello famiglie (`tab:fam`, 8 seed) con onestà sulla
  bimodalità per-seed. + tabelle **per-seed reach-by-distance** (chain_plus, 1chain,
  path_union; n20+n40; ER vs mixed) e **per-graph-diameter** (path_union, er).
- **Thread 2 — "the spectral gap: does a bottleneck add difficulty?".** barbell
  (bottleneck a d=3 dentro capacità, `tab:bar_n40/n20`); barbell_var presentato
  **per diametro** (è densità+ponte, NON gap — vedi errore 22); expander_var
  (confound densità, errore 23); e il **probe parallel_paths CONFOUND-FREE** (errore
  24, `eval_parallel_paths_clean.py`): risultato POSITIVO — oltre la capacità più
  route (resistenza più bassa) → connessione confermata (`tab:ppclean`,
  `parallel_paths_clean.png`). Conclusione: il bottleneck È isolabile con un probe
  costruito, ma collineare col diametro sui grafi naturali.
- **Thread 3 — "the read-out".** similarity raddoppia il reach (`tab:simlin_n40/n20`
  per-seed solo famiglie interessanti, + `tab:chainplus_n40` + `fig:chainplus_n40` +
  depth-sweep `tab:depthsim`); loss Laplaciana **con formula** (`tab:lambda`); reach
  vs cut (`tab:reachcut`); near-miss cut (`eval_near_miss_cut.py`, `fig:nearmiss`);
  degree heuristic (`tab:degree`, 5 arm incl. loss, n20). Near-miss e degree heuristic
  sono **sottosezioni di Thread 3** (NON sezioni a sé — l'utente le voleva omogenee).

**Deliverable nuovi fatti questa sessione (script + risultati in repo):**
- `eval_difficulty_map.py` (la mappa di difficoltà; bug `offdiag.reshape` fixato →
  errore #20 nel codice: usare `offdiag.sum()`). Figure aggregate poi rifatte più
  chiare: `runs/report4/report4_figs/acc_vs_distance.png`, `acc_vs_density.png`,
  `diam_gap_collinear.png`. (Le vecchie `runs/report4/difficulty_map/map_combined.png` e
  `partial_dependence.png` esistono ma sono superate.)
- `eval_near_miss_cut.py` + `scripts/near_miss_cut.sbatch` (test del cut).
- `eval_parallel_paths_clean.py` + `scripts/parallel_paths_clean.sbatch` (probe gap
  confound-free; girato, risultato positivo).
- Figure report in `runs/report4/report4_figs/`.

**Stato job HPC: niente in coda/running.** Tutti i job completati (fam20 8 seed,
fam40 4 seed, diffmap, near_miss_cut, ppclean).

**Possibili follow-up (non urgenti):** depth-sweep a **n>64** per separare i read-out
a L=3 (a n=64 sia linear che similarity saturano tutto il range, non si distinguono).

- **Workflow git ricorrente**: a fine job su HPC `git add runs/... && commit &&
  push`; in locale `git pull` poi generare le figure. NON lanciare `plot_families.py`
  prima del pull (vedi errore 14).

---

*Per aggiungere questo file a git (lo fa l'utente):*
`git add istruzioni.md && git commit -m "Add project handoff instructions" && git push origin main`
