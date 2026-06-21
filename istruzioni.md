# Istruzioni del progetto — handoff per Claude

Documento di riferimento per chiunque (incluso Claude in una nuova chat) riprenda
questo progetto. Leggere **tutto** prima di agire.

> **⚠️ REGOLA DI LETTURA NON NEGOZIABILE (prima di toccare qualsiasi cosa).** Leggere
> **attentamente OGNI riga** di questo `istruzioni.md` e **OGNI riga dei 5 report** dai
> sorgenti `.tex` (`report/2..5/*.tex`; il report 1 è solo PDF). In particolare il
> **Report 5** (`report/5/transformer_for_graphs_5.tex`) va **riletto 4 VOLTE riga per
> riga** prima di scriverci dentro: è lungo, denso, e gli errori più gravi di questa sessione
> sono nati dal **non aver letto tutto** (vedi errore 52: analizzato solo il barbell e lasciato
> figura/tabella incomplete perché non si era ragionato su cosa l'esperimento misurava su
> *tutte* le famiglie). Non saltare righe perché "sembrano un dettaglio": le caption, i `%`
> commento negli stub, e le righe-errore qui sotto contengono i vincoli che fanno fallire chi
> non li legge. Non rispondere/agire da una vista parziale di un file: se un `Read` è troncato,
> continuare fino in fondo.

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

**Report 5** (`report/5/transformer_for_graphs_5.tex`, compilare da `report/5/`):
nuova domanda della **prof** — il transformer impara **matrix-powering** o un **DFS**
(traversata sequenziale che si ferma dopo un budget di passi)? I due fanno predizioni
**opposte** su grafi *corti ma grandi*. Esperimento core: **bridged-cliques vs split-cliques**
(due clique da n/2; label A = unite da **un solo arco** → 1 componente, label B = niente arco
→ 2 componenti). Matrix-power vede il ponte a distanza ≤3 (banale a ogni clique-size, dentro
capacità 9); un DFS visit-bounded si "blocca" nella clique vicina prima di attraversare, e
peggiora con la clique-size. Quattro misure: (1) discriminazione delle label; (2) **per-blocco**
within-A/within-B/cross (dove si ferma); (3) **sweep clique-size** (cross-acc piatto=matrix-power,
in discesa=DFS); (4) **confronto con oracoli** (matrix-power + bounded-BFS/DFS in `dfs_oracle.py`).
Più: classificatore binario dedicato (accuracy-by-clique-size), e **densità in ottimizzazione**
(ER a varie p: denso aiuta o rallenta il training?). **STATO al 2026-06-20 (7a sessione): §5.1–5.5 analizzate e SCRITTE (26 pp.), compila pulito da
`report/5/`.** §5.1–5.4: affinamento chiave = TRE oracoli MP/DFS/BFS (errore 45), esito = il modello è
una traversata visit-bounded (MP fino a budget ~6–7 nodi, poi BFS-bloccato), DFS rifiutato.
§5.5 (similarity-budget, 6a sessione): il read-out di similarità SPOSTA il budget-nodi (knee n40 mixed da
c≈7 lineare a c≈11–12 similarity ≈ il doubling previsto = stesso meet-in-the-middle che raddoppia il reach in
Report IV → budget-nodi e reach-distanza = stessa capacità in due geometrie); a n20 invariato; ER = confound.
**§5.4 chiusa del tutto (7a sessione): il capstone DFS-incluso (`oracle_families_dfs/`) è stato ANALIZZATO su
TUTTE e 12 le famiglie (non solo barbell): pooled, ai budget discriminanti il modello segue matrix-power su
SIA BFS (≤0.05) SIA DFS (≤0.06) → "non è una traversata bounded" vale per ENTRAMBE; il barbell n40 è l'unica
firma bounded e lì il contrasto DIRETTO BFS-vs-DFS dà BFS 0.61–0.74, non DFS → DFS rifiutato anche sul grafo
naturale. Figura `oracle_families_follow.png` ora a 3 curve (MP/BFS/DFS), `tab:capstone` con blocco follows-DFS.
§5.2 portata a 8 seed n20-mixed (`broldsd`) + frase xcheck.** **STATO al 2026-06-20 (8a sessione): §5.6 (train-on-bridged
`trbr`) + §6 VERDETTO SCRITTI → REPORT 5 COMPLETO (30 pp.). §5.6: il crollo cross-block di §5.2 era un BUCO DI DATI,
non un muro duro (col bridged nello stream il modello propaga il ponte a OGNI clique-size, c≤20) → verdetto =
matrix-powering distance-bounded, NON traversata bounded; DFS morto. PROSSIMO = solo il BONUS opzionale depth-sweep §5.7
(`sec:res-depth`, stub pronto; dati 6/8, n40-L3 in coda). Vedi §11 "Stato Report 5" per il dettaglio.**

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
21. **⚠️ REGOLA NON NEGOZIABILE — OGNI tabella e OGNI figura deve dichiarare MODELLO +
    TRAINING SET + TEST SET nella caption. SEMPRE. Come nel Report 4.** È l'errore che
    l'utente segnala più spesso: NON dare per scontato che si capisca dal testo. Ogni caption,
    da sola, deve rispondere a:
    - **MODELLO**: architettura (es. minimal/A.1 vs RoBERTa; read-out linear vs similarity),
      `L=2`, `d_model=512`, n.ro teste, e — se è un classificatore — `mean-pool + 1 logit`.
    - **TRAINING SET**: *trainato su cosa* (ER vs mixed; o lo stream specifico, es. 50/50
      bridged/split random-c), **quanti seed**, **quale dimensione** (`n=20`/`n=40`).
    - **TEST SET**: su *quale famiglia/pool* è misurato, e quanti grafi.
    - **METRICA**: quale (exact-match / pairwise / reach / accuracy / loss) e se i numeri sono
      **media o mediana** e di **quali seed**.
    Template caption: «\emph{Model:} … (arch, L, d_model, read-out), trainato su … (training set),
    N seed, n=…. \emph{Test set:} … (famiglia, #grafi). \emph{Metrica:} … (media/mediana su …).»
    Se una caption non ha questi campi, è INCOMPLETA: aggiungerli prima di consegnare.
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
32. **`__pycache__/*.pyc` sono TRACCIATI e bloccano `git pull` su HPC.** I job
    rigenerano i `.pyc` → modifiche locali ai file tracciati → `git pull` aborta
    ("local changes would be overwritten"). Analogo dei png (#14) e degli `out/*.out`
    untracked che collidono quando un commit li aggiunge. Sblocco sicuro sul login node:
    `git checkout -- __pycache__/` (scarta i .pyc, si rigenerano) poi
    `git stash push -u -m "..."` (mette via gli out untracked senza perderli) poi
    `git pull`. Fix definitivo (lo committa l'utente sul Mac):
    `git rm -r --cached __pycache__ && printf '__pycache__/\n*.pyc\n' >> .gitignore`.
33. **`runs/` è organizzata in bucket per-report** (vedi §5, riorganizzata 2026-06-13):
    `runs/report1…4/` + `runs/extra/`. Ogni nuovo output va nel bucket del report
    giusto; i path negli script/`.tex`/docs sono già aggiornati. `n40_cross_experiment`
    è splittata fra report2 (convergence/per-distance) e report3 (resto). Cross-report:
    report3 legge i modelli big da `report2/`, report4 legge `reach_depth` e
    `repro_paper_n20_roberta` da `report3/`.
34. **CORREZIONE (2026-06-19): i `.pt` dei report 3/4 NON erano cancellati — sono tutti su HPC.**
    L'allarme "cancellati per quota" era un FALSO POSITIVO: il comando diagnostico
    `find runs/report3 runs/report4 -name "*.pt"` era scoped alle cartelle SBAGLIATE. La
    riorganizzazione del 2026-06-13 ha spostato nei bucket `runs/reportN/` **solo i file
    tracciati da git** (json/png/eval); i `.pt` sono **gitignored**, quindi non sono mai stati
    spostati fisicamente e sono rimasti ai loro **path ORIGINALI pre-bucket**. Per questo i
    bucket `report3/4` non contengono `.pt`, ma i pesi esistono — altrove. Quota OK su HPC:
    104G/180G, `runs/` = 7.4G. **Dove sono davvero i `.pt`** (path non-bucketati):
    `runs/families_n20/` e `runs/families_n40/` (linear **E similarity**, n20 seed 1000-8000,
    n40 seed 1000-4000), `runs/repro_paper_n20_roberta/` (base RoBERTa ER, tutti i seed),
    `runs/repro_paper_n20/`, `runs/reach_depth/` (L1-4, linear+similarity), `runs/laplacian/`,
    `runs/exp3_chaincount/`, `runs/curriculum_er_n40_*`, `runs/retrain_er_n40_big_*`. I `.pt` di
    report2/report5 invece SONO nei bucket (report2 spostato a mano il 13/6, report5 creato dopo).
    → REGOLA: per trovare un `.pt` fai **`find runs -name '*.pt'` SENZA scoping al bucket** e cerca
    per run-name. Conseguenza pratica: i checkpoint **similarity** per l'eval bridged-on-similarity
    (esp. proposto) CI SONO GIÀ → **eval-only via `sbatch`, niente retrain**. (I base linear di
    Report 5 erano stati comunque riallenati in `runs/report5/base_n{20,40}/`, ridondanti coi
    `families_n*` originali — nessun danno.)
35. **La reference "si blocca" (DFS) pulita è un BFS visit-bounded, NON un DFS.** Su due clique
    dense + 1 ponte, un DFS budget-bounded *si tuffa* e attraversa il ponte quasi subito
    (cross-rate ~0.81 a budget=c) → NON modella il "bloccato nella clique vicina". Il BFS
    visit-bounded riempie prima la clique vicina e attraversa solo se budget>c (cross ~0.01 a
    c=10): è la reference pulita e monotona. `dfs_oracle.py` ha `matrix_power_connectivity`,
    `bounded_bfs_connectivity` (primaria), `bounded_dfs_connectivity` (secondaria). Predizioni:
    matrix-power cross-acc **piatto** nella clique-size; bounded-traversal **in discesa**.
    Modello piatto→matrix-power, in discesa→DFS-like.
36. **Il read-out lineare del RoBERTa NON è simmetrizzato** (`model.py:349`), a differenza del
    minimal `GraphConnectivityTransformer` che simmetrizza (`:148`). Quindi sul base RoBERTa
    `R̂_ij ≠ R̂_ji` è possibile → l'**asimmetria DFS è osservabile** (eval_bridged_cliques misura
    il disaccordo per-direzione sul blocco cross).
37. **`DataLoader(prefetch_factor=..., persistent_workers=True)` richiede `num_workers>0`**:
    `train_bridged_classifier`/`train_families` crashano con `--num_workers 0` (debug locale).
    Su HPC usa 16.
38. **Densità a n20 alta = R quasi tutto-uno (trivialmente connesso)** → l'in-dist satura subito
    (val→1.0 già a p=0.05). Il range utile per l'effetto densità-in-ottimizzazione è
    **sparso→soglia** (p~0.05–0.15 a n20, soglia connettività ~0.15). A p≥0.2 il segnale è
    confuso dal fatto che il task diventa banale.
39. **Il classificatore bridged/split è un DISCRIMINATORE DEBOLE di matrix-power-vs-DFS.** I due
    grafi differiscono di **un solo arco** con footprint **LOCALE** (i 2 estremi del ponte hanno
    grado `c` invece di `c−1`; l'edge-count differisce di 1), separabile **senza** propagare la
    connettività. Accuracy piatta a 1.0 **NON** prova il matrix-powering: mostra solo che il ponte
    è *rilevabile* a qualsiasi clique-size (smonta solo la forma forte del DFS). Il test DECISIVO è
    il **base model** sulla matrice R (per-blocco, asimmetria, oracoli), NON il classificatore.
40. **Classificatore: l'ordinamento DFS appare SOLO nel transitorio.** Le clique grandi sono
    imparate per ultime (a step 2000 l'accuracy cala con `c`; nei run a `c` fisso la **loss
    iniziale** cresce con la densità), ma l'accuracy **finale è piatta a 1.0**. Training rumoroso:
    alcuni seed crollano a ~0.5 a metà run prima di stabilizzarsi → l'ordinamento è una *tendenza*,
    non una legge monotona. **NON** vendere il transitorio come un muro.
41. **Il classificatore n20 arriva solo a `c=10`** (2c≤20): la conclusione "nessun muro" è testata
    **solo fino a c=10**. Per spingere il range serve **n40** (c fino a 20). Lo sbatch è ora
    parametrico: `N=40 FIXED_C="5 10 15 20" sbatch --array=0-11%4 scripts/bridged_classifier.sbatch`.
42. **La bridged eval a n20 ER è CONFUSA.** Il modello ER è OOD sulle cricche dense (degree
    heuristic, Report IV) → un cross-block che cala con la clique-size **NON** è attribuibile a DFS
    (densità *e* lunghezza-da-attraversare crescono insieme, e l'ER fallisce le cricche comunque:
    within-clique ~0.69, discrimination ~0.5–0.65). Il test pulito è il modello **MIXED** (ha visto
    cricche dense in training → densità in-distribuzione).
43. **`eval_bridged_cliques.sbatch` SALTA in silenzio i checkpoint mancanti** (gira ~30s e finisce
    COMPLETED anche valutando quasi nulla). Sempre controllare il `.out` (righe `-- skip ...`) e
    contare i json in `runs/report5/bridged_cliques/` prima di fidarsi. Va **ri-lanciata DOPO** che i
    retrain base (`base_n20`/`base_n40`) sono finiti.
44. **Concatenare job SLURM: `sbatch --dependency=afterany:JOB1:JOB2 ...`** (parte quando i
    precedenti FINISCONO, comunque vada). Preferire **`afterany`** ad `afterok` per gli array:
    `afterok` lascia il dipendente PD per sempre (`DependencyNeverSatisfied`) se anche un solo task
    fallisce; la bridged eval salta comunque i checkpoint mancanti, quindi `afterany` è più robusto.
45. **TRE oracoli DISTINTI, non confondere "DFS" e "BFS troncato".** matrix-power (distance-bounded,
    parallelo, simmetrico, denso=più facile); bounded-**DFS** (visit-bounded, sequenziale single-start,
    *si tuffa* → sulla bridged il cross-block **SALE** con c, ed è **asimmetrico**); bounded-**BFS**
    (visit-bounded, palla che riempie i vicini → cross **SCENDE** a ~0, simmetrico). REGOLA: ogni
    esperimento va letto vs **tutti e tre**. La density-sweep NON separa DFS da BFS (entrambi
    visit-bounded → "denso rallenta"): separa MP dalla famiglia visit-bounded. La bridged eval (§5.2) È
    il test che li separa: il modello SCENDE → **= BFS, ≠ DFS** (rifiutato per trend + zero asimmetria).
    Correzione onesta: NON "modello=BFS quasi esatto" — a c piccolo il modello è MP; la lettura giusta è
    **MP fino a budget ~6–7 nodi, poi BFS-bloccato**. Capstone oracle-vs-famiglie (job 533393) confronta
    MP-vs-BFS. **CORREZIONE 6a sessione (vedi errore 51): il DFS NON è scartato definitivamente** — lo era
    solo sui bridged COSTRUITI, e il capstone mostra che il **barbell** è l'unico grafo naturale con la
    firma → si RIESEGUE il capstone con anche l'oracolo DFS (`oracle_families_dfs/`) per confermare lì
    BFS-non-DFS. Non assumere DFS morto finché quel job non torna.
46. **`model_follows_bfs_on_disagree` ESCE da [0,1] ai budget piccoli (b=2,3): artefatto numerico, da
    SCARTARE.** In `eval_oracle_agreement_families.py` su un pair di disaccordo MP≠BFS i due `follows_mp`
    e `follows_bfs` devono **sommare a 1** (il modello binario ne matcha esattamente uno). A b=2,3 alcune
    celle per-seed danno follows_bfs>1 (es. 1.018, 1.209) → somma ~2: inaffidabile. REGOLA: usa solo i
    budget dove `follows_mp+follows_bfs ≈ 1` (controllo di reliability per-cella). Pooled-su-4-seed è pulito.
47. **TRAPPOLA del budget GRANDE nel capstone (§5.4): a b→n la BFS(b) DIVENTA la connettività vera**
    (visita tutta la componente). Quindi "il modello segue BFS" ad alto budget significa solo che il modello
    connette correttamente le coppie oltre-capacità = il **recupero oltre il muro 3^L** (Report IV), **NON**
    una traversata bounded. La firma §5.2 è l'OPPOSTO: matchare una **palla PICCOLA bloccata**. → la regione
    discriminante è **b piccolo + massa di disaccordo (`disagree_frac`) alta**. NON leggere il follows_bfs
    ad alto budget come evidenza visit-bounded. (Stessa logica: la colonna b=9 stampata nel `.out` è
    fuorviante a n40, dove il budget rilevante varia per famiglia — guarda lo SWEEP, non una colonna.)
48. **ESITO §5.4 (capstone), letto con cautela: sui grafi NATURALI il modello = matrix-power dove il test
    ha potere.** Pooled sulle 12 famiglie, modello **mixed**: a b≤9 (disaccordo 19–52%) follows-MP **0.95–0.98**,
    follows-BFS ≤0.05; il BFS-following sale solo a b≥16 (n20)/≥30 (n40) dove il disaccordo è collassato e
    BFS≈verità (= recupero oltre il muro). La firma BFS di §5.2 **NON si estende** ai grafi naturali — *perché*
    su grafi naturali distanza≈nodi (collinearità −0.82): il test è in gran parte **cieco**. L'UNICA eccezione
    è il **barbell** (held-out: ponte sottile tra due cricche dense = la struttura di §5.2): lì il mixed inclina
    verso BFS nel range medio (n40: 0.29→0.49 su b=2–14; n20: 0.21→0.58). **ER = cross-check confuso**
    (degree-heuristic OOD dentro i cluster → sembra più BFS, ~0.25, per il motivo sbagliato; non leggere).
    Lettura: il capstone **non estende** la firma visit-bounded, la **localizza** (visibile solo su grafi
    COSTRUITI per esporla: bridged §5.2 + barbell). Coerente con §5.2, non lo contraddice. SCRITTO in §5.4
    (`sec:res-capstone`), figura `oracle_families_follow.png`, `tab:capstone`.
49. **`plot_oracle_families.py` aveva un BUG** (`conds = sorted(groups)` → iterava le chiavi, crash su unpack).
    **FIXATO** in `sorted(groups.items())`. Lo script ora gira e scrive `runs/report5/report5_figs/oracle_families_follow.png`.
50. **DFS NON è scartato definitivamente — solo sui bridged COSTRUITI (§5.2).** L'utente (giusto) ha
    notato: abbiamo rifiutato il DFS proprio sulla struttura bridge+cricche-dense, e il capstone §5.4 mostra
    che il **barbell** (= la stessa struttura, ma naturale) è l'**unico** grafo naturale dove appare la firma
    visit-bounded. Sarebbe circolare dichiarare il DFS morto sulla costruzione e poi non testare l'unico grafo
    naturale che la riproduce. → **`eval_oracle_agreement_families.py` ora calcola TRE oracoli** (aggiunto
    `bounded_dfs_connectivity`): per ogni budget riporta `model_vs_dfs`, e i "follows" sulle coppie di
    disaccordo **MP-vs-DFS** e **BFS-vs-DFS** (`disagree_frac_bfs_dfs`, `model_follows_{bfs,dfs}_on_bfsdfs` — il
    separatore diretto: §5.2 DFS sale, BFS scende → disaccordano sul cross). Nuovo sbatch
    `scripts/oracle_agreement_families_dfs.sbatch` (job `oraclefamdfs`, stessi 16 ckpt) → output in dir SEPARATA
    `runs/report5/oracle_families_dfs/` (così il §5.4 già scritto NON è toccato). Quando torna: confrontare
    `model_follows_bfs_on_bfsdfs` vs `model_follows_dfs_on_bfsdfs` **sul barbell** (e famiglie dense) — se il
    modello segue BFS (si blocca) e NON DFS (si tuffa), il rifiuto DFS regge anche sui grafi naturali. Smoke-test
    fatto (invariante `follows_x+follows_y=1` ok). §5.4 e §6 nel `.tex` già ammorbiditi (DFS "rejected on the
    constructed bridged cliques only, not yet on the natural graphs"). **NON scrivere §6 finché `oracle_families_dfs` non torna.**
51. **`git restore runs/` (usato per sbloccare un `git pull`) SCARTA le figure tracked che una sessione
    passata aveva RIGENERATO ma non committato** (es. `base_bridged_*.png` rifatte a 3 oracoli). Nessuna
    perdita reale → si **rigenerano** con `plot_bridged_cliques.py`/`_classifier.py`/`_density_sweep.py` (no GPU,
    dati locali). REGOLA: dopo un `git restore runs/`, rilancia i plot-script prima di fidarti delle figure
    locali. Le figure §5.1–5.3 + i 3 plot-script erano rimasti **non committati da sessioni precedenti** →
    pushati questa sessione (commit "missing 5.1-5.3 figures"). Anche la revisione editoriale del **Report IV
    `.tex`** (retitoli sezioni + cross-ref) era non committata → pushata.
52. **⚠️ QUANDO UN ESPERIMENTO MISURA UNA QUANTITÀ SU *TUTTE* LE FAMIGLIE/CONDIZIONI, ANALIZZALA SU TUTTE,
    non solo sul caso "interessante".** Errore della 7a sessione: il capstone DFS-incluso
    (`oracle_families_dfs/`) calcola model-vs-DFS su **tutte e 12 le famiglie naturali**, esattamente come
    model-vs-BFS. Avevo letto i campi DFS **solo sul barbell** e lasciato Figura `oracle_families_follow.png`
    + `tab:capstone` **con solo BFS** → la conclusione «il modello = matrix-power, non una traversata bounded»
    parlava solo di BFS, e il lettore (giustamente) ha chiesto "dove sta il DFS sulle 12 famiglie?".
    **Lettura giusta:** pooled sulle 12 famiglie, ai budget discriminanti il modello segue **matrix-power
    su SIA BFS (≤0.05) SIA DFS (≤0.06)** (le due curve "follows-traversal" si sovrappongono ~0) → "non è una
    traversata bounded" vale per ENTRAMBE. Fix fatto: `plot_oracle_families.py` ora legge
    `oracle_families_dfs/` (verificato: riproduce le curve MP/BFS originali identiche) e disegna la **3a curva
    arancione follows-DFS** (`model_follows_dfs_on_mpdfs`, massa `disagree_frac_mp_dfs`); `tab:capstone` ha il
    blocco **follows-DFS** (b=6,10). REGOLA generale: prima di scrivere, chiediti "questo json/script copre più
    casi di quelli che sto guardando?" e copri TUTTI quelli rilevanti, non solo l'highlight.
53. **NON confondere DUE confronti diversi nel capstone (errore concettuale facile).** (a) **MP-vs-traversata**
    (follows-MP vs follows-BFS *e* follows-DFS, ognuno sulle proprie coppie di disaccordo con MP): dice se il
    modello = matrix-power o una traversata bounded — risposta = MP su tutte le 12 famiglie, contro ENTRAMBE le
    traversate. (b) **BFS-vs-DFS DIRETTO** (`model_follows_{bfs,dfs}_on_bfsdfs`, sulle coppie dove BFS e DFS
    *fra loro* disaccordano): serve SOLO a dire *quale* traversata quando una firma bounded esiste — cioè sul
    **barbell n40** (→ BFS 0.61–0.74, non DFS). La `tab:capstone` di per sé **non separa** BFS da DFS sul barbell
    (le colonne follows-BFS/follows-DFS sono entrambe "segue una traversata vs MP", e sul barbell sono simili);
    quella separazione è il contrasto diretto (b), riportato nel TESTO. Tenere i due confronti distinti nelle
    caption (l'ho scritto esplicito in `tab:capstone` e nel paragrafo "Setup" del §5.4). **Reliability (errore
    46):** usare solo i budget dove `follows_x+follows_y≈1`; b=2,3 scartati (artefatto), pooled è pulito; nel
    barbell n20 il separatore è **troppo debole** (≈0.54 mass-weighted, near toss-up) → dirlo, non forzare.
54. **Il read-out del capstone è MASS-WEIGHTED o al budget di picco-massa, MAI un singolo budget arbitrario.**
    Il `model_follows_*` va letto dove la **massa di disaccordo** è alta (la regione discriminante): a budget
    grande la massa collassa e l'oracolo bounded ≈ verità → "segue BFS/DFS" lì è solo il **recupero oltre-muro**
    (Report IV), NON una firma bounded (errore 47, la TRAPPOLA del budget grande). Le famiglie a clique singola
    densa (2cliques, clique_blocks) hanno `disagree_frac_bfs_dfs=0` (BFS e DFS coincidono su una palla densa) →
    **non separano** e non vanno lette come evidenza. Il barbell è l'unica famiglia naturale con massa BFS-DFS
    utile, e solo a **n40** (a n20 il canvas è troppo piccolo).

---

## 5. Cosa Claude deve sapere in una nuova chat

- Repo locale: `~/transformer-for-graphs` (Mac). Su HPC: `~/transformer-for-graphs`.
- HPC alias ssh: `hpc`. Utente: `3352759`. Env conda: **`graph_tf`**.
- Report: `report/1/` (solo PDF, niente sorgente), `report/2/`, `report/3/`,
  `report/4/transformer_for_graphs_4.tex`. **Compilare dalla cartella del report**
  (i path figure usano `\graphicspath{{../../}}`; `\includegraphics{runs/reportN/...}`).
  Due passate di `pdflatex` per i `\ref`/`\part`. Tutti i .tex hanno l'helper
  `\figorbox{path}{width}` (fallback se la figura non è ancora pullata).
- I checkpoint `.pt` stanno **solo su HPC** (gitignored). I risultati piccoli
  (json/png) sono in `runs/...` e versionati.
- **Struttura `runs/` (riorganizzata 2026-06-13): tutto è dentro un bucket per
  report**, per poter controllare i risultati per report:
  - `runs/report1/` — i primi esperimenti del PDF italiano (baseline, capacity_test,
    restrict_diameter_dynamics/_pairwise/_sweep, curriculum_diameter_dynamics).
  - `runs/report2/` — Ch1 n10/n14 (`retrain_er_3352759`, `retrain_2chains/2cliques_…`),
    Ch2 n40 small (`retrain_er_n40_4853xx`), Ch3 n40 big (`retrain_er_n40_big_{494467,
    495198,495199,495903}`, `curriculum_er_n40_big_494470`), e gli OOD-eval relativi
    (`ood_cross_experiment`, `ood_eval_n40_big_495904`, `ood_eval_curriculum_big_…`,
    `ood_eval_n14_…`, `ood_eval_n40_…`).
  - `runs/report3/` — repro n20 (`repro_paper_n20`, `repro_paper_n20_roberta`),
    capacity wall (`exp3_chaincount`, `reach_depth`), round exp2
    (`retrain_er_n40_big_exp2_4993xx`, `ood_eval_n40_big_exp2_…`).
  - `runs/report4/` — difficoltà/architettura (`families_n20`, `families_n40`,
    `difficulty_map`, `laplacian`, `family_gallery`, `report4_figs`).
  - `runs/report5/` — DFS vs matrix-power: `bridged_cliques/` (json per checkpoint, tag
    `n{N}_{set}_seed{S}`), `bridged_clf/` (classificatore binario), `density_sweep/p{XX}/`
    (ER a varie densità), `base_n20/`+`base_n40/` (base linear riallenati), `report5_figs/`.
  - `runs/extra/` — superato/orfano (`ood_eval_n40_big_495201`, `retrain_er_n20_diam11_…`).
  - **`n40_cross_experiment` è splittata**: `report2/` ne ha solo `01_convergence.png`
    e `02_per_distance.png`; il resto (cycles/diagnose/2chains/embed) è in `report3/`.
  - **Cross-report**: gli script del report 3 leggono i 4 modelli big da
    `runs/report2/…`; gli script del report 4 leggono `reach_depth` e
    `repro_paper_n20_roberta` da `runs/report3/…`. I path negli script/`.tex` sono già
    aggiornati di conseguenza.
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
- Checkpoint riusati: roberta n20 unrestricted (8 seed) = baseline `ER/linear`. **NB (2026-06-19,
  correzione errore 34): questi `.pt` ESISTONO su HPC, ma al path non-bucketato
  `runs/repro_paper_n20_roberta/...` (NON `runs/report3/...`, che ha solo json/png).**

**File chiave Report 5 (matrix-powering vs DFS):**
- `data.py`: `generate_bridged_cliques_graph(n, clique_size)` (due clique da n/2 + 1 ponte → 1
  componente) e `generate_split_cliques_graph` (niente ponte → 2 componenti). Differiscono di 1 arco.
- `dfs_oracle.py`: `matrix_power_connectivity(adj, L)` (distance-bounded, `(A+I)^{3^L}`),
  `bounded_bfs_connectivity(adj, budget)` (visit-bounded, reference "stuck" primaria),
  `bounded_dfs_connectivity` (secondaria). NumPy puro, no GPU.
- `eval_bridged_cliques.py` (eval-only): discriminazione, per-blocco (within-A/within-B/cross),
  asimmetria R̂_ij≠R̂_ji, sweep clique-size, confronto oracoli. Output
  `<out>/bridged_cliques.json`. `plot_bridged_cliques.py` rigenera le figure pooled per seed; include
  `plot_combined_cross_sweep` (la 2×2 `base_bridged_cross_sweep.png` con TUTTI E TRE gli oracoli: MP piatto,
  DFS che sale, BFS che scende) e `plot_combined_oracle_follow` (`base_bridged_oracle_follow.png`, MP-vs-BFS).
- `eval_oracle_agreement_families.py` (eval-only, CAPSTONE): per ogni base checkpoint e ogni famiglia
  naturale confronta R̂ del modello con **TRE oracoli** matrix-power, bounded-BFS, bounded-DFS (budget-sweep),
  overall + sui pair dove gli oracoli DISACCORDANO (MP-vs-BFS, MP-vs-DFS, BFS-vs-DFS); output
  `<out>/oracle_families.json` (per-famiglia + `pooled`). Due sbatch, stessi 16 checkpoint:
  `scripts/oracle_agreement_families.sbatch` → `runs/report5/oracle_families/` (§5.4, MP-vs-BFS, FATTO) e
  `scripts/oracle_agreement_families_dfs.sbatch` (job `oraclefamdfs`) → `runs/report5/oracle_families_dfs/`
  (re-run con DFS, errore 50, in coda). Aggregazione MP-vs-BFS `plot_oracle_families.py` (FIXATO errore 49 →
  `oracle_families_follow.png`); per i campi DFS (`model_follows_{bfs,dfs}_on_bfsdfs`) un plot va ancora
  scritto/esteso quando i json `oracle_families_dfs/` tornano. Groundwork oracle-vs-oracle (senza modello):
  `runs/report5/report5_figs/oracle_disagreement.{png,json}`.
- `experiments2/train_bridged_classifier.py`: classificatore binario bridged/split
  (`GraphBinaryClassifier`, trunk minimal single-head d512), clique-size random (accuracy-by-c)
  o fissa (convergenza). Output `history.json` (`val_acc_by_c`) + curve png.
- `plot_density_sweep.py`: convergenza/finale in-dist + OOD vs densità p (legge `density_sweep`).
- `plot_bridged_classifier.py`: figura aggregata del classificatore dai `history.json` (no GPU;
  per-seed accuracy-by-clique-size a step 2000 + loss-by-c dei run a c fisso → `runs/report5/
  report5_figs/classifier_by_clique_size.png`). **NB: cablato sui run n20, da estendere se si fa n40.**
- `scripts/`: `eval_bridged_cliques.sbatch` (eval-only; n20 ER da `density_sweep/p08`, resto dai
  retrain), `train_base_bridged_n20.sbatch` (n20 mixed + auto-eval bridged),
  `train_base_bridged_n40.sbatch` (n40 er+mixed + auto-eval bridged), `density_sweep.sbatch`
  (training ER a p={0.05,0.08,0.12,0.16,0.22}), `bridged_classifier.sbatch` (**parametrico in `N` e
  `FIXED_C` via env-var, default n20 c={3,6,10}**; n40 con `N=40 FIXED_C="5 10 15 20" --array=0-11%4`).
- Generatori chiave: il base RoBERTa linear è il modello sotto esame (read-out NON simmetrizzato,
  errore 36). I `.pt` base ESISTONO su HPC (correzione errore 34): linear+similarity in
  `runs/families_n{20,40}/`, base ER in `runs/repro_paper_n20_roberta/`, più i retrain Report 5 in
  `runs/report5/base_n{20,40}/`. → eval-only, niente retrain per gli esperimenti su pesi esistenti.

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

**>>> AGGIORNAMENTO 8a sessione (2026-06-20): §5.6 (train-on-bridged, il DECISIVO) ANALIZZATA e SCRITTA + §6
VERDETTO SCRITTO → IL REPORT 5 È COMPLETO (30 pp., compila pulito da `report/5/`).** §5.6 (`sec:res-capacity`):
i `runs/report5/bridged_cliques_trained/` (8 seed: n20×4 c≤10, n40×4 c≤20) mostrano che, col bridged+split AGGIUNTO
allo stream mixed (`--include_bridged`, fam_tag `mixedbr`), il cross-block è **1.00 a OGNI clique-size** (piatto
sull'oracolo matrix-power, knee SPARITO), disc=1.00, split-exact=1.00 (NON over-connette), asimmetria≈0 — vs il
baseline held-out §5.2 che crolla a 0 oltre c≈6–7. **ESITO: il "node budget" di §5.2 era un BUCO DI DATI, non un
muro di capacità duro** (il ponte è SEMPRE a distanza ≤3 = dentro capacità 3^L=9 a ogni c → l'architettura L=2 ne è
capace, mancava solo nei dati). Caveat onesto scritto: NON sposta il muro-distanza 3^L (skill within-capacity); il
depth-sweep `brdepth` (L1/L3, ancora in coda) è il test del knee∝3^L. Connesso a §5.5 (il read-out muove il budget):
budget malleabile su 2 leve ⇒ artefatto soft, non secondo limite ⇒ inclina verso matrix-power. Figura
`base_bridged_trained_knee.png` (held-out vs trained, 2 pannelli n20/n40) da `plot_bridged_cliques.py`
ESTESO (`load_all_trained` + `plot_trained_vs_heldout`, regex `n{N}_seed{S}`, `--trained_root`). §6 (`sec:verdict`):
sintesi — DFS rifiutato 2 volte (§5.2 costruiti + §5.4 barbell naturale); non matrix-power LETTERALE (budget ~6–7 a
distanza ≤3) MA il budget è rimovibile (§5.5 read-out + §5.6 dati) = euristica data-prior, non capacità; density §5.3
(denso=più veloce) e capstone §5.4 (MP dove il test ha potere) confermano ⇒ **verdetto = matrix-powering
distance-bounded, NON traversata bounded**; caveat: evidenza comportamentale (serve probe meccanicistico) + §5.6 testa
solo within-capacity. **Consegnati i comandi git all'utente** (Claude non committa): `report/5/transformer_for_graphs_5.{tex,pdf}`,
`plot_bridged_cliques.py`, `runs/report5/report5_figs/base_bridged_trained_knee.png` (+ i png rigenerati identici). <<<**

**>>> AGGIORNAMENTO 7a sessione (2026-06-20): §5.4 chiusa COMPLETAMENTE col DFS-su-tutte-le-famiglie + §5.2 a
8 seed. Il `.tex` è a 26 pp., compila pulito. RESTA SOLO §5.6 (dati pronti) + §6 verdetto (ora SBLOCCATO).** Cosa
fatto: (1) **§5.4 — capstone DFS-incluso ANALIZZATO su TUTTE e 12 le famiglie** (non solo barbell, errore 52):
pooled, ai budget discriminanti il modello segue matrix-power su SIA BFS (≤0.05) SIA DFS (≤0.06) → "non è una
traversata bounded" vale per ENTRAMBE; il **barbell n40** è l'unica firma bounded naturale e il contrasto DIRETTO
BFS-vs-DFS lì dà **BFS 0.61–0.74, non DFS** (n20 troppo debole, ≈0.54; 2cliques/clique_blocks danno 0 disaccordo
BFS-DFS). `plot_oracle_families.py` ora legge `oracle_families_dfs/` e disegna la **3a curva follows-DFS**
(figura `oracle_families_follow.png` a 3 curve MP/BFS/DFS); `tab:capstone` ha il blocco **follows-DFS**; testo+caption
§5.4 riscritti per coprire entrambe le traversate (errori 52–54). (2) **§5.2 a 8 seed n20-mixed** (`broldsd`,
seeds 5000–8000 poolati ri-girando `plot_bridged_cliques.py`): numeri invariati (knee c≈6–7, cross≈0; c=6 0.95→0.94),
tabelle/caption aggiornate a 8 seed, + 1 frase **xcheck** (retrain `bridged_cliques_xcheck/` riproduce gli originali
a 2 decimali). §5.5 linear-n20 etichettato a 8 seed per coerenza con la figura rigenerata. **Consegnati i comandi
git all'utente** (Claude non committa): `plot_oracle_families.py`, `plot_bridged_cliques.py`,
`report/5/transformer_for_graphs_5.{tex,pdf}`, figure in `runs/report5/report5_figs/`. <<<**

**Stato Report 5 (aggiornato 2026-06-20, 5a sessione, vedi 7a sopra per il delta): §5.1, §5.2, §5.3 E §5.4 (CAPSTONE)
analizzati e SCRITTI. Il `.tex` arriva a 22 pagine (ORA 26), compila pulito da `report/5/`.** Restano: §5.5 similarity-budget
(SCRITTA in 6a sessione), §5.6 capacità-vs-prior-dati (dati ORA PRONTI in `runs/report5/bridged_cliques_trained/`,
era "job trbr in corso"), §6 verdetto (per ultimo, ora sbloccato lato DFS). §5.1: setup due regimi + `tab:clf_early`/`tab:clf_early_n40`
+ `fig:clf`/`fig:clf_n40` + `fig:clf_loss` + caveat (errori 39–42). §5.2: per-blocco `tab:base_blocks`, sweep
`tab:base_sweep`+`fig:base_sweep`, oracle-follow `fig:base_oracle`, confound ER `tab:base_er`. §5.3:
`tab:dens_conv`/`tab:dens_ood` + `fig:dens_conv`/`fig:dens_ood`. §5.4 (NUOVA, errori 46–49): `fig:capstone`
(`oracle_families_follow.png`) + `tab:capstone` — esito: modello = matrix-power sui grafi naturali, firma BFS
solo su barbell (vedi errore 48).

**AGGIORNAMENTO 5a sessione (2026-06-20): §5.4 capstone CHIUSA + push del "dimenticato".** I 3 job leggeri
sono FINITI e pullati: `533393` (oraclefam → `runs/report5/oracle_families/`, 16 json), `533476` (brsim →
`runs/report5/bridged_similarity/`, 24 json, PRONTO per §5.5), `532846` (bridged eval re-run, ridondante con
§5.2). §5.4 scritta e committata (tex/pdf + `plot_oracle_families.py` fixato + `oracle_families_follow.png`).
Pushate anche le figure §5.1–5.3 + i 3 plot-script (erano non committati da sessioni vecchie, errore 50) e la
revisione editoriale del Report IV `.tex`. **Job ANCORA in corso (NON analizzare/pushare finché non finiscono
TUTTI):** `trbr` train-on-bridged §5.6 (533479_0-3 RUNNING su 8 seed, 533480_4-7 PD), `brdepth` depth-sweep
§5.6-bonus (533482/483 PD), `broldsd` +4 seed §5.2 (533484 PD).

**AGGIORNAMENTO 4a sessione (2026-06-19): scoperto che i `.pt` NON erano cancellati (errore 34 corretto)
+ progettati e creati gli script per gli esperimenti CONCLUSIVI + messi i PLACEHOLDER nel `.tex`.** Il
report 5 ora ha, dopo §5.3, gli stub: §5.4 capstone (`sec:res-capstone`), §5.5 similarity-budget
(`sec:res-sim`), §5.6 capacità-vs-prior-dati (`sec:res-capacity`), §6 verdetto (`sec:verdict`). Ogni stub
ha nei commenti `%` del `.tex`: script che lo produce, path di output, predizione/cosa-guardare, e i campi
caption (regola 21). **La prossima chat: UN esperimento, riempi lo stub corrispondente.** Nuovi script
(eval-only quelli leggeri): `scripts/eval_bridged_similarity.sbatch` (Exp 3, sui pesi similarity esistenti
`runs/families_n{20,40}/...similarity...`), `scripts/eval_bridged_oldseeds.sbatch` (+4 seed n20-mixed 5000-8000
nel pool §5.2 + cross-check `bridged_cliques_xcheck/`), `scripts/train_bridged_in_stream.sbatch` (Exp 2,
train-on-bridged: `--include_bridged`, decisivo capacità-vs-dati), `scripts/train_bridged_depth_sweep.sbatch`
(Exp 1 bonus, `--n_layers` 1/3). Modifica retrocompatibile a `train_families_n20.py`: aggiunti `--n_layers`
(default 2) e `--include_bridged` (famiglie `bridged`+`split` con clique-size random; fam_tag→`mixedbr`;
run-name prende `_L{L}` se L≠2).

**LEARNING-CHIAVE 3a sessione (errore 45): TRE oracoli DISTINTI (matrix-power vs bounded-DFS vs
bounded-BFS), non due.** L'utente ha fatto notare che il report mischiava "DFS" e "BFS troncato". Ora ogni
esperimento è letto vs tutti e tre, in modo coerente. §2.4 ridefinita (titolo "Reference algorithms: matrix
power, bounded DFS, and bounded BFS"; tolto il paragrafo "A subtlety... recorded for honesty", ora
"Depth-first dives, breadth-first gets stuck"). §5.2 è il test che SEPARA i tre: il DFS oracle SALE con c
(si tuffa), il modello SCENDE → **DFS RIFIUTATO** (per trend + zero asimmetria); il modello = MP-fino-a-budget-
~6-7-nodi-poi-BFS-bloccato. **Da §5.2 in poi si porta avanti solo matrix-power vs bounded-BFS.** §5.3 NON
separa DFS da BFS (entrambi visit-bounded → "denso rallenta"): separa MP dalla famiglia visit-bounded.
Figure base rifatte a 3 oracoli (`plot_bridged_cliques.py` → `plot_combined_cross_sweep`/`_oracle_follow`).

**>>> WORKFLOW PER LE PROSSIME CHAT: UN esperimento per chat. Leggere PRIMA, RIGA PER RIGA, TUTTO
`istruzioni.md` + TUTTI i 5 report dai `.tex` — e **rileggere il Report 5 (`report/5/transformer_for_graphs_5.tex`)
4 VOLTE riga per riga** (vedi regola di lettura in cima al file): la maggior parte degli errori nasce dal non aver
letto tutto. POI analizzare l'esperimento assegnato dai json già in `runs/report5/`, **coprendo TUTTE le
famiglie/condizioni che l'esperimento misura, non solo il caso interessante** (errore 52), scrivere la sotto-sezione
nei Results (caption complete — regola 21; tabella+figura per ogni claim — regola 20), e CONSEGNARE i comandi git
all'utente (Claude non committa). Non rifare i job: i dati ci sono già. <<<**

**>>> PIANO PER CHIUDERE IL REPORT — AGGIORNATO 8a sessione: §5.4/§5.5/§5.6/§6 e old-seed sono TUTTI FATTI E
SCRITTI. IL REPORT 5 È COMPLETO (30 pp.). RESTA SOLO un BONUS OPZIONALE: il depth-sweep §5.7 (`sec:res-depth`),
stub già pronto nel `.tex` con scaffolding completo nei `%` → vedi la voce "Exp 1 — depth-sweep" nella mappa sotto
(dati 6/8 pronti, n40-L3 in coda). La cronologia 1–4 qui sotto è storica. NB: se il depth-sweep non viene mai scritto,
il report resta comunque completo (il muro 3^L è già stabilito nei Report III–IV; §5.7 lo aggancia causalmente al
node-budget, ma non è un gate).**

1. **§5.4 CAPSTONE oracle-vs-famiglie** (`sec:res-capstone`). ✅ **FATTO e SCRITTO** (5a sessione, vedi
   errori 46–49). Job `533393` finito, json in `runs/report5/oracle_families/` (16). Figura ufficiale
   `oracle_families_follow.png` da `plot_oracle_families.py` (fixato, errore 49). Esito in errore 48.

2. **§5.5 SIMILARITY-budget** (`sec:res-sim`) — **QUESTO È IL PROSSIMO.** Dati GIÀ pullati: `533476` (brsim)
   FINITO → `runs/report5/bridged_similarity/<tag>/bridged_cliques.json`, **24 json** (tag
   `n{20,40}_{mixed,er}_sim_seed{S}`: n20 8 seed mixed+8 er, n40 4 seed mixed+4 er). Sono l'output di
   `eval_bridged_cliques.py` sui pesi SIMILARITY (`runs/families_n{20,40}/...similarity...`), stesso formato
   dei json linear di §5.2 (`runs/report5/bridged_cliques/`). **DA FARE nella nuova chat:** (a) **estendere
   `plot_bridged_cliques.py`** perché legga anche `bridged_similarity/` (oggi è cablato su `bridged_cliques/`;
   il readout è auto-rilevato nel json — campo `readout`); (b) confrontare il **knee del cross-block-vs-clique-size**
   col baseline LINEAR di §5.2 (lì knee a c≈6–7, crollo 1→0). PREDIZIONE: knee ~6–7 → ~13–14 ⇒ il budget-nodi
   = capacità-di-distanza (il similarity raddoppia il reach a 2·3^L in Report IV ⇒ aggancio causale forte);
   knee INVARIATO ⇒ budget-nodi e reach-distanza sono meccanismi DIVERSI (comunque informativo). Leggere SOLO
   il **mixed** come test pulito (cricche dense in-distribution); ER = cross-check confuso (errore 42). Caption
   regola 21. Riempire lo stub `sec:res-sim`. (Errore 43: conta i 24 json / controlla skip prima di fidarti.)

3. **§5.6 CAPACITÀ-vs-PRIOR-DATI** (`sec:res-capacity`). Lancia `scripts/train_bridged_in_stream.sbatch`
   (Exp 2, `--array=0-3%4` n40 prima, poi `4-7%4` n20; ~12h/job) e — opzionale — `train_bridged_depth_sweep.sbatch`
   (Exp 1, n20 L1/L3 economico). Auto-evalutano bridged → `runs/report5/bridged_cliques_trained/` e
   `runs/report5/bridged_cliques_depth/`. Lettura: cross-block crolla LO STESSO con c ⇒ capacità dura;
   impara a propagare ⇒ buco di dati. Knee che si sposta con L ⇒ è la capacità 3^L.

   **PIÙ (gratis, da fare quando torna comodo):** `scripts/eval_bridged_oldseeds.sbatch` (eval-only) →
   +4 seed n20-mixed (5000-8000) nel pool `runs/report5/bridged_cliques/` (rigenera la figura/tabella §5.2 a
   8 seed) + `runs/report5/bridged_cliques_xcheck/` (consistenza retrain vs originali, 1 frase in §5.2).

4. **§6 VERDETTO** (`sec:verdict`). Da scrivere PER ULTIMO, dopo §5.6. **ORA SBLOCCATO lato DFS** (il capstone
   DFS-incluso è chiuso, 7a sessione: DFS rifiutato anche sul barbell naturale n40) → **l'UNICO gate residuo è
   §5.6** (train-on-bridged: capacità dura o buco di dati?). Sintesi: non-DFS, non-MP-puro, visit-bounded (MP fino
   a budget ~6-7 nodi, poi BFS-bloccato); densità (§5.3) punta verso MP; future work = probe meccanicistico +
   densità-clique a node-count fisso. Lo stub `sec:verdict` nel `.tex` ha già la scaletta aggiornata nei `%`
   (DFS landed, manca solo §5.6). <<<**

**Mappa per-esperimento** (✅ fatto&scritto · 🟢 dati pronti, DA analizzare · 🔵 job HPC in coda, dati NON
ancora pullati · ⏳ in attesa/opzionale):

- ✅ **Classificatore** (`runs/report5/bridged_clf/`, 22 run: n20 ×10, n40 ×12). ANALIZZATO e
  SCRITTO (§5.1). Esito: accuracy finale **piatta a 1.0** fino a c=10 (n20) e c=20 (n40) → smonta
  il DFS forte; l'ordinamento DFS è solo nel transitorio (clique grandi imparate per ultime; loss∝
  densità). Discriminatore **debole** (errore 39). Plot: `plot_bridged_classifier.py`.

- ✅ **Base-model bridged eval** (`runs/report5/bridged_cliques/`, 16 json). ANALIZZATO e SCRITTO
  (§5.2) — **il test decisivo, ed è quello che SEPARA i 3 oracoli**. Esito (sul MIXED, il test pulito):
  within-A=within-B=1.00 e split exact=1.00 (le cricche dense NON sono il problema), ma il
  **cross-block crolla con la clique-size** (piatto a 1.0 fino a c≈6, →0 oltre). **Confronto a 3 vie
  (rifatto questa sessione, errore 45):** MP oracle piatto a 1.0 (distanza ≤3 a ogni c); **DFS oracle
  SALE** con c (0.19→0.62 n20, →0.69 n40: si tuffa e attraversa il ponte → con budget più grande va più
  in profondità); **BFS oracle resta ~0** (bloccato nella clique vicina). Il **modello SCENDE** (1→0) →
  trend OPPOSTO al DFS, uguale al BFS → **DFS rifiutato** (1) per il trend e (2) per ZERO asimmetria
  (DFS single-start la predirebbe; il transformer emette R̂ in parallelo). Lettura precisa: **matrix
  power fino a un budget di ~6–7 nodi, poi traversata-BFS bloccata** (visit-bounded; NON "modello=BFS
  quasi esatto" — a c piccolo il modello è genuinamente MP, non BFS: correzione onesta del testo
  precedente). Budget assoluto (~6–7 uguale a n20 e n40) → visit-bounded, non distanza. Oracle-follow:
  sui cross dove MP e BFS DISACCORDANO il modello segue BFS (99.6% n20, 100% n40), MP 0.4%/0%. ER =
  cross-check confuso (errore 42, `tab:base_er`). Coerente col classificatore. **D'ora in poi si porta
  avanti solo MP-vs-BFS (DFS scartato in §5.2).** Plot: `plot_bridged_cliques.py` (ora con funzioni
  `plot_combined_cross_sweep` a 3 oracoli + `plot_combined_oracle_follow`) → `base_bridged_cross_sweep.png`
  (2×2, 3 oracoli)/`base_bridged_oracle_follow.png`.

- ✅ **Density sweep** (`runs/report5/density_sweep/p{05,08,12,16,22}/`, 20 run: n20 ER linear × 4
  seed per p). ANALIZZATO e SCRITTO (§5.3). Esito: **denso = più veloce e più affidabile** (steps→0.99
  da ~10⁵ a p=0.05 a ~5k a p≥0.16; seed-lottery sparito a p≥0.12) → direzione **matrix-power**, contro
  la forma forte di **QUALSIASI traversata visit-bounded** ("denso rallenta il training" — sia DFS che
  BFS lo prevedono; questo esp. NON separa DFS da BFS, separa MP dalla famiglia visit-bounded). CAVEAT ONESTO (errore 38): in ER più denso =
  distanze più corte = target quasi all-ones → a p≥0.16 il task è banale; l'ER **non isola** l'asse
  "più nodi da attraversare a distanza fissa" (lì densità e distanza si muovono opposte) — quell'asse lo
  isolano solo le bridged-clique (§5.1/§5.2), dove il costo di traversata RIAPPARE → i due risultati
  sono coerenti. Secondo dato interessante: l'**OOD transfer è non-monotono, picco a p=0.12** (sotto
  soglia connettività ~0.15) e lì meno seed-dipendente; troppo sparso = lottery, troppo denso = niente
  reach a lunga distanza. Tabelle `tab:dens_conv`/`tab:dens_ood` (per-seed), figure
  `density_convergence.png` (traiettorie + steps-to-0.99 + final, 3 pannelli) e `density_ood.png`
  (per-seed). Plot: `plot_density_sweep.py` (riscritto). Aperta: density a **n40**?

- ✅ **Capstone: oracle-vs-famiglie** (job HPC **533393** `oraclefam` FINITO; `runs/report5/oracle_families/`,
  **16 json**). ANALIZZATO e SCRITTO (§5.4, errori 46–49). Test **matrix-power-vs-bounded-BFS** su TUTTE le
  famiglie naturali (n20/n40, ER/mixed). Metrica = sui pair dove MP≠BFS, `follows_bfs`/`follows_mp` vs budget
  (`eval_oracle_agreement_families.py`; aggreg. `plot_oracle_families.py` FIXATO errore 49 →
  `oracle_families_follow.png`). **Esito (errore 48):** sul mixed, a b piccolo/discriminante il modello segue
  **matrix-power 0.95–0.98** (follows-BFS ≤0.05); il BFS-following sale solo ad alto budget = recupero oltre il
  muro (errore 47 = trappola), NON visit-bounded. La firma §5.2 **non si estende** ai grafi naturali (cieco:
  distanza≈nodi); UNICA eccezione il **barbell** (held-out, ponte+cricche dense → inclina BFS, n40 0.29→0.49).
  ER = confuso. Conclusione: il capstone **localizza** la firma (solo su grafi costruiti), non la estende.
  **ADDENDUM 6a sessione (errore 50): DFS NON dato per morto** → capstone RIESEGUITO con anche l'oracolo DFS
  (`scripts/oracle_agreement_families_dfs.sbatch` → `runs/report5/oracle_families_dfs/`).
  **✅ ANALIZZATO e SCRITTO (7a sessione, errori 52–54).** Il DFS è stato letto su **TUTTE e 12 le famiglie**, non
  solo barbell (errore 52): pooled, ai budget discriminanti il modello segue **matrix-power su SIA BFS (≤0.05) SIA
  DFS (≤0.06)** → "non è bounded" vale per entrambe le traversate; al budget grande entrambe salgono = recupero
  oltre-muro (errore 47/54). Il contrasto DIRETTO `model_follows_{bfs,dfs}_on_bfsdfs` (sulle coppie dove BFS e DFS
  fra loro disaccordano) separa le due **solo sul barbell n40** (→ **BFS 0.61–0.74, non DFS**; n20 ≈0.54 troppo
  debole; 2cliques/clique_blocks = 0 disaccordo BFS-DFS). `plot_oracle_families.py` ora legge `oracle_families_dfs/`
  (verificato identico a `oracle_families/` su MP/BFS) e disegna la **3a curva follows-DFS** →
  `oracle_families_follow.png` a 3 curve; `tab:capstone` esteso col blocco follows-DFS; testo/caption §5.4 riscritti
  (la frase "still in the queue" è stata SOSTITUITA col risultato). **§6 ora sbloccato lato DFS.**

- ✅ **Exp 3 — similarity-budget** (§5.5, `sec:res-sim`). ANALIZZATO e SCRITTO (6a sessione). Job `533476`
  (brsim), **24 json** in `runs/report5/bridged_similarity/<tag>/` (tag `n{20,40}_{mixed,er}_sim_seed{S}`: n20
  8+8, n40 4+4). `plot_bridged_cliques.py` ESTESO (`load_all_sim` + `plot_similarity_vs_linear` → figura
  `base_bridged_similarity_knee.png`; regex `_sim` separata, non tocca le figure §5.2). **Esito (sul MIXED, il
  test pulito):** il similarity NON rimuove il muro — a clique-size pieno il cross-block crolla a 0.00 ESATTO
  ogni seed (within=1.00, split-exact=1.00, disc=0.50, asym=0.000), come il lineare — ma **SPOSTA il budget**:
  knee **n40 da c≈7 (lin) a c≈11–12 (sim)**, +4/5 nodi, ≈ il doubling previsto. Stesso meet-in-the-middle che
  raddoppia il reach 3^L→2·3^L (Report IV): il read-out *one-sided* (lin: j nel ball di i) vs *two-sided* (sim:
  ball di i ∩ ball di j) → ogni nodo deve raggiungere solo il ponte → budget-nodi ~×2. ⇒ **budget-nodi (§5.2) e
  reach-distanza (Report IV) = STESSA capacità, due geometrie**, e poiché un read-out di OUTPUT la allarga, il
  limite vive in *quanto lontano aggrega il trunk* = matrix-power-con-budget, NON uno step-count di traversata
  sequenziale (un output map non lo toccherebbe). **Caveat onesti:** il knee si muove pulito SOLO a n40 (a n20
  c≤10 riempie il canvas, budget ~12–14 non esercitabile → knee invariato ~6–7, anzi crollo più netto); shift
  ~×1.5 non esatto ×2 (anche a n40 c→20 riempie il canvas). **ER = confound:** cross=1.00 PIATTO a ogni c ma per
  il motivo SBAGLIATO (over-connect tutto il denso → chiama connesse anche le split → split-exact=0.00,
  disc=0.50). Tabelle `tab:sim_sweep` (knee lin-vs-sim per-c) + `tab:sim_blocks` (per-blocco a c pieno),
  figura `fig:simknee`. Compila pulito (25 pp.).

- 🟢 **Exp 2 — train-on-bridged** (§5.6, stub `sec:res-capacity`; il test DECISIVO capacità-vs-prior-dati).
  **>>> QUESTO È IL PROSSIMO ESPERIMENTO DA ANALIZZARE (la chat che riceve questo handoff fa SOLO questo). <<<**
  **TUTTI gli 8 seed FINITI** — `trbr` 533479_0-3 + 533480_4-7 COMPLETED. **DATI PRONTI** (già in repo).
  Output eval in `runs/report5/bridged_cliques_trained/n{N}_seed{S}`.
  Script `scripts/train_bridged_in_stream.sbatch` (training ~12h/job; `--array=0-3%4` n40 prima, `4-7%4` n20).
  Usa `train_families_n20.py --include_bridged` (famiglie `bridged`+`split` random-c nello stream; fam_tag
  `mixedbr`). Checkpoint `runs/report5/train_bridged_in_stream/n{N}_mixedbr_roberta_linear_lam0_seed*`; auto-eval
  bridged → `runs/report5/bridged_cliques_trained/n{N}_seed{S}`. Lettura: cross-block crolla LO STESSO con c ⇒
  capacità dura a L=2; impara a propagare a ogni c ⇒ era buco di dati (rivedere il verdetto). (Non lanciato.)

- 🟢 **Exp 1 — depth-sweep su bridged** (BONUS elegante; ORA È L'UNICO ESPERIMENTO RIMASTO → stub `sec:res-depth`
  §5.7 PRONTO nel `.tex` con tutto lo scaffolding nei `%`). **DATI PARZIALI al 2026-06-20 (8a sessione): 6/8 task
  COMPLETED, 2 ANCORA RUNNING** (`brdepth` 533483_6, 533483_7 = n40 L3 su gnode05/06). Mapping array (N P L seed):
  0:n20-L1-s1000, 1:n20-L1-s2000, 2:n20-L3-s1000, 3:n20-L3-s2000, 4:n40-L1-s1000, 5:n40-L1-s2000, 6:n40-L3-s1000,
  7:n40-L3-s2000 → **FINITI: tutto n20 (L1+L3) e n40-L1; MANCA solo n40-L3 (task 6,7)**. Script
  `scripts/train_bridged_depth_sweep.sbatch` (`train_families_n20.py --n_layers` 1/3; mixed, bridged HELD-OUT;
  **L=2 = base esistente, riusa `runs/report5/bridged_cliques/n{N}_mixed_seed*` di §5.2**). Checkpoint
  `runs/report5/depth_sweep/n{N}_mixed_roberta_linear_lam0_L{1,3}_seed*`; auto-eval → `runs/report5/bridged_cliques_depth/n{N}_L{L}_seed{S}/`.
  **PROSSIMA CHAT — riempire `sec:res-depth`:** (1) aspettare task 6/7 (o scrivere n20+n40-L1 e aggiungere n40-L3
  quando atterra; contare i json, errore 43); (2) estendere `plot_bridged_cliques.py` con `load_all_depth` (regex
  `n(\d+)_L(\d+)_seed(\d+)` → chiave `(n,L)`) + `plot_depth_knee` che sovrappone model-cross-acc-vs-clique-size per
  L=1,2,3 per n (L=2 da `bridged_cliques/`), con oracolo MP → figura `base_bridged_depth_knee.png`; specchiare il
  wiring `plot_trained_vs_heldout`/`--trained_root`. Predizione: **knee cresce con L (≈3^L) ⇒ il node-budget di §5.2 È
  la capacità matrix-power in nodi (= muro distanza-9, due geometrie); knee fermo ~6–7 a ogni L ⇒ NON è 3^L** (sorpresa
  da riportare onestamente). **CONFOUND da NON mancare:** a L=1 la capacità MP è esattamente 3 hop e il cross è a
  distanza ≤3 → l'L=1 può perdere il cross per la DISTANZA, non per il budget; leggere L=2-vs-L=3 come confronto pulito.
  Poi piegare 1 frase in §5.6 ("what it does not settle") e nel §6 verdetto.

- ✅ **Old-seed bridged eval** (GRATIS, eval-only). **ANALIZZATO e FOLDATO in §5.2 (7a sessione).** `broldsd`
  533484. (a) `runs/report5/bridged_cliques/n20_mixed_seed{5000..8000}` → **+4 seed nel pool §5.2** (ri-girato
  `plot_bridged_cliques.py`, auto-poola per seed): numeri invariati (knee c≈6–7, cross≈0; c=6 0.95→0.94), tabelle
  `tab:base_blocks`/`tab:base_sweep` + caption a **8 seed**, oracle-follow 99.6%→99.4%. (b)
  `runs/report5/bridged_cliques_xcheck/` → retrain riproduce gli originali a **2 decimali** ogni seed → **1 frase
  xcheck** aggiunta al §5.2 setup. NB: la figura §5.5 ora mostra linear-n20 a 8 seed → etichette `tab:sim_sweep`
  linear-n20 e caption `fig:simknee` aggiornate (numeri linear invariati).

- ⏳ **(opzionale) Re-read del barbell a livello blocco** (dati Report 4, `runs/report4/`): la §3.2
  del `.tex` prevede di ri-aprire il barbell (ponte-path) a livello per-blocco per cercare la stessa
  asimmetria, senza nuovi run. Solo analisi.

- ⏳ **Base-model training** (`base_n20/`, `base_n40/`): servono soprattutto come pesi per la bridged
  eval; di per sé poco interessanti (convergenza in-dist). Bassa priorità.

- ⏳ **Verdetto finale matrix-power vs bounded-BFS** (§6, stub `sec:verdict`; DFS scartato in §5.2 E §5.4 —
  costruiti E barbell naturale): sintesi conclusiva, da scrivere PER ULTIMO, DOPO §5.6 (gli altri sono FATTI). Le fila già tirate: classificatore (capacità piatta a 1.0, costo solo in
  ottimizzazione → smonta la forma forte della traversata) + base-model bridged (cross crolla con
  clique-size a budget assoluto ~6–7 nodi → traversata visit-bounded NON matrix-powering; DFS rifiutato
  **sui bridged COSTRUITI**, modello = MP-fino-a-budget-poi-BFS) + density (denso = più veloce in training →
  direzione MP, ma ER conflà densità e distanza, non isola l'asse visit-bounded) + **capstone (§5.4, FATTO): la
  firma BFS NON regge su tutte le famiglie naturali — il modello è MP dove il test ha potere, BFS solo sui grafi
  costruiti (barbell). Localizza, non estende.** + **capstone-DFS (§5.4, 7a sessione, FATTO): su tutte le 12
  famiglie il modello segue MP su SIA BFS SIA DFS (≤0.06); DFS rifiutato anche sul barbell naturale n40 (BFS
  0.61–0.74) → DFS morto sia sui costruiti sia sul naturale.** + §5.5 (FATTO: il budget-nodi raddoppia col
  read-out = stessa capacità del reach-distanza). **Manca SOLO §5.6** (capacità dura o buco di dati?). Scrivere
  §6 SOLO dopo §5.6.

**Job HPC (aggiornato 2026-06-20, 8a sessione, `squeue`/`sacct`):**
- ✅ COMPLETED, pullati e SCRITTI: `533393` oraclefam (§5.4), `533476` brsim (§5.5), `533651` oraclefamdfs
  (§5.4 DFS), `533484` broldsd (§5.2 8-seed), `532846` bridged (re-run §5.2, ridondante), **`533479_0-3` +
  `533480_4-7` trbr (§5.6 train-on-bridged, SCRITTO 8a sessione)**, + i vecchi (clf, density, base20/40).
- 🟢 **COMPLETED — il PROSSIMO da analizzare (riempie `sec:res-depth` §5.7):** `brdepth` depth-sweep —
  task 0-5 COMPLETED (tutto n20 L1/L3 + n40-L1) → `runs/report5/depth_sweep/` + `bridged_cliques_depth/`.
- 🟡 **ANCORA RUNNING — non stagiare la dir finché non finisce:** `brdepth` `533483_6` e `533483_7` (n40 L3,
  gnode05/06) → la parte n40-L3 di `bridged_cliques_depth/` è PARZIALE. Quando finiscono: pullare e completare §5.7.
- **PUSH (su HPC) SCOPED solo le dir dei job FINITI** (NON `runs/report5` intero, o stagi i parziali di
  `brdepth`): `git add runs/report5/bridged_cliques_trained runs/report5/train_bridged_in_stream
  runs/report5/oracle_families_dfs runs/report5/bridged_cliques runs/report5/bridged_cliques_xcheck`, poi
  `git status` per VERIFICARE che NIENTE di `depth_sweep/`/`bridged_cliques_depth/` sia staged, poi commit+push.
  In locale `git pull` poi i plot/analisi. (Ordine: prima pusha §5.5 dal Mac, poi pull+push da HPC — errore
  "divergent branches" se entrambi i lati hanno commit non sincronizzati.)
  Comando per stampare la tabella RUNS di un array sbatch SENZA lanciarlo: `bash scripts/<nome>.sbatch`.
- Domanda aperta: density anche a **n40**?

- **Workflow git ricorrente**: a fine job su HPC `git add runs/... && commit &&
  push`; in locale `git pull` poi generare le figure. NON lanciare `plot_families.py`
  prima del pull (vedi errore 14).

---

*Per aggiungere questo file a git (lo fa l'utente):*
`git add istruzioni.md && git commit -m "Add project handoff instructions" && git push origin main`
