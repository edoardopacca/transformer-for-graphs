# Istruzioni del progetto — handoff per Claude

Documento di riferimento per chiunque (incluso Claude in una nuova chat) riprenda
questo progetto. Leggere **tutto** prima di agire.

> **⚠️ REGOLA DI LETTURA NON NEGOZIABILE (prima di toccare qualsiasi cosa).** Leggere
> **attentamente OGNI riga** di questo `istruzioni.md` e **OGNI riga dei 6 report** dai
> sorgenti `.tex` (`report/2..6/*.tex`; il report 1 è solo PDF). In particolare il
> **Report 6** (`report/6/transformer_for_graphs_6.tex`) va **riletto 4 VOLTE riga per
> riga** prima di scriverci dentro: è lungo, denso, e gli errori più gravi di questa sessione
> sono nati dal **non aver letto tutto** . Non saltare righe perché "sembrano un dettaglio": le caption, i `%`
> commento negli stub, e le righe-errore qui sotto contengono i vincoli che fanno fallire chi
> non li legge. Non rispondere/agire da una vista parziale di un file: se un `Read` è troncato,
> continuare fino in fondo.

---

> **🟥 STATO ATTUALE — LEGGI PRIMA DI TUTTO.** Report 1–5 **CONGELATI** (consegnati). In corso il
> **REPORT 6** (`report/6/transformer_for_graphs_6.tex`, da `report/6/`). Tema: **simulare una "path
> di ragionamento" — quali dati servono per imparare un certo ragionamento; sequenzialità vs
> parallelismo**. È il **report conclusivo**: **pochi esperimenti, puntuali, per un paper.** Piano +
> tesi + mindset + indicazioni della prof in **§13** (leggere PRIMA di agire). **Stato al 2026-06-30
> (fine 17a sessione):** ✅ **A (A1/A2/A3) ANALIZZATO e SCRITTO** nel `.tex` (§sec:res-multipath-clean/
> -samples/-truncated + figure/tabelle; dettaglio e lezioni in **§13.7**); ✅ **B ANALIZZATO e SCRITTO**
> (§sec:res-asym-chains: la sezione mostra SOLO i risultati, il *perché* dello split è una DOMANDA APERTA
> — dettaglio in **§13.8**); ✅ **C ANALIZZATO e SCRITTO** (§sec:res-chain-cliques C.1 + §sec:res-thick-bridges
> C.2: un ponte imparato compone SOLO DEBOLMENTE — regge 1 hand-off in più e ponti spessi, ma crolla su catene
> lunghe (K≥4) e transferisce solo in parte a blocchi ER non visti; tesi 4 forte REFUTATA, debole OK; dettaglio
> e numeri in **§13.9**); ⚪ **D (alberi)** non iniziato
> (esplorativo). **Lavoro ancora aperto** (lo spartiranno chat diverse, senza ordine obbligato): eventuali nuovi
> esperimenti (Thread D); eventuale analisi più a fondo di A/B (es. il meccanismo
> di B); il **Verdetto §sec:verdict** (da scrivere per ultimo, ora che A–C sono tutti dentro). Mappa codice/run in **§13.7** (A),
> **§13.8** (B), **§13.9** (C). Vincolo NON negoziabile: **niente "mixed" random in
> training** — UNA distribuzione esplicita e sensata per esperimento (ER, cliques, bridged cliques,
> multipath, path_union…). Restano validi gli **errori §4** (caption 21, 55–60) e Claude **NON
> committa** (consegna i comandi git). **Claude PUÒ fare `git pull`** (non è commit/push): i risultati
> HPC vanno committati lato HPC dall'utente, poi Claude pulla in locale.

> **🟩 NOVITÀ 16a sessione (2026-06-30) — VARIANTE SIMILARITY READ-OUT (Threads A/B/C), LANCIATA.** Prima di
> analizzare il Thread C (linear, lo fa un'altra chat), abbiamo creato il **gemello similarity** degli
> esperimenti A/B/C: stessi run, **solo `--readout similarity`**, output in cartelle `*_sim` separate (non
> tocca il filone linear). 7 nuovi sbatch `scripts/r6_{a1_train,a1_eval,a2_samples,a3_truncated,b_eval,
> c_train,c_eval}_sim.sbatch` (pushati, commit `2d782eb`). **STATO LANCIO:** ✅ A1-sim train sottomesso
> (job **551389** n10 [0-7], **551390** n20 pu [8-11], **551391** n40 pu [12-15], **551392** n64 [16-23],
> tutti `%4`) + eval gateati **551400** (r6_a1_eval_sim, afterany su tutti e 4) e **551401** (r6_b_eval_sim,
> afterany su 551390:551391:551392). ⚪ **NON ancora lanciati:** Onda 2 = **C-sim** (`r6_c_train_sim` +
> `r6_c_eval_sim`); Onda 3 = **A2/A3-sim** (`r6_a2_samples_sim` + `r6_a3_truncated_sim`). Dettaglio completo,
> mappa output e comandi residui in **§13.10**. È un **confronto similarity-vs-linear esplorativo**: NON c'è
> ancora uno stub `.tex`; l'analisi (e se entra nel report) viene dopo.

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
(ER a varie p: denso aiuta o rallenta il training?). **Stato: Report 5 COMPLETO e poi ACCORCIATO
(31→26 pp.), compila pulito da `report/5/`.** Esito (= il verdetto §6): il modello è
**matrix-powering distance-bounded** (soffitto = muro-distanza 3^L=9), NON una traversata bounded;
il DFS è rifiutato (sui bridged costruiti e sul barbell naturale n40); il "node budget" ~6–7 nodi
visto sui bridged è un'**euristica soft/data-prior** (si sposta col read-out di similarità, sparisce
mettendo i bridged nello stream, NON cresce con la profondità) e non una seconda capacità. Report 5
CONSEGNATO e congelato.

**Report 6** (`report/6/transformer_for_graphs_6.tex`, compilare da `report/6/`): dopo aver mostrato
il Report 5 alla prof, ci si **specializza su "simulare una path di ragionamento"** — capire **quali
dati servono per imparare una certo path di ragionamento (proprio guardando la logica, tipo vedere delle cliques come dei se e solo se, e nei bridged cliques il modello deve trovare quale nodo del blocco ci porta alla nuova cliques tramite il ponte)**, lungo i due assi che l'architettura rende
concreti: **parallelismo** (più cammini fra due estremi `a,b`) e **struttura iterata/ripetuta** (una
catena di link decisi localmente). È il **report conclusivo**: pochi esperimenti puntuali per un
paper. Filoni: (A) **multipath** = i grafi del §4.4 del Report 4 (k cammini disgiunti `a→b` a
distanza fissa `ℓ`), rifatti per bene (multi-seed, celle informative) — più cammini aiutano anche
senza averli mai visti in training; quanti samples per impararne la connettività; i **multipath
troncati** confondono. (B) **two-chains asimmetriche**: capire perché nel Report 4 una split (4,36)
risultava più facile di (17,23). (C) **bridged cliques iterate**: train su una bridge singola, test
su clique-ponte-clique-ponte-…; ponti con **più di un edge** (purché distanza ≤9); blocchi ER densi
(p≈0.6) al posto delle cricche. (D) **alberi** (esplorativo). Vincolo dati: **niente mixed random**,
una distribuzione esplicita per esperimento. **Tesi, mindset e indicazioni dettagliate della prof in
§13.**

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
55. **⚠️ LA CAPTION NON RIFÀ L'ANALISI DEL CORPO (lezione 10a sessione, raffina regola 21).** Ogni
    didascalia deve contenere **SOLO** Model + Training set + Test set + Metric, più le **legende** strettamente
    necessarie a leggere figura/tabella (cosa marca il **grassetto**, i **colori** delle curve, la linea
    *dashed/dotted*, il `×`, i footnote-dato `†`/`*`). **Zero interpretazione**: niente "il modello crolla / segue
    BFS / denser=faster / non è un muro" nelle caption. Regola operativa: per OGNI caption, l'analisi che togli
    **deve già stare nel testo che annuncia** quella figura/tabella; se non c'è, **aggiungila in UNA frase al
    corpo** (non lasciarla solo in caption). Nel Report 5 era già tutta nel corpo tranne la *discrimination=0.50*
    del mixed in §5.2, aggiunta al corpo. Risultato: doppia-analisi rimossa, ~2 pagine in meno.
56. **NESSUN riferimento a CODICE/FILE/PATH/FLAG nel testo renderizzato del report.** La prof non legge il
    codice. Via dal PDF: nomi di funzioni/generatori (`generate_*`), classi (`GraphBinaryClassifier`), script
    (`*.py`/`*.sbatch`), path di output (`runs/...`), field json (`disagree_frac`), flag (`--include_bridged`),
    fam-tag (`mixedbr`). Vanno tenuti nei **commenti `%`** del `.tex` (invisibili) o in queste istruzioni (§5).
    UNICA eccezione: i path in `\includegraphics`/`\figorbox` (necessari). Sostituisci il nome con la descrizione
    concettuale (es. "un classificatore binario: stesso trunk, mean-pool + 1 logit").
57. **NIENTE 'advisor'/attribuzioni personali, NIENTE storia/sequenza degli esperimenti.** (a) Non scrivere
    "suggested by the advisor / the advisor's question" → usa "this report asks / the question we investigate".
    (b) Non rivelare che un esperimento è stato fatto in due tempi: "we re-ran the capstone with the DFS oracle
    **added**", "originally we did only BFS" → presentalo come **un solo esperimento** ("we test against both
    traversals"). Al lettore non importa l'ordine cronologico; importa il disegno finale. (NB: i "re-run" che sono
    esperimenti **diversi** — es. classificatore n40, depth-sweep — restano legittimi.)
58. **La sezione "piano/esperimenti" è un RIASSUNTO TEMATICO con il verdetto in vista, non un elenco.** Non fare
    un elenco arido (3.1/3.2/3.3) che poi **duplica** l'intro dei Results ("we report, in order..."). Tieni: (i) il
    setup comune (modello), (ii) la **costruzione core** (qui: bridged vs split cliques) in un paragrafo, (iii) una
    **roadmap "in hindsight"** raggruppata per **macro-aree tematiche**, con i `\ref` alle sezioni. UNA sola
    roadmap (accorcia l'intro dei Results a una riga). Le definizioni che servivano a più sezioni (es. i 3 oracoli)
    si mettono UNA volta dove si introducono i meccanismi, non ripetute in una sotto-sezione "reference algorithms".
59. **Chiarezza locale: titoli-paragrafo PIANI (non enigmatici), simboli DEFINITI alla prima occorrenza anche in
    caption, e sweep discreti con XTICK ESPLICITI.** (a) Evita titoli gergali ("the cutoff is an absolute node
    budget, not a fraction of the graph") → fattuali ("the breaking point is the same clique size at n20 and
    n40"). (b) Se una caption usa `c`/`c^2`, scrivi cos'è `c` lì (es. "con `c` la clique size: `c` nodi per lato,
    quindi `c×c` coppie cross"). (c) **matplotlib su sweep di pochi valori discreti** (es. 5 densità
    {0.05,0.08,0.12,0.16,0.22}): `ax.set_xticks(valori)` + label esatte, NON la griglia continua auto a 0.025
    (estende errori 17/30). Posizioni numeriche vere (fedeli alla spaziatura) ticchettando solo i valori reali.
    Helper `style_p_axis` in `plot_density_sweep.py`.
60. **La firma di ASIMMETRIA DFS è un lever SECONDARIO/debole — non venderla come prova forte.** L'asimmetria
    dell'oracolo DFS è intrinseca ("raggiungibile da i entro budget b" non è simmetrica). È evidenza *genuina* solo
    perché il read-out **lineare** del RoBERTa **non è simmetrizzato** (`model.py:349`, errore 36) → un `R̂_ij≠R̂_ji`
    era architetturalmente POSSIBILE e il modello non lo mostra (disagreement≈0). MA un transformer parallelo è
    comunque predisposto a output simmetrico, quindi "zero asimmetria" sorprende poco: il **discriminatore FORTE è
    il TREND** clique-size (DFS-oracle SALE, modello SCENDE = opposto), l'asimmetria solo conferma. Se la prof
    chiede "perché l'asimmetria?": è la firma di un algoritmo sequenziale single-start; il transformer non ce l'ha.

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

> **🟦 AGGIORNAMENTO SLURM (2026-06-23, mail HPC "new partition configuration").** Lo
> scheduler è stato riconfigurato per favorire i job corti (82% dei job finisce <1h). Per
> OGNI famiglia di nodi ora ci sono 5 fasce, da **alta a bassa priorità per i job brevi**:
> - **`short_*`** (cap **1h10**, PRIORITÀ ALTA): `short_gpuh200` (8 nodi gnode09–16),
>   `short_gpunew` (4 nodi gnode05–08), `short_cpu`. **Usa QUESTE per gli eval-only**
>   (bridged eval ~1–3 min/ckpt, reeval ~17 ckpt ~25 min, plot-regen): partono subito.
> - **`medium_*`** (cap **6h10**): `medium_gpuh200` (8 nodi), `medium_gpunew` (4 nodi),
>   `medium_cpu`. Per i training **n20** (~1h40, non entra nei short).
> - **`gpunew`/`gpuh200`** (cap ridotto a **1 giorno**, era 24h/non-cap): `gpuh200` 6 nodi
>   (gnode13–16 + …), `gpunew` 4 nodi (gnode05–08). Per i training **n40** (~6–8h, sfora i 6h).
> - **`long_gpunew`/`long_gpuh200`** (cap **3 giorni**, era 72h): 2 nodi ciascuno
>   (`long_gpuh200` gnode15–16, `long_gpunew` gnode05–06). Quasi mai serve (errore 18).
> - **`debug_*`** (15 min, serve `--qos=debug`).
>
> **REGOLE pratiche:** (a) eval-only → `short_gpuh200` (più nodi → parte prima) o
> `short_gpunew`, con `--time=01:00:00`. (b) n20 training → `medium_*`. (c) n40 training →
> `gpunew`/`gpuh200` (1 giorno). (d) Spostare un job PD senza scancel:
> `scontrol update JobId=<id> Partition=short_gpuh200 TimeLimit=01:00:00` (ridurre il time è
> permesso, aumentarlo NO). (e) Resta vietata la **multi-partizione** (errore 12): UNA sola.
> NB: il job `brblocks` (eval bridged-blocks, §11 esperimento OOD) è eval-only → short.

- Partizioni GPU H200: `gpuh200`, `long_gpuh200`,
  `debug_gpuh200` (15 min, serve `--qos=debug`). H100: `gpunew`/`long_gpunew`/
  `debug_gpunew`. **`gpuh200` e `long_gpuh200` condividono gli stessi nodi fisici**
  (gnode09–16); H100 sono un pool separato (gnode05–08). (Cap aggiornati nel blocco
  AGGIORNAMENTO sopra: gpuh200/gpunew = 1 giorno, long = 3 giorni.)
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

## 11. Stato attuale e compito corrente

**Report 1–5 CONGELATI (consegnati).** Il Report 5 è stato mostrato alla prof ed è chiuso: non si
tocca più, salvo richiesta esplicita. Esito Report 5 (= verdetto §6, qui per riferimento): il
transformer L=2 risolve la connettività con **matrix-powering distance-bounded** (muro a 3^L=9), NON
una traversata bounded; DFS rifiutato; il "node budget" ~6–7 sui bridged è un'**euristica
soft/data-prior** (si sposta col read-out, sparisce coi bridged in training con transfer a blocchi
held-out, NON cresce con la profondità). Dettagli nel `.tex` e in §1.

**>>> COMPITO CORRENTE: REPORT 6** (`report/6/transformer_for_graphs_6.tex`, compilare da
`report/6/`). Tema: **"simulare una path di ragionamento" — quali dati servono per imparare un
ragionamento; sequenzialità vs parallelismo.** È il **report conclusivo**: **pochi esperimenti,
puntuali, vendibili in un paper.** **Indicazioni complete della prof + piano + tesi + mindset in
§13** (leggerlo PRIMA di agire). Vincolo nuovo: **niente mixed random in training** (§13).

**STATO al 2026-06-30 (fine 15a sessione).** Intro + framing + piano + setup nel `.tex` scritti. Lo stato
per thread (è una FOTOGRAFIA, non una to-do list: il lavoro aperto lo prenderanno chat diverse, senza
ordine imposto):
- **Thread A — FINITO, ANALIZZATO, SCRITTO** (§13.7 stato finale + lezioni). 44 `multipath.json` (A1) +
  56 `history.json` (A2/A3) in locale. Sezioni `.tex`: §sec:res-multipath-clean (A.1), -samples (A.2),
  -truncated (A.3). Figure in `runs/report6/report6_figs/`.
- **Thread B — FINITO, ANALIZZATO, SCRITTO** (§13.8). 36 `asym_chains.json` in locale. §sec:res-asym-chains:
  mostra SOLO i risultati (puzzle riprodotto: sbilanciato facile, bilanciato no; Fig exact-vs-split + 2 Tab
  by-split clean & ER + Fig per-distanza). **Il *perché* dello split è una DOMANDA APERTA** — l'utente ha
  scartato la "spiegazione" over-connect perché era una ri-descrizione, non una causa (§13.8).
- **Thread C — esperimenti FINITI, risultati PUSHATI, NON ancora analizzati/scritti** (§13.9). 48 json su
  origin/main in `runs/report6/{clique_chain,clique_chain_er,thick_bridges}/` (+ `c_train` history). Gli
  stub §sec:res-chain-cliques (C.1) e §sec:res-thick-bridges (C.2) sono ANCORA VUOTI.
- **Thread D (alberi)**: non implementato (esplorativo, §13.5).
- **Variante SIMILARITY read-out (A/B/C)**: gemello con `--readout similarity`, **Onda 1 (A1-sim) GIÀ
  LANCIATA** su HPC (job 551389–551392 train + 551400/551401 eval); Onda 2 (C-sim) e Onda 3 (A2/A3-sim) da
  lanciare. Output in cartelle `*_sim`. Tutto in **§13.10** (esplorativo, no stub `.tex` ancora).
- **Verdetto §sec:verdict**: da scrivere per ultimo, quando A–C sono tutti in.

**Lavoro ancora aperto (NON un ordine, solo la lista delle cose che restano):** (a) pullare in locale i 48
json di C e analizzarli/scriverli (mappa e strumenti in §13.9 — `plot_clique_chain.py`, figure di
`plot_bridged_cliques.py`); (b) eventuali NUOVI esperimenti (Thread D alberi, o varianti nuove); (c)
eventuale analisi più a fondo dei thread già fatti (es. il meccanismo aperto di B); (d) il Verdetto finale.

**Note git (la modalità, non un compito):** Claude NON committa/pusha, **PUÒ** `git pull`. I risultati HPC
si committano lato HPC e si pullano in locale; il lavoro di scrittura `.tex`/figure si committa in locale sul
Mac. I risultati di C (15a sessione) sono GIÀ stati pushati dall'utente. Il lavoro di scrittura di Thread B
(15a sessione: `report/6/*.tex,*.pdf`, `plot_asym_chains.py`, `runs/report6/report6_figs/asym_chains_*.png`,
`istruzioni.md`) potrebbe non essere ancora committato in locale → verificare con `git status` prima.

**Riuso da Report 4/5 (punti di partenza del Report 6):** Thread A (multipath) parte da
`eval_parallel_paths_clean.py` + `generate_parallel_paths_graph(n, n_paths, path_len)` (Report 4
§4.4); Thread C (bridged iterate) da `generate_bridged_cliques_graph`/`generate_split_cliques_graph`
+ `eval_bridged_cliques.py` e dai "bridged dense blocks" del Report 5 §5.6 (blocchi ER densi p≈0.6 +
ponte). Training puliti via `train_families_n20.py` (`--families`, `--n_nodes`, `--p`,
`--include_bridged`, `--n_layers`), **ma con UNA famiglia/distribuzione esplicita per run, non il
mixed**. Per i nomi-cartella e i path-output vedi §5.

**Idee future già citate (NON ora):** probe meccanicistico (`attention_maps`/`embeddings`) per
trasformare "somiglia a MP" in "calcola A^{3^L}".

---

## 12. Riferimento: regole per accorciare il Report 5 (pass principale FATTO)

Il pass di accorciamento §1–§6 è **FATTO** (31→26 pp.): prosa di contorno tagliata, tutte le
tabelle/figure e i numeri tenuti, caption complete, §3 "piano" accorpato (rimossa la sotto-sezione
§3.2 senza risultati propri), §5.7 era già il template del taglio. Queste regole restano qui per i
prossimi tagli, se l'utente ne chiede ancora:

1. Esperimento non-interessante/non andato → NON nei Results: una riga in un caveat. Se un
   esperimento esce, esce col suo risultato; non si tiene un esperimento amputandone il risultato.
2. Paragrafi da mezza pagina che spiegano UN concetto → 2–3 righe. Dire il concetto UNA volta;
   niente "this is the natural objection…", niente ri-spiegoni.
3. Spiegare semplice (cosa è una "coppia", il "cross block", perché conta) MA conciso.
4. **I RISULTATI non si tagliano MAI** — numeri chiave in grassetto/tabella + una frase di "cosa
   significa". Si taglia solo la prosa di contorno.
5. Tenere ogni tabella/figura interessante (NON quelle tutte-1.00 o tutte-0) e le caption complete
   MODELLO+TRAINING+TEST+METRICA (regola 21 — accorciabili, NON svuotabili).
6. Onestà intatta: i caveat veri (confound ER, seed-lottery, evidenza comportamentale) restano in
   forma compatta.
7. Dopo ogni sezione: ricompila da `report/5/` (2 passate), 0 reference indefinite, consegna i
   comandi git all'utente.

**Possibile passo successivo (a richiesta utente):** riordinare le sezioni del Report 5.

---

## 13. Report 6 — "path di ragionamento": indicazioni della prof, piano, tesi, mindset

> Blocco aggiunto dopo che l'utente ha mostrato il Report 5 alla prof. **Resta valido per OGNI nuova
> chat sul Report 6.** Sono le indicazioni della prof (tradotte e ripulite) più il piano operativo.
> Lo scheletro `.tex` è già in `report/6/transformer_for_graphs_6.tex`.

### 13.1 Indicazioni della prof (tradotte e ripulite)
Il tema su cui specializzarci è principalmente **simulare una "path di ragionamento"** e **capire
quali dati servono per imparare un certo ragionamento** → i concetti di **sequenzialità e
parallelismo**.

- **L'esperimento §4.4 del Report 4 (parallel paths) è piaciuto molto.** A diametro/distanza
  fissata, con due estremi `(a,b)`, se ci sono **più cammini** da `a` a `b` il modello impara
  meglio: forte per mostrare *come* un transformer impara i ragionamenti. **MA era fatto male**: un
  solo seed, dati poco sensati (alcune celle saturavano a 1.0). Va **rifatto multi-seed** e prima va
  **analizzato** quali combinazioni (numero di cammini `k`, lunghezza `ℓ`) sono davvero informative
  → poi decidere che esperimenti hanno senso. Chiamiamo **multipath** questi grafi (più cammini
  `a→b`).
- **Sample efficiency del multipath:** quanti samples di multipath servono per imparare
  **esattamente** la connettività — forse del solo pair `(a,b)`, forse dell'intera matrice (magari
  **entrambi**, da decidere dai dati).
- **Interessante anche il setting "mai visto in training":** come nel Report 4, allenare con **solo
  ER** (mai multipath in training) e valutare sul multipath.
- **Multipath troncati:** se nel training metto anche multipath **troncati** (un cammino tagliato
  che NON raggiunge `b`), questi **confondono** il training? (tesi: sì, va peggio).
- **Two-chains asimmetriche:** nel Report 4 risultava che due catene di lunghezza **(4, 36)** erano
  **più facili** di **(17, 23)**. Vanno **analizzati quei dati** e capito **perché**, con la lente
  "testare/simulare una path di ragionamento".
- **Bridged cliques iterate:** train con bridged cliques (come Report 5), test su
  **clique–ponte–clique–ponte–clique–…**. Piaceva perché simula un gruppo di "se e solo se": il
  modello deve trovare **qual è il nodo che passa al ponte** e quindi la clique successiva — un
  **passaggio forzato** ripetuto. Da fare per simmetria con **un solo edge**, ma per il nostro
  ragionamento ha senso anche con **più di un edge** nel ponte, **purché non si superi il diametro
  9**. Provare anche i grafi dell'ultimo esperimento del Report 5 (§5.6 "bridged dense blocks"): due
  **blocchi** di catene connesse con, tra i nodi interni al blocco, un edge a **p≈0.6** (una sorta
  di blocchi ER) + edge di ponte → vedere se più facile o più difficile.
- **Alberi:** provare qualche esperimento con grafi ad **albero**.
- **⚠️ Il mixed training NON è chiaro: non usarlo così randomico.** Non si possono scrivere paper
  mettendo grafi random in training. Deve essere **chiaro e sensato**: usare **ER**, oppure
  **cliques**, oppure **bridged cliques** (a seconda dell'esperimento), oppure **random graphs**, o
  altro — anche **più grafi insieme**, basta che la distribuzione sia esplicita e motivata.

### 13.2 Principio sui dati di training (NUOVO, non negoziabile)
Ogni esperimento si allena su **UNA distribuzione esplicitamente nominata** (ER, cliques, bridged
cliques, multipath, alberi, o una combinazione di poche **dichiarata e motivata**). **NON** usare di
default il "mixed" uniforme-su-9-famiglie dei Report III–V. Se si riusa un numero di un report
precedente che veniva dal mixed, dirlo e trattarlo da baseline, non da condizione pulita. Tecnicamente
`train_families_n20.py` resta lo strumento, ma lanciato con `--families <UNA>` (o una combinazione
esplicita), **mai** col mixing opaco.

### 13.3 Mindset
- È tra i **report conclusivi**: si cerca di arrivare alla conclusione, si vogliono **risultati spendibili/vendibili
  in un paper**. **Pochi esperimenti, puntuali**, ciascuno a sostegno di **una tesi precisa** — NON
  tanti esperimenti diversi come nei report passati.
- Onestà intatta (preferenze §3): risultati negativi puliti meglio di positivi sporchi; segnalare
  confound (ER/densità-OOD, seed-lottery), evidenza **comportamentale**.

### 13.4 Tesi da sostenere (lo spine del report)
1. **Più cammini paralleli aiutano** a risolvere la connessione, **anche senza averli mai visti in
   training**; e c'è un **costo in samples** misurabile quando invece i multipath sono in training.
2. **I cammini troncati nel training peggiorano** (confondono la skill di connessione).
3. **Non gestire i bridged cliques è "mai visto", non "troppo difficile":** è un buco OOD (come un
   ER troppo denso), non una difficoltà intrinseca — coerente col Report 5 §5.6 (coi bridged in
   training il modello li gestisce). Da spingere a versioni ripetute/variate.
4. **Un ponte imparato compone:** un modello che impara **una** bridged clique sa gestirla anche
   **ripetuta** (catene) e con **blocchi diversi** (es. ER densi), purché entro la capacità (≤9).
   *(Tutte da TESTARE — ipotesi, non ancora risultati.)*

### 13.5 Piano esperimenti (gruppi → vedi le sezioni-stub nel `.tex`)
- **Thread A — Parallelismo (multipath).** A.1 rifare il probe del Report 4 §4.4 multi-seed, prima
  mappando le celle `(k,ℓ)` informative; train **ER puro** (mai multipath), eval su multipath. A.2
  sample-cost: multipath **in** training, sweep del budget, quanti samples per connettività esatta
  (pair `(a,b)` e/o matrice intera). A.3 multipath **troncati** mescolati in training → peggiora?
- **Thread B — Two-chains asimmetriche.** Analizzare i dati del Report 4 ((4,36) vs (17,23)), poi un
  esperimento controllato; lettura "path di ragionamento".
- **Thread C — Bridged cliques iterate.** C.1 train su bridge singola (pulita, stile §5.6) → test su
  catene clique–ponte–clique–…; C.2 ponti **>1 edge** (distanza ≤9) e **blocchi ER densi** (p≈0.6)
  invece delle cricche.
- **Thread D — Alberi** (esplorativo, breve salvo risultato chiaro).
- **Verdetto** (scrivere per ultimo): per ciascuna tesi, se l'evidenza la sostiene.

### 13.6 Note tecniche da non dimenticare
- **multipath**: il generatore PULITO è ora `generate_multipath_graph(...)` in `data.py` (route
  piene + dead-end troncate + padding + filler + struttura per-route); il vecchio
  `generate_parallel_paths_graph` resta solo come riferimento Report 4. Vedi **§13.7** per tutto il
  Thread A (già implementato e lanciato): NON reimplementare.
- Per il Thread C tenere SEMPRE le distanze cross **≤ 9** (entro capacità 3^L), così il fallimento
  eventuale è "non visto"/propagazione, NON il muro-distanza.
- Riusare gli eval esistenti dove possibile (`eval_bridged_cliques.py` per il Thread C **solo per le
  varianti a DUE blocchi**, es. ponte spesso e blocchi densi C.2; le **CATENE** C.1 a K blocchi
  richiedono un eval NUOVO — vedi §13.9; per il Thread A è già fatto, §13.7) e mettere gli output nel
  bucket `runs/report6/...`.
- Valgono tutte le regole §4 (caption 21, no-codice 56, no-advisor/no-storia 57, ≤/≥ unicode nei
  titoli matplotlib, xtick espliciti 59) e le regole di scrittura/accorciamento §12.

### 13.7 Thread A — IMPLEMENTATO e LANCIATO (11a sessione, 2026-06-26/28). Codice, run, lezioni.
**Generatori condivisi** (in `data.py`, riusati da train ed eval — niente duplicazione):
`generate_multipath_graph(n, n_full, path_len, rng, n_trunc=0, term_deg=4, trunc_len=None)` → 2
terminali `s,t` + `n_full` route piene (distanza fissa `ℓ`) + opzionali `n_trunc` route **dead-end**
(non arrivano a `t`, per A3) + foglie di padding a grado fisso + filler path sparso; ritorna
`(adj, meta)` con la struttura per-route (per l'analisi del meccanismo) o `None` se non ci sta.
`permute_with_meta(adj, meta, rng)` permuta e rimappa gli indici. `s,t` connessi sse `n_full≥1`.
Fattibilità: `need = 2 + n_full·(ℓ−1) + n_trunc·trunc_len + 2·max(0,term_deg−n_full)` (e simmetrico
per i trunc) `≤ n`.

**Script (tutti nuovi salvo dove detto):**
- `eval_multipath.py` (A1, eval-only): sweep `(k,ℓ)`, per cella riporta **pair (s,t)** + matrice
  (exact/pairwise/active) + **meccanismo** (`n_intact_hist` = quante route piene risolte → single
  path vs multipath). Metrica primaria = pair; matrice sempre per contesto. Output
  `runs/report6/multipath/<tag>/multipath.json`.
- `experiments2/train_multipath.py` (A2/A3): stream multipath puro (`--trunc_frac 0`) o con frazione
  troncata (`--trunc_frac f`); logga su val pulito **pair (s,t) E matrice** vs step + `steps_to`.
  Output `runs/report6/multipath_train/n{N}_k{K}_ell{ELL}_{clean|trunc f}_.../history.json`.
- `experiments2/train_families_n20.py`: **esteso** per accettare una **singola famiglia esplicita**
  via `--families path_union` (oltre a `er`/`mixed`); usato per i training puliti A1. DataLoader reso
  robusto a `num_workers 0` (debug locale, err. 37).
- `plot_multipath.py` (locale, no-GPU): aggrega i json A1 per seed → curve rescue + figura meccanismo
  in `runs/report6/report6_figs/`.
- sbatch: `scripts/r6_a1_train.sbatch` (24 task: ER n10/n64 + path_union n10/20/40/64, 4 seed,
  auto-eval), `scripts/r6_a1_eval.sbatch` (eval-only, riusa ER n20/n40 esistenti + valuta i nuovi +
  mixed di riferimento; trova i `.pt` provando più path, err. 34), `scripts/r6_a2_samples.sbatch`
  (40 task), `scripts/r6_a3_truncated.sbatch` (16 task; baseline = i run trunc=0 di A2).
- **A1 distribuzioni**: **ER** (riuso n20=`repro_paper_n20_roberta` 8 seed, n40=`families_n40` 4 seed;
  nuovi n10 p=0.20, n64 p=0.03) + **path_union** (tutte le taglie, nuovo). L'effetto rescue (k≥2,
  ℓ>9) è fattibile **a n40 (k≤2–3) e pulito a n64 (k≤4)**; n10/n20 sono controlli within-capacity.

**Sequenza HPC (in ordine):**
1. push da locale (l'utente): `git add ...; git commit; git push`.
2. su HPC: `git pull`.
3. A1 training a blocchi (rispetta cap 4 GPU `%4`, partizioni per taglia):
   `sbatch scripts/r6_a1_train.sbatch` (SENZA `--array` → stampa la tabella ed esce; **NON**
   `--array=0`, che esegue il task 0!), poi
   `sbatch -p medium_gpuh200 --time=02:00:00 --array=0-7%4 scripts/r6_a1_train.sbatch` (n10),
   `... --time=04:00:00 --array=8-11%4` (n20 pu), `-p gpunew --time=12:00:00 --array=12-15%4` (n40 pu),
   `-p gpunew --time=14:00:00 --array=16-23%4` (n64). Ogni run auto-evala il proprio checkpoint.
4. A1 eval-only. **Ordine-indipendente**: i ckpt ER/mixed riusati esistono già; i nuovi sono
   auto-evalati dal training (punto 3). Per una passata FINALE completa e robusta (single source,
   regge anche se un training muore prima dell'auto-eval) gateala sul training con **afterany**
   (err. 44, non afterok): `sbatch --dependency=afterany:<JID_train1>:<JID_train2>:... scripts/r6_a1_eval.sbatch`.
   Un run anticipato (`sbatch scripts/r6_a1_eval.sbatch`) è innocuo: dà subito i risultati riusati e
   fa `-- skip` sui non ancora allenati. **Trappola lanci**: `--array=0` NON stampa la tabella (esegue
   il task 0, duplicando un indice già in un blocco) — la tabella si stampa SENZA `--array`.
5. A2: tabella SENZA `--array` (`sbatch scripts/r6_a2_samples.sbatch`), poi a blocchi
   `sbatch --array=0-23%4 ...` e `--array=24-39%4 scripts/r6_a2_samples.sbatch` (cap submit ~30).
6. A3: `sbatch --array=0-7%4` e `--array=8-15%4 scripts/r6_a3_truncated.sbatch`.
7. a fine job: su HPC committa `runs/report6/**/*.json *.png out/*.out` e push; in locale `git pull`,
   poi `python plot_multipath.py` (A1) per le figure. **Controlla il conteggio dei json** (gli eval
   saltano in silenzio i ckpt mancanti, err. 43).

**§4.4 → A1, l'analisi che ha guidato il design (NON ripeterla a vuoto).** Rileggendo i json del
Report 4 §4.4 (`runs/report4/families_n40/*/parallel_paths_clean/`): l'effetto "più route salvano"
(k=1 fallisce ~0.55, k=2 ~0.93, k=3 ~1.00) è netto **solo OLTRE capacità (ℓ>9) e solo sul modello
MIXED**; su **ER puro k=1 NON falliva** (0.88–0.94 a ℓ=11–15) perché l'ER è OOD sui grafi-path,
rumoroso e tende a **over-connettere** → la sua `pair_acc` non è segnale pulito. Inoltre la
**fattibilità geometrica** impedisce a n20 di ospitare ≥2 route lunghe (2 path da ℓ=11 = 26 nodi
>20): **il rescue è testabile solo a n40 (k≤2–3) e pulito a n64 (k≤4)**; n10/n20 restano controlli
within-capacity. Per questo A1 usa **ER (richiesta prof) + path_union** (vede path singoli → muro
netto a 9, ma MAI route parallele → il test pulito) e **n10/20/40/64**, con la metrica primaria sul
**pair (a,b)** + matrice di contesto + meccanismo per-route.

**STATO RUN (fine 11a sessione, 2026-06-28).** Tutto lanciato su HPC, coda quasi vuota.
- **A1 COMPLETO**: training 0–23 tutti COMPLETED (ER n10 seed1000-4000, path_union n10/20/40/64;
  ER n64), ognuno auto-evalato; eval anticipato (riuso ER n20=`repro_paper_n20_roberta` best.pt,
  n40=`families_n40` last.pt — **path trovati, riuso OK**) + eval finale gateato `afterany` (n64
  incluso). json in `runs/report6/multipath/n{N}_{er|pathunion|mixed}_seed{S}/multipath.json`.
- **A2 COMPLETO**: 0–39 COMPLETED (k-sweep k1–4 @ n40 ℓ7; ℓ-sweep ℓ5/9/11/13 @ n40 k2; n-sweep
  n20/n40/n64 k2 ℓ7). history in `runs/report6/multipath_train/n{N}_k{K}_ell{ELL}_clean_*/`.
- **A3 LANCIATO** (troncati): 0–13 sottomessi (config `n40 k3 ℓ7` e `n40 k2 ℓ9`, trunc 0.3/0.6);
  **mancano gli ultimi 2 task `--array=14-15`** (gli altri 2 seed di `n40 k2 ℓ9 trunc0.6`) — da
  mandare appena c'è budget. history nello STESSO dir di A2 (tag `trunc{f}` accanto al `clean`).
- **Da fare nella chat di analisi**: pull `runs/report6/` (commit lato HPC + `git pull` locale),
  `python plot_multipath.py`, poi gli snippet di lettura rapida qui sotto.

**Lezioni operative HPC (11a sessione, oltre §6/err. 11–12–18–44):**
- **`--array=0` NON stampa la tabella RUNS**: esegue il task 0 (`SLURM_ARRAY_TASK_ID="0"` ≠ vuoto).
  La tabella si stampa lanciando **senza** `--array`. (Capitato: un task 0 duplicato, poi scancel.)
- **Due cap distinti**: `QOSMaxSubmitJobPerUserLimit` ≈ **30 task sottomessi** (queue+run; il `%N`
  NON lo riduce) e `QOSMaxGRESPerUser` = **4 GPU concorrenti**. Riempire oltre i 4 in esecuzione non
  accelera nulla, serve solo a tenere la coda primata. Manda array ≤ ~25–30 per volta.
- **Pattern eval-after-train robusto**: l'auto-eval dentro ogni training copre i ckpt nuovi; un
  `r6_a1_eval` gateato `--dependency=afterany:<JIDs>` è la passata finale autorevole (rilegge i
  `last.pt` definitivi, regge a un training morto). Un eval anticipato è innocuo (`-- skip`).
- Tempi reali misurati: A1 n40 path_union ~3h20/run (1M step), **n64 ~5h/run**, n10/n20 più veloci;
  A2 n40 (300k step) **~50 min/run**.

**Lettura rapida dei risultati (read-only, ok su login — è roba da `cat`):**
- A1 (rescue + meccanismo): legge `runs/report6/multipath/*/multipath.json`, stampa per (n,dist) la
  tabella `pair_acc` [riga=ℓ, col=k] marcando ℓ>9, e per le righe oltre-capacità un verdetto
  `k1 -> best k` + se è "via 1 path" (mean_n_intact≈1) o "tutte le route". (Snippet completo dato in
  chat; in alternativa `python plot_multipath.py` in locale dopo il pull.)
- A2/A3 (sample-cost): legge `runs/report6/multipath_train/*/history.json`, stampa per config i
  `steps_to{pair_0.99, exact_0.99}` (mediana sui seed; samples = step×1000) e la `pair` finale →
  confronta k (più route = meno samples?), ℓ (oltre cap la matrice resta `—`), e clean-vs-trunc (A3).
- **Caveat di lettura**: scarta le celle **saturate** (pair=1.0 ovunque anche a k=1 dentro capacità)
  o **non informative**; la curva ER è il cross-check confuso (over-connessione), il test pulito è
  **path_union**. Onestà §3/§12.

**STATO FINALE Thread A (13a sessione, 2026-06-29): ANALIZZATO e SCRITTO nel `.tex`.** Cosa è andato
nel report (§sec:res-multipath-clean / -samples / -truncated) e cosa ho imparato:
- **A.1 — più route salvano la connessione, anche mai viste in training (TESI 1, regge).** Due
  condizioni PULITE: **path_union n40** (rescue monotono, tight sui 4 seed) e **ER n64** (la cornice
  prof "mai visto path"; serve il canvas grande `n64` perché a `n40` l'ER over-connette e maschera il
  fallimento). Oltre capacità: pair sale ~0.4→1.0 (n40 pu, k1→3) e ~0.1→1.0 (n64 er, k1→4). Tabella
  `tab:a1rescue` (pair per (k,ℓ), beyond-cap in grassetto). Confound dichiarati: n40 ER over-connette;
  **n64 path_union è NON-monotono** (k2<k1, k3 seed-bimodale) → route sottile in canvas grande è OOD;
  riportato in tabella per-seed `tab:a1n64`. Solo il pair è salvato, la matrice intera resta ~0 (filler).
- **MECCANISMO (cuore):** ogni capolinea fa crescere un **vicinato raggiungibile di profondità limitata**
  (~3-4 nodi a k=1); più route **spingono la reach più lontano** lungo la route. NON è "incontro in
  mezzo" — **scartato** dopo verifica (a ℓ=13 i vicinati NON si incontrano affatto, reach solo a 3-4).
  Figure `a1_mechanism.png` (pair vs reach vs route-intatte) e `a1_profile.png` (conn→s/conn→t per
  posizione, ℓ=9) + `a1_profile_far.png` (ℓ=13, k=1/2/3: a distanza maggiore servono più route).
- **A.2 — sample cost (TESI 1 2a metà, riformulata onestamente).** Il **pair si impara subito** (>0.99
  al primo checkpoint = 2M, `eval_every=2000` troppo grosso per separare k). Segnali puliti: (a)
  **samples per la MATRICE calano con k** (20→8M a ℓ7) — più route = più nodi connessi + filler più
  corto; (b) **collasso "predico tutto disconnesso"** quando la massa connessa è sottile (route corte):
  letto sullo sweep di **ℓ a k=2** (tengono il pair 1/4→4/4 da ℓ5 a ℓ13). ⚠️ Il collasso lungo **k è
  RUMOROSO** (colpisce k=2, risparmia k=1, con 4 seed) → NON vendere "più k = meno collassi". La metrica
  whole-matrix è confondata dalla **lunghezza del filler** (cambia con k,ℓ) TRANNE a (k,ℓ) fisso (→
  pulita per il confronto clean-vs-trunc di A.3). Tabella `tab:a2` con colonna **connected nodes** (=
  dim. componente di s,t, **include le foglie di padding**: 1 route ℓ7 → 2+6+6=14). Figure `a2_routes.png`
  e `a2_length.png` (small-multiples, **un colore per seed**).
- **A.3 — i troncati confondono (TESI 2, regge ma modesto).** A (k,ℓ) fisso, mettere route dead-end nello
  stream **rallenta/destabilizza** la matrice intera (k3ℓ7: 3/4 seed la risolvono ~15M nel clean → 2/4
  col troncato; ~3× più lento al 60%), il pair resta imparabile. Solo tabella `tab:a3` (la figura
  `a3_truncated.png` è stata TOLTA dal report su richiesta utente — il file resta su disco).

**File NUOVI/usati (13a sessione):**
- `plot_multipath_report.py` (NUOVO, locale no-GPU): figure curate del report A (rescue, mechanism,
  profile near+far, routes, length). Sostituisce le figure generiche di `plot_multipath.py` (che
  plottava celle sature + un pannello matrice inutile). I figure escono in `runs/report6/report6_figs/`.
- `eval_multipath_profile.py` (NUOVO, eval-only): per (k,ℓ) dumpa conn→s/conn→t per posizione lungo la
  route (overall + split pair-ok/no) → `runs/report6/multipath_profile/<tag>/profile.json`. Accetta più
  `--ell` e `--k`. Girato in LOCALE su Mac MPS sui `.pt` n40 path_union scaricati (vedi lezione scp).

**LEZIONI 13a sessione (oltre §4 errori):**
- **Claude PUÒ fare `git pull`** (non è commit/push). I risultati HPC, una volta committati lato HPC
  dall'utente, si pullano in locale (fast-forward; ha funzionato perché le modifiche locali non
  committate erano disgiunte dai file in arrivo `runs/report6/`).
- **Per diagnostici per-nodo servono i `.pt`** (non bastano i json aggregati): i checkpoint A1 sono su
  HPC in `runs/report6/a1_train/n{N}_{er|path_union}_roberta_linear_lam0_seed{S}/last.pt`. Per girare
  l'eval-profile li ho **`scp`-ati in locale** (`scp hpc:~/transformer-for-graphs/runs/report6/a1_train/
  .../last.pt /tmp/...`, ~24MB l'uno) e fatti girare sul **Mac (MPS)**. `ssh hpc` funziona
  **non-interattivo** (chiave/BatchMode), ma **MAI calcolo sul login node** (sono su `lnode01`) → scp +
  Mac, oppure `sbatch`.
- **⚠️ REGOLA `.pt` IN LOCALE (richiesta utente, 13a sessione): i checkpoint pesano tanto e NON li vuole
  in locale.** Quando servono per un diagnostico, scaricarli **SOLO in `/tmp`** (mai dentro il repo /
  `runs/...`, mai in cartelle persistenti), usarli, e **cancellarli appena finito** (`rm -rf /tmp/...`).
  I `.pt` restano gitignored e vivono solo su HPC; in locale tieni solo i **json/png leggeri** (es. i
  `profile.json` dei profili, ~4KB, bastano a rigenerare le figure senza GPU). Se una chat futura rifà un
  diagnostico per-nodo: ri-`scp` in `/tmp`, usa, ricancella.
- **PREFERENZE UTENTE confermate/raffinate (valgono per Thread B e oltre):** (a) **prosa SEMPLICE,
  termini definiti alla prima occorrenza** — l'utente ha respinto duramente gergo come "meet-in-the-
  middle", "redundancy", "rescue point", "connected mass", paragrafi densi → spiegare il meccanismo in
  modo banale (regola 20). (b) **Figure per-seed con un COLORE DISTINTO per seed** (mai stesso colore per
  i seed di una condizione → indistinguibili). (c) **Tabelle chiare su cosa è fissato/variato**, niente
  3 sweep impacchettati in modo confuso; aggiungere colonne che rendono ESPLICITO il meccanismo (es.
  "connected nodes"). (d) **Onestà**: scartare le cornici che non reggono (meet-in-the-middle) invece di
  difenderle; dichiarare il rumore (collasso seed-dependent). (e) L'utente verifica i numeri (es. "perché
  14 connected nodes?") → ogni numero in tabella deve poter essere giustificato e spiegato in caption.

### 13.8 Thread B — ANALIZZATO e SCRITTO (14a sessione, 2026-06-29). Two-chains asimmetriche.
**STATO FINALE (14a sessione):** ✅ ANALIZZATO e SCRITTO in §sec:res-asym-chains (prosa + Fig
`asym_chains_exact_n40.png` (headline) + Tab `tab:basplit` (n40 clean by-split) + Fig
`asym_chains_perdist_n40_pathunion.png` (meccanismo)). Report 6 compila pulito (≈13 pp.).
**ESITO (la sezione è SOLO RISULTATI, nessuna spiegazione — n40 clean = path_union, 4 seed, TIGHT):**
il puzzle si riproduce — split sbilanciato (4,36) exact≈1.0, bilanciato (17,23)/(20,20) exact=0; exact
≈1 per a≤7, crolla a 0 da a≈8. Osservazione forte nel profilo per-distanza: sugli sbilanciati il modello
marca connesse TUTTE le coppie nel lungo **fino a 35–38 hop**, sui bilanciati lo STESSO modello (stessi
pesi) è perfetto ≤9 ed esatto 0 da 10 (il muro 3^L=9). Cioè connette coppie lontanissime nel lungo dello
sbilanciato ma non nel bilanciato, a parità di n e pesi. Numeri per-split in tab (exact/reachL/cut/Lblock):
reachL e Lblock alti per a≤7 e cadono da a=8; cut≈1 tranne a=8,9.
**⚠️ NIENTE MECCANISMO (richiesta utente, 15a sessione).** Avevo scritto una "spiegazione" (il modello fa
over-connect / "stessa massa = connesso indipendente dalla distanza" / evita di ragionare) → l'utente
(giustamente) l'ha **bocciata: NON è una spiegazione, è solo una ri-descrizione del dato**. Al momento NON
abbiamo un perché. La sezione ora **mostra i risultati e basta**, con una riga esplicita che il "perché"
è lasciato a un'analisi successiva. **Il "perché lo split cambia il comportamento" è una DOMANDA APERTA
per una chat futura dedicata** (l'utente la farà a parte). Non reintrodurre la storia over-connect senza
una prova meccanicistica vera (probe su embedding/attention).
**Cross-check (scritti, fattuali):** ER (mai visto cammini) stessa ordinazione ma più rumoroso, picco a
a≈3–4 (0.72 a 4+36); **mixed (R4)
= seed-lottery** a (4,36) (1 seed su 4 a 0.76, gli altri ≈0) → per questo il read primario è il clean,
non il mixed. n64 stesso pattern ma seed più sparsi. n20 tutto entro capacità (distanze ≤9) → muro
assente, quasi tutto risolto (coerente con R4).
**Figure relabel (regola 56):** `plot_asym_chains.py` COND_LABEL ora "disjoint paths (clean)" (era
"path_union"). **File toccati (per commit utente):** `report/6/transformer_for_graphs_6.{tex,pdf}`,
`plot_asym_chains.py`, `runs/report6/report6_figs/asym_chains_*.png`, `istruzioni.md`.
**(storico) STATO 2026-06-29 pre-analisi:** eval COMPLETED (job 547018), 36 json pullati.
**Cosa fa.** Trasforma il puzzle del Report 4 (split (4,36) sembrava più facile di (17,23) a n40)
in una **leva controllata**: sweep esplicito dello split `(a, n−a)` a `n` fisso, **eval-only sui
checkpoint puliti già esistenti** (nessun retrain → parte appena c'è una GPU libera). Per ogni
split misura, BY SPLIT (mai collassato in un numero, errore 19): `exact` (intera matrice),
`reach_long`/`reach_short` (pairwise within componente lunga/corta), `cut` (cross, target 0),
`long_block_exact`/`short_block_exact`/`cut_block_exact` (dove si rompe l'exact), e il
**profilo per-distanza nella componente lunga** (muro≤9 → valle → recupero) — è ciò che spiega il
meccanismo "un solo lungo reach vs due reach al confine di capacità". Lettura: NON assumere il
meccanismo, leggilo dalle curve (es. la lunga quasi-piena recupera end-to-end → `long_block_exact`
alto; due medie restano in valle → basso).
**Distribuzioni (principio §13.2, una nominata per checkpoint):** **path_union** (clean, il read
pulito; Thread A1 n20/40/64) + **ER** (mai visto cammini, cross-check; riuso n20
`repro_paper_n20_roberta` 8 seed, n40 `families_n40` 4 seed, n64 da Thread A1) + **mixed** (baseline
R4; `families_n{20,40}` 4 seed). NON serve nuovo training.
**File nuovi:** `data.py::generate_split_chains_graph(n, short_len)` (due path che partizionano tutti
gli n nodi in `short_len` e `n−short_len`, niente padding isolato; `short_len=n//2` = il 2chains
bilanciato); `eval_asym_chains.py` (eval-only, sweep split, remap della permutazione → metriche
vettoriali; output `runs/report6/asym_chains/<tag>/asym_chains.json`); `scripts/r6_b_eval.sbatch`
(short partition, ~36 ckpt, pattern `find_ckpt`/`run` come r6_a1_eval); `plot_asym_chains.py` (locale,
no-GPU → `runs/report6/report6_figs/asym_chains_{exact,blocks,perdist}_*.png` + tabella per-split).
**Lancio:** `sbatch scripts/r6_b_eval.sbatch` (eval-only, niente `--array`). Se sfora il cap short
(1h10), commenta il blocco n64 o una condizione e ri-sottometti (è additivo, salta i ckpt mancanti,
errore 43 → conta i json). **Da fare poi (chat analisi):** pull `runs/report6/asym_chains/` →
`python plot_asym_chains.py` → scrivere §sec:res-asym-chains (caption regola 21, no-codice 56).
Smoke-test locale fatto (generatore, metriche, remap permutazione, plot).

### 13.9 Thread C — ANALIZZATO e SCRITTO (17a sessione, 2026-06-30). Codice, run, mappa, esito.
**STATO FINALE (17a sessione):** ✅ ANALIZZATO e SCRITTO in §sec:res-chain-cliques (C.1) e
§sec:res-thick-bridges (C.2). Report 6 compila pulito (18 pp, 0 ref indefinite, 0 overfull).
**ESITO (la tesi 4 "un ponte imparato COMPONE" regge solo in forma DEBOLE):** il modello è trainato su una
distribuzione minima e pulita — DUE cliques unite (o no) da UN ponte, clique-size random 50/50 (`--families
bridged,split`, NON il mixed) → ogni successo su strutture diverse è composizione, non diversità vista.
- **C.1 catene cricche (clique_chain):** K=2 (allenato) risolto a ogni size; ma AGGIUNGERE blocchi degrada in
  fretta — K=3 exact 0.38, K=4 0.16, K=5 0.00 (n40, c=3). **Lettura-chiave (pulita):** la cross-acc è ~PIATTA
  nel gap g (vicino e lontano falliscono insieme) ma il LIVELLO scende col numero di blocchi K → il limite è
  la **quantità TOTALE di struttura** da attraversare, NON la distanza/numero ponti. Prova: il hand-off più
  vicino (gap1, SEMPRE distanza 3) decade con K (1.0→0.86→0.53→0.44) benché la sua distanza non cambi.
  within-block resta alto (legge i blocchi) → è under-connect (default "disconnesso" su catene lunghe mai
  viste), non over-connect. Collassa prima per blocchi grandi (= node budget di Report V contato in nodi).
- **C.1 catene blocchi ER densi (clique_chain_er):** stessa storia, prima (within-block dist ≈2 → cap a K più
  piccolo) + costo di tipo-blocco (held out).
- **C.2 ponti SPESSI (thick_bridges cliques bw1/2/3):** **NESSUN effetto** — la cricca singolo-ponte è già
  risolta a ogni size (cross≈1.0, disc 1.0, simmetrico, asym 0.0), ponti ridondanti non aiutano né danneggiano.
  Contrasto con Thread A: lì le route parallele salvavano una connessione OLTRE capacità; qui il ponte è ≤3 hop
  (DENTRO capacità) → un edge basta, niente da salvare.
- **C.2 blocchi ER (thick_bridges blocks bw1):** **transfer PARZIALE** — cross decade col size a floor ~0.62
  (n40)/~0.76 (n20), disc→chance; a n40 c=20 il WITHIN-block crolla a ~0.55 in 3/4 seed (il blocco ER da 20
  nodi è esso stesso OOD per un modello allenato solo su cricche) → la domanda-ponte è moot. L'asimmetria 0.36
  è SINTOMO del within rumoroso, NON una firma sequenziale (errore 60). "Unseen, non too hard" → fixabile coi
  dati (Report V §5.6).
**File nuovi/toccati (per commit utente):** `report/6/transformer_for_graphs_6.{tex,pdf}` (C.1+C.2 scritte),
`plot_thick_bridges.py` (NUOVO, locale no-GPU → `thick_bridges_n{20,40}.png`), figure rigenerate
`runs/report6/report6_figs/{clique_chain,clique_chain_er}_{composition,length}_n{20,40}.png` (con `--clique_size 3`)
+ `thick_bridges_n{20,40}.png`, `istruzioni.md`. Figure nel `.tex`: `clique_chain_composition_n40.png` (C.1
headline, c=3) e `thick_bridges_n40.png` (C.2). Tabelle: tab:c1chain (n40 c=3 K-decay), tab:c1chainer (ER),
tab:c2 (full-size cliques bw1-3 vs ER blocks).
**(storico) Stato pre-analisi 2026-06-30: train + eval COMPLETED, 48 json su origin/main, sezioni .tex vuote.** Train
n20 4/4 (job 547043) + n40 4/4 (job 547044, ultimo seed finito 2026-06-30T05:53) → 8 ckpt
`runs/report6/c_train/n{20,40}_bridged+split_*/last.pt` (gitignored, restano su HPC). Eval `547048`
(`r6ceval`) COMPLETED 2026-06-30T09:19, **48 json pushati** (8 ckpt × 6 condizioni). Gli stub
§sec:res-chain-cliques (C.1) e §sec:res-thick-bridges (C.2) sono ANCORA VUOTI → analisi+scrittura aperte.
**Mappa risultati (48 json, tutti in `runs/report6/`):** C.1 catene cricche →
`clique_chain/n{N}_bridgedsplit_seed{S}/clique_chain.json` (8); C.1 catene di blocchi ER densi →
`clique_chain_er/.../clique_chain.json` (8); C.2 ponti spessi bw∈{1,2,3} →
`thick_bridges/n{N}_cliques_bw{W}_seed{S}/bridged_cliques.json` (24); C.2 blocchi ER densi ponte singolo →
`thick_bridges/n{N}_blocks_bw1_seed{S}/bridged_cliques.json` (8). N∈{20,40}, seed∈{1000..4000}. NB: nelle
catene le celle con diametro >9 sono saltate apposta (trappola #2) → l'`out` ha ~112 skip di CELLA (non di
checkpoint: i 48 json ci sono tutti). **Strumenti analisi (locale, no-GPU, dopo `git pull`):**
`python plot_clique_chain.py` (+ `--block er` per le catene ER), figure di `plot_bridged_cliques.py` per i
ponti spessi. **Tesi che il thread testa (§13.4 #3–4):** un ponte imparato **compone** (ipotesi, da
verificare sui dati — non assumere l'esito).

**Distribuzione di training (pulita, §13.2):** `--families bridged,split` (due cliques, clique-size
random 2..n/2, ±1 ponte, 50/50) — la distribuzione MINIMA che contiene la decisione "iff il ponte
c'è", positivo+negativo e nulla più → ogni successo su catene/ponti-spessi/blocchi-densi è
COMPOSIZIONE, non diversità vista. **NON** riusa `mixedbr` (trappola #1). Base RoBERTa linear L2 d512.

**File nuovi/estesi (smoke-test locale OK):**
- `data.py`: aggiunto `bridge_width` a `generate_bridged_cliques_graph` e `generate_bridged_blocks_graph`
  (w archi di ponte fra le due cliques, distanza cross ≤3). NUOVO `generate_clique_chain_graph(n,
  clique_size, n_cliques, rng, bridge_width=1, block="clique"|"er", p_in=0.6, broken_link=None)` — K
  blocchi in fila, K−1 ponti, 1 componente; `broken_link=l` droppa un ponte → 2 componenti (negativo
  anti-"predici-tutto-connesso").
- `eval_clique_chain.py` (NUOVO, eval-only, C.1): sweep (n_cliques, clique_size); `cross_by_gap[g]` =
  acc cross fra blocchi a g ponti di distanza (il cuore: la connessione propaga o decade coi link?),
  `per_link`, `exact`, `chain_block_exact`, **catena rotta** (cut-acc target 0 + discrimination
  intatta/rotta), `max_dist`/`within_capacity` (calcola APSP, **salta** le celle con diametro >9 —
  trappola #2). Output `runs/report6/clique_chain/<tag>/clique_chain.json`. Supporta `--block er` (blocchi
  ER densi incatenati). Tag `n{N}_bridgedsplit_seed{S}`.
- `eval_bridged_cliques.py`: aggiunto `--bridge_width` (passato a `_gens`/`build` via `partial` sul solo
  generatore bridged; split non ha ponte) per il sweep clique-size per spessore-ponte (C.2).
- `plot_clique_chain.py` (NUOVO, locale no-GPU): aggrega i json catena per seed → figura COMPOSIZIONE
  (cross-acc vs #ponti, una linea per K), figura LUNGHEZZA (exact + end-to-end vs K), GUARD
  (discrimination vs K), tabella per-cella. `--block er` legge `clique_chain_er/`. → `runs/report6/report6_figs/`.
- `scripts/r6_c_train.sbatch` (parametrico N: 0-3 n20, 4-7 n40; tabella SENZA `--array`),
  `scripts/r6_c_eval.sbatch` (eval-only `medium_gpuh200`: C.1 catena cliques + catena blocchi-ER +
  C.2 ponti spessi bw∈{1,2,3} + blocchi densi bw1, su tutti i ckpt bridged+split).

**Sequenza lancio HPC (ordine):** (1) locale: `git add ...; git commit; git push`. (2) HPC: `git pull`.
(3) train: `sbatch scripts/r6_c_train.sbatch` (stampa tabella), poi
`sbatch -p medium_gpuh200 --time=03:00:00 --array=0-3%4 scripts/r6_c_train.sbatch` (n20),
`sbatch -p gpunew --time=12:00:00 --array=4-7%4 scripts/r6_c_train.sbatch` (n40). (4) eval gateato:
`sbatch --dependency=afterany:<JID_n20>:<JID_n40> scripts/r6_c_eval.sbatch`. (5) a fine job: HPC committa
`runs/report6/**/*.json *.png out/*.out` + push; locale `git pull` → `python plot_clique_chain.py`
(e `--block er`) + le figure di `plot_bridged_cliques.py` per i ponti spessi. **Conta i json** (err. 43).
**Da fare poi (chat analisi):** scrivere §sec:res-chain-cliques (C.1) e §sec:res-thick-bridges (C.2),
caption regola 21, no-codice 56. Note: a n20 le catene cliques fattibili ≤cap arrivano a K=5 (c=4, diam 9);
i blocchi ER incatenati superano spesso il cap (diam cresce >9) → molte celle saltate, è atteso.

**Brief storico (com'era prima dell'implementazione):** un ponte imparato
**compone** — un modello che impara UNA bridged clique pulita la gestisce **ripetuta** (catene
clique–ponte–clique–…) e **variata** (ponti >1 edge, blocchi ER densi), **purché ogni distanza cross
resti ≤9** (capacità 3^L). Stub `.tex` già pronti: §sec:res-chain-cliques (C.1), §sec:res-thick-bridges
(C.2).

**⚠️ TRAPPOLA #1 — principio dati (§13.2): NON riusare i checkpoint `mixedbr` del Report V §5.6**
(`runs/report5/train_bridged_in_stream/n*_mixedbr_*`): sono **mixed+bridged**, violano §13.2. Allena
**fresco** su una distribuzione bridged **pulita e nominata**: `experiments2/train_families_n20.py
--families bridged,split` (il trainer accetta liste esplicite; `bridged`/`split` sono famiglie note →
cartella `n{N}_bridged+split_roberta_linear_lam0_seed{S}` in `runs/report6/...`). Decisione di design
da motivare (NON il mixed opaco): bridged+split puro, oppure + `path_union`/`2cliques` per dare
reach-within-block e cut espliciti.

**Cosa ESISTE e si riusa (NON reimplementare):**
- `data.py`: `generate_bridged_cliques_graph`/`generate_split_cliques_graph` (2 cliques ± 1 edge);
  **`generate_bridged_blocks_graph(n, rng, clique_size, bridged, p_in=0.6)` = i BLOCCHI ER DENSI di
  C.2, GIÀ pronti**.
- `eval_bridged_cliques.py --family {cliques,blocks}` (eval-only, n auto): discrimination, per-blocco
  within-A/within-B/cross, asimmetria, **sweep clique-size vs oracoli MP/BFS/DFS**. **Gestisce SOLO la
  struttura a DUE blocchi** (roles 0/1/pad). + `scripts/eval_bridged_blocks.sbatch` (template lancio).

**Cosa MANCA (da creare):**
- **C.1 generatore CATENA** `generate_clique_chain_graph(n, clique_size, n_cliques, bridge_width=1)`:
  K cliques in fila, ogni coppia adiacente unita da un ponte (1+ edge), UN componente. NON esiste (il
  bridged fa solo 2 cliques).
- **C.1 eval NUOVO** `eval_clique_chain.py`: `eval_bridged_cliques.py` è cablato a 2 blocchi → NON vale
  per K>2. Misurare **per-LINK** se la connessione propaga attraverso ciascun ponte (cross-block tra le
  cliques i,i+1), + exact intera matrice, + sweep su n_cliques/clique_size; output
  `runs/report6/clique_chain/<tag>/...`.
- **C.2 ponte SPESSO**: nessun generatore supporta bridge_width>1 (il ponte è 1 edge hardcoded, riga
  `# single bridge edge` in `generate_bridged_cliques_graph`/`generate_bridged_blocks_graph`).
  Aggiungere il parametro a `generate_bridged_cliques_graph`/`generate_bridged_blocks_graph`
  (più edge fra le due cliques, distanza cross sempre ≤3), poi riusare `eval_bridged_cliques.py` per il
  sweep clique-size per bridge_width.

**⚠️ TRAPPOLA #2 — la distanza ≤9 LIMITA la lunghezza catena (§13.6).** Con ponti a 1 edge la distanza
fra due cliques adiacenti è ~2–3 hop, e end-to-end fra le cliques estreme cresce ~2 hop per clique
aggiunta (≈ 2(K−1)+1) → entro 9 entrano solo ~K=4–5 cliques (dipende da clique_size e densità del
blocco). Il generatore/eval **DEVE calcolare la distanza max esatta** (APSP) e scegliere le celle
(clique_size, n_cliques) **fattibili** — esattamente come la feasibility del Thread A (§13.7). Se la
distanza supera 9 il fallimento è il muro-distanza, NON la propagazione "mai vista" → snatura il test.

**Pattern (come A/B):** distribuzione pulita, multi-seed, eval-only dove si può, output in
`runs/report6/...`, lanci eval-only su **short partition**, plot locale no-GPU; caption regola 21,
no-codice 56, ≤/≥ unicode e xtick espliciti (59).

### 13.10 Variante SIMILARITY READ-OUT dei Threads A/B/C (16a sessione, 2026-06-30). Codice, lancio, mappa.
**Cosa è.** Il **gemello similarity** degli esperimenti A/B/C: identici ai linear ma con
`--readout similarity` nel modello (read-out `R̂_ij = scale·cos(h_i,h_j)+bias` invece del lineare).
Scopo: vedere **come vanno A/B/C cambiando solo la testa**. È un confronto similarity-vs-linear
**esplorativo** — NON c'è ancora uno stub `.tex`; se/quando entra nel report lo decide l'analisi.
**Perché interessa** (non è ripetizione): la similarity **raddoppia il reach a `2·3^L=18`** (Report IV §sec:readout)
→ su **A** la soglia di rescue dovrebbe spostarsi; su **B** il bilanciato 20+20 (max dist within=19) cade
**proprio sul muro raddoppiato** (a 18); su **C** il node-budget bridged si sposta (Report V §5.5: knee 7→11–12
a n40) → le catene potrebbero **comporre più a lungo**.

**Principio di non-collisione (importante).** Gli eval auto-rilevano il read-out dal checkpoint
(`eval_families.load_model` legge la chiave `sim_scale` nello state_dict), quindi gli **stessi eval**
valgono. I checkpoint similarity convivono coi linear nello **stesso** OUT_ROOT perché il run-name porta
`_similarity_` (vs `_linear_`). Gli **output json degli eval** invece NON codificano il read-out nel tag →
vanno in **cartelle `*_sim` separate** per non sovrascrivere i json linear.

**7 nuovi sbatch** (in `scripts/`, gemelli 1:1 dei linear; pushati, commit `2d782eb`):
- `r6_a1_train_sim.sbatch` (24 task, `--readout similarity`): ER n10/n64 + path_union n10/20/40/64, 4 seed;
  auto-eval del proprio ckpt → `runs/report6/multipath_sim/`. ER n20/n40 NON allenati (riusati nell'eval).
- `r6_a1_eval_sim.sbatch` (eval-only): eval_multipath sui ckpt similarity → `runs/report6/multipath_sim/`.
  **Riusa i ckpt similarity di Report IV** (ESISTONO su HPC, verificato: ER n20 8 seed in
  `runs/report4/families_n20/n20_er_roberta_similarity_lam0_seed{1000..8000}`, ER/mixed n40 4 seed in
  `runs/report4/families_n40/...similarity...`; candidati anche al path non-bucketato `runs/families_n*`).
- `r6_a2_samples_sim.sbatch` (40 train, `--readout similarity`) e `r6_a3_truncated_sim.sbatch` (16 train):
  stessa OUT_ROOT `runs/report6/multipath_train/` (run-name porta `_similarity_` → niente collisione coi linear).
- `r6_b_eval_sim.sbatch` (eval-only): eval_asym_chains sui ckpt similarity → `runs/report6/asym_chains_sim/`.
  Riusa path_union-sim (da A1-sim) + ER/mixed-sim di Report IV; **dipende dai train A1-sim** (path_union + ER n64).
- `r6_c_train_sim.sbatch` (8 train, `--readout similarity`, `--families bridged,split`) → ckpt
  `runs/report6/c_train/n{N}_bridged+split_roberta_similarity_lam0_seed{S}/last.pt`.
- `r6_c_eval_sim.sbatch` (eval-only): C.1 catene + C.2 ponti spessi/blocchi ER → cartelle `*_sim`:
  `runs/report6/{clique_chain_sim,clique_chain_er_sim,thick_bridges_sim}/`.

**Mappa output (tutto in `runs/report6/`):** A1-sim eval → `multipath_sim/n{N}_{er|pathunion|mixed}_seed{S}/multipath.json`;
A2/A3-sim history → `multipath_train/n{N}_k{K}_ell{ELL}_{clean|trunc f}_roberta_similarity_*/history.json`;
B-sim → `asym_chains_sim/<tag>/asym_chains.json`; C.1-sim → `clique_chain_sim/` + `clique_chain_er_sim/`;
C.2-sim → `thick_bridges_sim/`. I `.pt` (gitignored) restano in `runs/report6/{a1_train,c_train}/...similarity...`.

**STATO LANCIO (2026-06-30, 16a sessione).** ✅ **Onda 1 = A1-sim LANCIATA**: train job **551389** (n10 0-7,
medium_gpuh200 2h), **551390** (n20 pu 8-11, medium 4h), **551391** (n40 pu 12-15, gpunew 12h), **551392**
(n64 16-23, gpunew 14h), tutti `%4`; eval gateati **551400** (`r6_a1_eval_sim`, afterany:551389:551390:551391:551392)
e **551401** (`r6_b_eval_sim`, afterany:551390:551391:551392), entrambi PD su short_gpuh200. ⚪ **NON ancora
lanciati:** Onda 2 = **C-sim**, Onda 3 = **A2/A3-sim**.

**Comandi residui per una nuova chat (HPC, dopo `git pull`; vincoli err. 11/18: ≤~30 task submit, 4 GPU run).**
Onda 2 — Thread C-sim (8 train + eval gateato):
```
sbatch scripts/r6_c_train_sim.sbatch                                                  # stampa tabella, esce
sbatch -p medium_gpuh200 --time=03:00:00 --array=0-3%4 scripts/r6_c_train_sim.sbatch   # n20
sbatch -p gpunew         --time=12:00:00 --array=4-7%4 scripts/r6_c_train_sim.sbatch    # n40
sbatch --dependency=afterany:<JID_C_n20>:<JID_C_n40> scripts/r6_c_eval_sim.sbatch
```
Onda 3 — A2/A3-sim (più pesante, a blocchi quando la coda si svuota):
```
sbatch scripts/r6_a2_samples_sim.sbatch                                  # tabella (40 task)
sbatch --array=0-23%4  scripts/r6_a2_samples_sim.sbatch
sbatch --array=24-39%4 scripts/r6_a2_samples_sim.sbatch
sbatch scripts/r6_a3_truncated_sim.sbatch                                # tabella (16 task)
sbatch --array=0-15%4  scripts/r6_a3_truncated_sim.sbatch
```
**A fine job:** HPC committa `runs/report6/**/*.json *.png out/*.out` + push; locale `git pull`. **Conta i json**
(eval saltano in silenzio i ckpt mancanti, err. 43). **Analisi/plot:** gli script `plot_multipath_report.py`/
`plot_asym_chains.py`/`plot_clique_chain.py` + le figure di `plot_bridged_cliques.py` vanno **puntati alle
cartelle `*_sim`** (vanno estesi/parametrizzati: ora leggono i path linear). Confronto chiave da leggere:
soglia rescue (A), muro raddoppiato a 18 (B), knee/composizione delle catene (C) — similarity vs linear.

---

*Per aggiungere questo file a git (lo fa l'utente):*
`git add istruzioni.md && git commit -m "Update project handoff instructions" && git push origin main`
