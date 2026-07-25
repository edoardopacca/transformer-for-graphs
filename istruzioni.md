# Istruzioni del progetto — handoff per Claude

Documento di riferimento per chiunque (incluso Claude in una nuova chat) riprenda
questo progetto. Leggere **tutto** prima di agire.

> **⚠️ REGOLA DI LETTURA NON NEGOZIABILE (prima di toccare qualsiasi cosa).** Leggere
> **attentamente OGNI riga** di questo `istruzioni.md` e **OGNI riga dei report** dai
> sorgenti `.tex`; il report 1 è solo PDF). In particolare l'ultimo report va **riletto 4 VOLTE riga per
> riga** prima di scriverci dentro: è lungo, denso, e gli errori più gravi di questa sessione
> sono nati dal **non aver letto tutto** . Non saltare righe perché "sembrano un dettaglio": le caption, i `%`
> commento negli stub, e le righe-errore qui sotto contengono i vincoli che fanno fallire chi
> non li legge. Non rispondere/agire da una vista parziale di un file: se un `Read` è troncato,
> continuare fino in fondo.

---

> **🟥 STATO ATTUALE (AGGIORNATO 2026-07-25) — Report 1–8 CHIUSI. Report 9 APERTO: il PIANO è scritto
> (§16); DUE pezzi hanno CODICE PRONTO (il sanity-check n=46 e il Thread A.4, §16.6), LANCIATI** — il
> Thread A.4 (job `604793`, CPU) è partito correttamente, in attesa che finisca; i due training n=46
> sono invece **falliti alla sottomissione** (partizione `gpunew` nuda sparita dal cluster, errore 68),
> **fix fatto** (`gpuh200` al posto di `gpunew`, §16.7) — **da ri-lanciare dall'utente**. Ancora
> **nessun dato reale tornato** da nessuno dei tre. Onboarding standard per la nuova chat: questo file
> per intero + tutti i `.tex` dei report, incluso `report/9/transformer_for_graphs_9.tex` (piano/
> scheletro + le due sezioni ora "code pronto, dati in arrivo"), come da regola di lettura in cima al
> file.
>
> **Perché nasce il Report 9 (dettaglio completo, richiesta VERBATIM della prof, in §16.0 — leggerlo
> per intero prima di qualunque altra cosa).** Il Report 8 (path\_union-trained, readout similarity,
> n40) ha mostrato uno split sbilanciato risolto ben oltre il muro raddoppiato $2\cdot3^L=18$ — ma
> quel successo è misurato **dentro** la stessa distribuzione di training (le disjoint-paths coprono
> già ogni split a due segmenti), quindi non distingue un meccanismo genuinamente generale (gli
> estremi/endpoints) da una semplice copertura in-distribution. Il Report 9 punta a isolare un caso di
> **vera generalizzazione OOD** su path (Thread A, §16.3) e a ripetere lo stesso esperimento sui
> **cicli** (Thread B) per capire se gli estremi sono davvero necessari o se anche i cicli, allenati
> per bene, possono contraddire "il diametro è la misura di difficoltà" — prima di concludere che
> l'architettura non impara proprio i cicli. Piano completo, sette esperimenti pianificati (A.1–A.5,
> B.1–B.2), in **§16.3**.
>
> **⚠️ Nota canvas size, NON negoziabile per OGNI training di questo report**: si usa **n=46** (non
> n40) — stesso identico setup del Report 8 (disjoint-paths training, readout similarity, stesso
> numero di sample, stesso modello RoBERTa-faithful $L{=}2$/$d_{\mathrm{model}}{=}512$/single-head),
> **cambia solo la dimensione del canvas**. Riusare il codice esistente (già parametrico in
> `--n_nodes`/`--n`, vedi §5/§9/§14/§15), non reimplementare da zero. Dettaglio in **§16.4**.
>
> **Stato Report 8 (INVARIATO, solo per riferimento — non toccarlo se non richiesto esplicitamente):**
> `report/8/transformer_for_graphs_8.tex`, compila pulito, **22 pagine**, 0 ref indefinite, 0 overfull —
> tema: dove/come l'informazione di connettività viene combinata nel trunk; readout **similarity**
> standard. Dettaglio completo in **§15** (**§15.6** = risultati finali del filone principale, **§15.8**
> = chiusura più recente, con le due nuove figure sui cicli e la scoperta MLP2-cancella-la-separazione).
> Restavano aperti per l'8 (nessun cambiamento, non richiesti per il 9 a meno di dirlo esplicitamente):
> Priorità 2/3 del filone stagewise, il controllo chord dell'edge-ablation, l'estensione di ogni sezione
> a ER/n64/n20, il plot `cosine_raw_examples` mai lanciato su dati reali, i 3 seed mancanti per le curve
> aggregate dei cicli.
>
> **Regole operative valide dal Report 6 in poi restano tutte in vigore** (caption 21, 55–60, niente
> "mixed" random, niente commit/push/scancel/sbatch da Claude — solo `git pull`). Dal Report 7 (sessione
> 2026-07-17) il report **non è più mostrato direttamente alla prof**: è materiale di lavoro da cui
> l'utente estrae le slide a mano — questo NON allenta nessuna regola di scrittura (vedi memoria
> `project_report7_audience_change`). Dettaglio storico completo di Report 6/7/8 (thread A/B/C, variante
> similarity, ricompilazioni, cambio destinazione) è in **§13/§14/§15**, non ripetuto qui.

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
- **⚠️ REGOLA NON NEGOZIABILE — il progetto ha SEMPRE accesso completo all'HPC Bocconi,
  training incluso.** Il sandbox in cui gira Claude non ha accesso di rete diretto (niente
  `ssh`/`scp` interattivi verso HPC — è un limite tecnico di QUESTA sessione, non del
  progetto), ma il pattern è identico a quello di git: **Claude prepara lo script e i comandi
  `sbatch`, l'utente li lancia su HPC.** Questo vale per QUALSIASI training, non solo per eval.
  **Non scrivere MAI** frasi tipo "non abbiamo accesso al training", "risorse che Claude non
  ha", "fuori portata da questo sandbox" riferite a HPC/training — sono FALSE e non vanno
  scritte, nei report né altrove. Se un esperimento richiede un retrain: prepara lo script/il
  comando `sbatch` e consegnalo, esattamente come per i comandi git; non presentarlo come un
  limite strutturale. Il solo vincolo reale è che Claude non può eseguire `ssh`/`scp` da sé in
  questa sessione (quindi il download dei `.pt` per l'eval locale lo fa l'utente, vedi §14.5) —
  questo non ha nulla a che vedere con l'accesso a HPC per il training, che è sempre disponibile
  tramite l'utente.
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

**Cambio di destinazione del Report 7 (dal 2026-07-17, 6a sessione, vedi §14.13):** l'utente
costruirà a mano un PowerPoint per la prof partendo dal report — il report NON va più mostrato
direttamente alla prof, diventa un documento di lavoro/riferimento per l'utente. Le regole di
scrittura sotto (20/21/55–60) restano valide finché non detto altrimenti: sono comunque lo
standard da cui l'utente estrarrà le slide.

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
61. **⚠️ REGOLA NON NEGOZIABILE (feedback prof, dopo aver visto il PPT del Report 7, §14.14) — OGNI grafico che
    aggrega su più seed deve mostrare error bars / la dispersione fra seed, MAI solo la media nuda.** La Figura 1
    del Report 7 (`r7_sweep_and_logit.png` e tutte le sue gemelle — ER/n64/n20/similarity) disegnava solo la
    curva media sui 4 seed: mancava un'informazione importante (quanto i seed concordano). Fix fatto:
    `plot_mechanistic_asym_chains.py::fig_sweep_and_logit` ora disegna error bars (std sui seed) su ogni curva in
    entrambi i pannelli (i dati per-seed erano già in `metrics.csv`/`readout.csv`, semplicemente non venivano
    usati per la dispersione); le 8 figure `r7_sweep_and_logit*.png` rigenerate in locale, nessun nuovo dato
    servito. REGOLA GENERALE (vale per OGNI report, non solo il 7): error bars (std sui seed) di default su ogni
    curva aggregata; se il fenomeno è bimodale/seed-lottery, preferire linee/punti per-seed distinti (regola 4/20,
    non si escludono — scegliere quella più leggibile) ma MAI una media senza indicare la dispersione.
62. **⚠️ REGOLA NON NEGOZIABILE (feedback prof, §14.14) — quando uno sweep discreto trova qualcosa di interessante
    (picco, salto, transizione) in un punto, valutare SEMPRE anche i punti immediatamente vicini, non saltare al
    prossimo valore già pianificato.** La Figura 3 del Report 7 (`r7_attention_leak.png`, §sec:res-attention) aveva
    uno spike riproducibile ad `a=10`, ma il prossimo split valutato era `a=14` — un buco di 3 split (`11,12,13`)
    proprio dove serviva vedere se lo spike scende subito o resta alto. Fix: il default di `attn_splits` in
    `mechanistic_asym_chains.py` ora include `11,12,13` per ogni run futuro; aggiunta anche una logica di MERGE con
    l'`attn_cache.npz` esistente (non sovrascrive più gli split già calcolati — costosi, un backward per
    nodo-query — quando se ne aggiungono altri). REGOLA GENERALE: un valore isolato "interessante" in uno sweep
    discreto non è mai sufficiente da solo — verificarne la forma locale con almeno 2-3 punti vicini prima di
    scriverlo nel report come una scoperta robusta. Dettagli/stato del fix in §14.14.
63. **Violata la regola 56 (niente nomi di file/funzioni nel testo renderizzato) DUE VOLTE nella stessa
    sessione (Report 8), nonostante fosse già nota e già corretta la prima volta.** È successo scrivendo
    `\texttt{generate_split_cycles_graph}` e poi, in una sezione diversa dello stesso report,
    `\texttt{stagewise_diagnostics.py}`. Sapere la regola non basta: dopo OGNI blocco di prosa nuovo in un
    report, prima di considerarlo finito fare subito
    `grep -n "texttt\|\.py\b\|\.sbatch" file.tex | grep -v "^[0-9]*:%"` (o equivalente) — non aspettare la
    verifica finale di fine sessione, il rischio è scriverne un'altra nel frattempo.
64. **Preparato uno sbatch (`r8_two_cycles.sbatch`) con `--time=05:00:00` ricalcando il default di altri
    script del progetto, SENZA controllare prima i tempi storici REALI di quella stessa identica
    computazione.** `mechanistic_asym_chains.py` con `contrib_n_graphs=64` aveva già impiegato fino a
    8h23m in un job precedente (`r7simn40`, con 3/8 task in TIMEOUT esattamente a quel limite di 5h,
    documentato in §14.7/§14.10) — l'ho scoperto solo perché l'utente ha chiesto "sei sicuro?" e mi ha
    fatto controllare `sacct`. REGOLA: quando un nuovo sbatch rilancia una computazione già documentata
    altrove nel progetto (stesso script/stessa quantità), cercare PRIMA il tempo storico reale (grep in
    istruzioni.md, o `sacct` se disponibile) e calibrare `--time` su quello — non riusare il default di
    un altro sbatch per inerzia.
65. **Due bug di layout matplotlib, validi in generale per ogni script di plotting di questo progetto.**
    (a) Un `\` seguito da spazio FUORI da un blocco `$...$` (es. `"vs.\\ stage"` in una stringa Python)
    viene renderizzato letteralmente da matplotlib invece di sparire come farebbe in LaTeX vero — nei
    titoli/label matplotlib usare `"vs. stage"` semplice, il backslash-spazio ha senso solo dentro
    `$...$`. (b) `fig.colorbar(im, ax=lista_di_assi, ...)` insieme a `fig.tight_layout()` e un `suptitle`
    su più righe produce sovrapposizioni (titoli dei pannelli coperti dal suptitle, colorbar sopra
    l'ultimo pannello) — matplotlib avvisa ("Axes not compatible with tight_layout") ma **il warning da
    solo non basta**, va guardata la figura. Fix: `fig.tight_layout(rect=[0,0,W,H])` per riservare lo
    spazio, POI un colorbar-axis dedicato via `fig.add_axes([...])` — mai lasciare che `colorbar` rubi
    spazio da una lista di assi dopo un `tight_layout` con `rect`. REGOLA GENERALE (già nello spirito
    della skill `verify`, qui applicata ai plot): dopo aver scritto un plot script, SEMPRE generare e
    **guardare** la figura (via `Read` sull'immagine) prima di considerarlo finito, non fermarsi
    all'assenza di errori/eccezioni Python.
66. **Leggere un grafico invece dei numeri grezzi ha quasi portato a riportare nel report una conclusione
    SBAGLIATA (invertita) per il test a due cicli del Report 8.** Nella figura del comportamento, ho letto
    "exact match ≈1.0 ovunque" perché la curva nera (exact) e quella magenta (cut) erano ENTRAMBE
    esattamente a y=0 e si sovrapponevano perfettamente (una disegnata sopra l'altra, stesso valore),
    nascondendo che exact fosse a 0 e non a 1. Solo incrociando con il CSV grezzo ho scoperto che il
    modello collassava a "tutto connesso" — l'opposto di quanto stavo per scrivere. REGOLA: quando una
    lettura visiva di un grafico sembra "troppo pulita" o sorprendentemente positiva (specialmente se
    contraddice l'ipotesi attesa in modo favorevole), verificare SEMPRE con i numeri grezzi (CSV/json)
    prima di scriverla nel report — più linee possono sovrapporsi esattamente allo stesso valore e
    nascondersi a vicenda.
67. **Comprimere una tabella copiata/adattata da un report precedente ha fatto sparire una colonna
    necessaria a capirla.** La tabella del falsification test nel Report 8, adattata da quella del
    Report 7, teneva solo la colonna `small` (dimensione della componente piccola) e aveva tolto
    `|L1|`/`|L2|` (le dimensioni delle due componenti grandi) — resa incomprensibile da sola (l'utente
    l'ha segnalato, giustamente infastidito: "che cazzo fai che metti solo quella dello small"). REGOLA:
    quando si copia/adatta una tabella da un report precedente, NON abbreviare le colonne che servono a
    capire il contesto sperimentale (dimensioni, split, condizioni) solo perché sembrano ridondanti — la
    leggibilità stand-alone della tabella conta più della sua compattezza.
68. **La partizione `gpunew` NUDA (senza prefisso) è SPARITA dal cluster (scoperto 2026-07-25, dopo la
    manutenzione RHEL 9.8/Slurm di cui sotto).** `sinfo` non la elenca più: restano solo
    `long_gpunew`/`medium_gpunew`/`short_gpunew`/`debug_gpunew` (i tier), mentre `gpuh200` **nuda esiste
    ancora** (cap 1 giorno, 6 nodi — più capienza del vecchio pool `gpunew`). Qualsiasi sbatch scritto
    prima di questa data con `--partition=gpunew` per un training >6h10 (che quindi non entra nei tier
    `medium_*`/`short_*`) fallisce con `sbatch: error: ... User's group not permitted to use this
    partition` — capitato ai due training $n{=}46$ del Report 9 (§16.6/§16.7). **FIX: usa `gpuh200`** al
    posto di `gpunew` nudo per ogni training che serva un cap >6h10 e ≤1 giorno (verifica comunque con
    `sinfo -o "%P %G %l %D"` prima di fidarti di un `--partition=` scritto in una sessione precedente,
    la lista dei nomi può cambiare dopo una manutenzione). Dettaglio completo (incluso il nuovo pool
    temporaneo `gpua100`, la finestra di manutenzione full-cluster e la nuova policy `/scratch`) in §6.

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

> **🟨 AGGIORNAMENTO 2026-07-25 — la `gpunew` NUDA è sparita; manutenzione full-cluster lunedì
> 27/7; nuova policy `/scratch`; nuovo pool temporaneo `gpua100`.** Scoperto lanciando i due
> training $n{=}46$ del Report 9 (§16.6/§16.7): `sinfo -o "%P %G %l %D"` in questa sessione **non
> elenca più una partizione `gpunew` nuda** — restano solo i tier `long_gpunew` (3gg, 2 nodi),
> `medium_gpunew` (6h10, 4 nodi), `short_gpunew` (1h10, 4 nodi), `debug_gpunew` (15min, 4 nodi).
> **`gpuh200` nuda invece esiste ancora** (cap 1 giorno, 6 nodi) insieme ai suoi tier. Un
> `--partition=gpunew` per un training >6h10 ora fallisce subito con `sbatch: error: ... User's
> group not permitted to use this partition` (errore 68) — **usa `gpuh200`** al posto di
> `gpunew` per qualunque training che serva più dei 6h10 di `medium_gpunew` ma stia dentro 1
> giorno (i training n40/n46 di questo progetto, ~6–14h, rientrano). Verificato che il resto
> della mappa partizioni sotto (H100 vs H200, cap dei tier) è altrimenti invariato.
>
> **Nuovo pool TEMPORANEO `gpua100`** (mail HPC, nodi NVIDIA A100 `gnode[03-04]`, riassegnati
> dalla partizione studenti per l'estate, **tornano a settembre 2026** — non farci affidamento
> per training che dovrebbero girare oltre quella data): `gpua100` (1 giorno), `medium_gpua100`
> (6h10), `short_gpua100` (1h10), `debug_gpua100` (15min). Utile come pool alternativo se
> `gpuh200`/`gpunew` sono congestionati, solo per lavoro di questa estate.
>
> **⚠️ Manutenzione FULL-CLUSTER lunedì 27 luglio 2026, 09:00–18:00 CEST (tentativo).** Login e
> compute node, filesystem BeeGFS/NFS (`/home`,`/data`,`/scratch`) e Slurm stesso non saranno
> disponibili. Ogni job che si sovrapporrebbe alla finestra resta PD con motivo
> `ReqNodeNotAvail, Reserved for maintenance` (i nodi coinvolti appaiono in stato `maint` su
> `sinfo`) e va **ri-sottomesso dopo**; un job **in esecuzione** che si sovrappone viene
> **cancellato** (non ripreso) e va ri-sottomesso. **Prima di lanciare un training lungo (12–14h)
> vicino a quella data, calcolare se finirebbe prima delle 09:00 di lunedì o dopo le 18:00** —
> altrimenti aspettarsi la cancellazione e pianificare il rilancio post-manutenzione.
>
> **Nuova policy di pulizia `/scratch` (dal 5 agosto 2026): retention ridotta da 90 a 30
> giorni.** Il 5/8 vengono cancellati i file non toccati dal 6/7/2026 in poi; da allora, pulizia
> automatica giornaliera dei file più vecchi di 30 giorni. Questo progetto vive sotto `~/` (home,
> non `/scratch`), quindi non dovrebbe essere toccato direttamente — ma se una chat futura sposta
> output pesanti in `/scratch` per qualunque motivo, tenerne conto (nessuna azione necessaria ora).

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

**Report 1–7 CONGELATI (consegnati/chiusi).** Il Report 8 è **sostanzialmente completo** (vedi banner
in cima al file e §15, specialmente §15.6): tutte le sezioni sono scritte con dati reali, il `.tex`
compila pulito. **Non c'è un "compito corrente" aperto in questo momento** — la fase attiva ora è
l'utente che costruisce a mano un PowerPoint per la prof a partire dal Report 8; eventuali richieste di
modifica al report arriveranno probabilmente da una nuova chat Claude (che farà l'onboarding leggendo
questo file + tutti i `.tex`, come da regola di lettura in cima).

Se in futuro riparte lavoro attivo su un nuovo report (Report 9+), aggiornare questa sezione con lo
stesso schema usato per i report precedenti (tema, stato per thread/sezione, cosa resta aperto),
lasciando lo storico di Report 6/7/8 nelle rispettive sezioni (§13 Report 6, §14 Report 7, §15
Report 8) invece di sovrascriverlo qui.

**Note git (la modalità, valida per tutti i report):** Claude NON committa/pusha (né lancia
sbatch/scancel), **PUÒ** `git pull`. La scrittura `.tex`/figure/json si committa in locale sul Mac,
i risultati HPC li committa l'utente lato HPC. Verificare sempre con `git status` prima.

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
- ~~**Thread D — Alberi** (esplorativo)~~ **RIMOSSO dal report** (richiesta utente, 18a sess.): mai
  implementato, sottosezione `.tex` cancellata. Resta un'idea futura solo se l'utente lo richiede esplicito.
- **Verdetto** (SCRITTO, 18a sess.): per ciascuna tesi se l'evidenza la sostiene (1 sì+meccanismo, 2 sì
  modesto, 3 sì, 4 solo debole) + tie a Report V + limiti onesti. Vedi §sec:verdict nel `.tex`.

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
- **C.2 catena di PONTI SPESSI (NUOVO, 18a sess.):** il "ponte spesso a 2 blocchi" da solo è banale (cross≈1.0 a
  ogni size, ogni width → tab:c2 clique rows, è solo il baseline within-capacity, NON un esperimento). Il test
  vero è **mettere i ponti spessi in CATENA** (come C.1, K=2..5): un ponte più spesso (w=2,3 archi paralleli =
  parallelismo Thread A dentro capacità) **AIUTA la composizione, parzialmente**. n40 c=3 exact a K=3: 0.38→0.54→
  0.76 (w1→2→3); K=5: 0.00→0.06→0.25; n20 c=3 K=3 arriva a 1.00 con w3. MA: (a) rescue parziale (le catene
  degradano comunque con K), (b) a c piccolo un ponte molto spesso è esso stesso OOD → intacca il K=2 allenato
  (c=3 w3: exact 1.00→0.51). Eval `eval_clique_chain.py --bridge_width 2/3`, output `clique_chain_bw{2,3}/`.
  **⚠️ L'effetto è BIMODALE/seed-lottery (la media inganna, l'utente l'ha notato): figura e tabella sono PER-SEED**
  (`thick_chains_n40_perseed.png` via `plot_thick_chains.py` riscritto a 4 pannelli per-seed; tab:c2chain per-seed,
  grassetto se ≥0.5). Per-seed: seed2000 compone lontano e i ponti spessi lo portano a K=5; seed1000/4000 guadagnano
  solo K=3 e solo coi ponti spessi; seed3000 non compone mai. (La vecchia `thick_chains_n40.png` a media è orfana.)
- **C.2 blocchi ER (thick_bridges blocks bw1):** **transfer PARZIALE** — cross decade col size a floor ~0.62
  (n40)/~0.76 (n20), disc→chance; a n40 c=20 il WITHIN-block crolla a ~0.55 in 3/4 seed (il blocco ER da 20
  nodi è esso stesso OOD per un modello allenato solo su cricche) → la domanda-ponte è moot. L'asimmetria 0.36
  è SINTOMO del within rumoroso, NON una firma sequenziale (errore 60). "Unseen, non too hard" → fixabile coi
  dati (Report V §5.6).
**File nuovi/toccati (per commit utente):** `report/6/transformer_for_graphs_6.{tex,pdf}` (C.1+C.2 scritte),
`plot_thick_bridges.py` + `plot_thick_chains.py` (NUOVI, locale no-GPU), `plot_clique_chain.py` (usato),
`eval_multipath.py` (aggiunti campi `conn_pairwise`/`disc_pairwise`, vedi revisione 18a sess. sotto),
figure `runs/report6/report6_figs/{clique_chain,clique_chain_er}_{composition,length}_n{20,40}.png`,
`thick_bridges_n{20,40}.png`, `thick_chains_n40.png`, **json nuovi** `runs/report6/clique_chain_bw{2,3}/n{20,40}_*`
(16, ponti spessi in catena — generati IN LOCALE via scp dei `.pt` in /tmp poi cancellati), `istruzioni.md`.
Figure nel `.tex`: `clique_chain_composition_n40.png` (C.1) + `thick_chains_n40.png` (C.2). Tabelle: tab:c1chain
(n40 c=3 K-decay), tab:c1chainer (ER chain), tab:c2chain (ponti spessi in catena), tab:c2 (full-size baseline +
ER blocks).
**REVISIONE 18a sessione (feedback utente sui Thread A/B/C, tutto fatto):** (1) **A.1 "where the rescue breaks
down" (tab:a1n64)**: la spiegazione "mette tutto disconnesso" era SBAGLIATA. Re-eval n64 path_union (scp `.pt` in
/tmp, `eval_multipath.py` esteso con accuracy su coppie connesse/disconnesse): a k=2 il modello **over-connette**
(disc-acc 0.53→0.00, a k=4 dichiara tutto un'unica componente → pair=1.0 banale), 3/4 seed reach≈0.9 ma non
chiudono (s,t); seed4000 fa l'opposto (collassa a disconnesso). Tabella augmentata con conn/disc-pair acc. (2)
**B (n64)**: aggiunta tab:basplit_n64 per-seed (esempio concreto dello "spread" — a a=1 i seed vanno 0.12→1.00 vs
≈1.00 tight a n40). (3) **C.2**: rimosso "thicker bridge changes nothing" (a 2 blocchi è banale), sostituito con
la **catena di ponti spessi** (sopra). Tutto ricompila (18 pp, 0 ref indefinite, 0 overfull). **Regola `.pt` in
/tmp rispettata**: scp → uso → `rm -rf` (mai nel repo).
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

## 14. Report 7 — aprire il trunk: perché lo split asimmetrico funziona (meccanicistico)

> Blocco aggiunto dopo la consegna del Report 6. **Resta valido per OGNI nuova chat sul Report 7.**

### 14.0 Le richieste originali dell'utente (trascritte VERBATIM — leggere prima di tutto)

L'utente ha chiesto esplicitamente che queste richieste restino trascritte parola per parola in
questo file, perché una chat futura deve poterle rileggere per intero e capire da sole tutto il
contesto (non un riassunto). Sono nell'ordine in cui sono arrivate in questa sessione (3 sessioni,
2026-07-07/08).

**Richiesta 1 (apertura della sessione, dopo aver letto istruzioni.md e i Report 1–6):**
```
il report 6 è completato. ora leggi attentamente il suo file .tex riga per riga, leggi anche
quello di 5,4,3,2,1. leggi riga per riga, poi crea un report 7. in questo report 7 ciò che
faremo è focalizzarci sull'esperimento di due chain di lunghezza diversa e capire perchè 4 36
lo capisce, mentre 20 20 non lo capisce. cioè capiremo proprio a livello di architettura e di
attention score e di pesi cosa sta imparando
```

**Richiesta 2 (dopo un primo tentativo di piano, rifiutato con "let's chat about this" — l'utente
ha incollato un'intera conversazione avuta con un'altra chat Claude, chiedendo di seguire QUELLA
analisi invece della mia prima bozza):**
```
intanto vorrei che creassi velocemente il report 7 .tex giusto per averlo.
poi, io ho fatto qudsta chat con claude esterno e avevo ddeciso questi esperimenti. dimmi se ci sta fare questi che mi sembrano meglio dei tuoi, dimmi se c'è da aggiungere.
incollo:
 voglio che implementi (non solo per 4/36 ma anche per tutti gli altri tipo
  3/37 ecc) queste cose qui di cui abbiamo parlato in questa risposta di
  claude.
Sì, con l'architettura che hai scritto cambia parecchio l'interpretazione. La cosa più importante è questa:
nei Report 5/6, se usate il linear read-out, il logit z_{ij} non viene calcolato confrontando direttamente h_i e h_j.
Viene calcolato usando solo l'embedding finale del nodo i e un classificatore specifico per la colonna j.
Quindi il modello non fa necessariamente z_{ij}=f(h_i,h_j), ma, nel readout lineare, fa più precisamente:
Z_i = W_out h_i + b_out, oppure z_{ij} = w_j^T h_i + b_j.
Dove h_i è l'embedding finale del nodo i, w_j è il vettore di readout associato alla colonna/nodo
target j, z_{ij} è il logit "il nodo i è connesso al nodo j".
Questa cosa è centrale. Nel linear readout, il nodo j entra nel logit attraverso il vettore imparato
w_j, non attraverso l'embedding finale h_j. Quindi h_i deve contenere, da solo, abbastanza
informazione per rispondere alla domanda "a quali nodi j sono connesso?". Il readout è come una
batteria di n classificatori w_1,...,w_n. Ogni w_j chiede all'embedding h_i: "contieni evidenza
che sei connesso al nodo j?" Questo rende molto sensato studiare W_out, perché nel vostro modello
non è un dettaglio finale: è proprio il modo in cui la matrice di connettività viene letta dagli
embedding.

1. Implicazione importante: il modello non è permutation-equivariant nel senso naive.
Dato che l'input è la riga della matrice di adiacenza A'_i e poi fate h_i^(0) = W_in A'_i + b_in,
il modello riceve le colonne della matrice come coordinate fisse. Cioè il nodo 7, il nodo 19, il
nodo 32 non sono anonimi: hanno coordinate distinte nell'input. In pratica, W_in contiene un
embedding imparato per ciascun possibile vicino-label. Se A'_{ik}=1, allora nel read-in entra il
contributo associato alla coordinata k. Per una chain ordinata, il nodo i ha riga circa
A'_i = e_{i-1}+e_i+e_{i+1}. Quindi il read-in diventa circa h_i^(0) ≈ E_{i-1}+E_i+E_{i+1}+b.
Questo vuol dire che il modello può imparare scorciatoie legate agli indici dei nodi, non solo
alla struttura astratta del grafo. Questo è importantissimo per il fenomeno 4/36 vs 20/20. Se le
chain sono sempre costruite con nodi contigui, tipo {0,1,2,3} e {4,...,39}, allora il modello
potrebbe imparare qualcosa del tipo "i nodi da 4 a 39 appartengono spesso a una grande componente
path-like, quindi completali come connessi." Questo non sarebbe vera connettività algoritmica.
Sarebbe una regola label-dependent. Quindi uno degli esperimenti più importanti è: random
relabeling test.

2. Prima cosa da fare: sweep sulle chain sbilanciate, ma con controlli giusti.
Tu vuoi studiare chain non equilibrate con varie lunghezze della piccola chain. Quindi farei
a + (40-a), a=1,2,...,20. Per ogni a, costruisci due path disconnessi P_a ⊔ P_{40-a}. Per ogni
split misura separatamente: (1) accuracy dentro la chain piccola; (2) accuracy dentro la chain
grande; (3) accuracy across-cut; (4) exact-match totale; (5) accuracy condizionata dalla distanza
nella chain grande; (6) accuracy condizionata dalla distanza nella chain piccola; (7) positive-rate
predetto, cioè quante coppie il modello dichiara connesse. Questo già ti dice se il modello sta
facendo: "tutti connessi"; "tutti disconnessi oltre distanza 9"; "completo la componente grande";
"taglio correttamente tra componenti"; "faccio una vera propagazione lungo la path". Però questo
va fatto in almeno tre condizioni: Condizione A: fixed labeling, piccola chain sempre sui primi a
nodi, grande chain sui restanti. Condizione B: random relabeling, stesso grafo astratto, ma permuti
casualmente le etichette dei nodi — se il modello risolve 4+36 solo nel fixed labeling e crolla nel
random relabeling, allora sta usando gli indici. Condizione C: shifted small chain, la piccola
chain non sta sempre all'inizio, per esempio per 4+36: {0,1,2,3} poi {18,19,20,21} poi
{36,37,38,39} — se cambia molto, allora W_in e W_out stanno usando posizione/label. Questo
esperimento per me viene prima dell'analisi sofisticata dei pesi.

3. Logit analysis, ma contestualizzata al vostro readout. Nel vostro linear readout,
z_{ij}=w_j^T h_i+b_j. Quindi quando guardi z_{i,30}, stai guardando quanto l'embedding del nodo i
attiva il classificatore target 30. Se z_{i,30} è alto, non significa necessariamente che h_i e
h_{30} sono simili. Significa che h_i contiene una direzione che il readout w_{30} interpreta come
"connesso al nodo 30". Questo vuol dire che devi analizzare non solo i logits, ma anche h_i^T w_j.
In particolare, per 4+36, vogliamo sapere: i nodi della componente lunga attivano positivamente
tutti i w_j corrispondenti ai nodi della componente lunga? Cioè, se la componente lunga è
{4,...,39}, vogliamo vedere se w_j^T h_i > 0 per tutti i,j in {4,...,39}. Se sì, il modello ha
costruito un embedding h_i che dice "sono nella grande componente contenente questi target-labels."
Nel 20+20, invece, probabilmente vedrai che w_j^T h_i diventa negativo quando j è troppo lontano da
i lungo la path. Quindi il plot logit utile non è solo "logit vs distanza", ma z_{ij}=h_i^T w_j+b_j
separato per: (i,j) nella piccola chain; (i,j) nella grande chain; (i,j) across-cut; distanza
d(i,j); label j.

4. Studiare W_out: sì, ha molto senso. Nel vostro modello, W_out è molto interessante. Se
readout = nn.Linear(d_model, n), allora W_out ∈ R^{n×512}. Ogni riga w_j è il classificatore per
il target-node j. Cose da plottare: A. Heatmap grezza di W_out (n×512, per n=40 diventa 40×512,
può mostrare pattern grossi ma spesso i canali interni sono arbitrari). B. Norme dei vettori
||w_j|| per j=1,...,40 — se certi nodi-label hanno norme molto più alte, il modello sta dando più
importanza ad alcuni target; per esempio negli split 4+36, potresti scoprire che i target nella
componente grande hanno vettori w_j più facilmente attivabili. C. Cosine similarity tra righe di
W_out: C^out_{jk}=cos(w_j,w_k), matrice 40×40 — se W_out ha imparato una struttura path-like o
component-like, potresti vedere bande diagonali o blocchi: se w_j e w_{j+1} sono simili, emerge
una banda diagonale; se tutti i w_j della grande componente sono simili, emerge un blocco; se non
c'è struttura, sembra rumore. D. PCA di w_j: prendi i 40 vettori w_j∈R^512, fai PCA in 2D, colori
per indice j — se il modello ha imparato una rappresentazione ordinata dei nodi, i punti
potrebbero disporsi lungo una curva secondo l'indice. Questo sarebbe molto interessante per capire
se W_out contiene una specie di geometria dei node-labels.

5. Studiare W_in: ancora più importante per capire shortcut di label. Il read-in fa
h_i^(0) = A'_i W_in + b. A seconda della convenzione, puoi pensare a W_in come una matrice che
contiene un vettore e_k per ogni input-coordinate k, cioè per ogni possibile vicino-label. Quindi
se il nodo i è collegato a k, l'embedding iniziale riceve il contributo e_k. Cose da plottare:
A. Heatmap di W_in (40×512 o 512×40). B. Norme degli input embeddings ||e_k|| per ogni node-label
k — se alcuni label hanno embedding più forti, il modello sta usando certi nodi come segnali
speciali. C. Cosine similarity tra e_k: C^in_{kl}=cos(e_k,e_l) — se emerge una struttura ordinata
lungo la chain, allora il modello ha imparato una geometria degli indici. D. Allineamento
W_in–W_out: questo è uno dei plot più interessanti. Calcola M_{kj}=e_k^T w_j dove e_k è il vettore
associato alla coordinata input k e w_j è il vettore readout per target j. La matrice M∈R^{40×40}
ti dice: se il nodo i vede k come vicino nell'input, quanto questo contribuisce direttamente ad
attivare il target j? Se M ha una diagonale forte, il modello conserva identità dei nodi. Se M ha
una banda, il modello ha imparato vicinanza lungo gli indici. Se M ha blocchi, il modello ha
imparato componenti/regioni. Questo è molto più interpretabile della heatmap grezza 512×512.

6. Studiare W_Q,W_K,W_V,W_O: sì, ma non solo come heatmap grezza. Capisco perfettamente l'idea:
vogliamo vedere W_Q,W_K,W_V. Però attenzione: una heatmap 512×512 dei pesi grezzi può essere poco
leggibile, perché le coordinate interne del modello non hanno un significato umano diretto. Due
modelli funzionalmente simili possono avere pesi ruotati in modo diverso. Quindi farei entrambe le
cose: 1. heatmap grezza dei pesi; 2. analisi "effective" nello spazio dei nodi. Per ogni layer hai
Q=HW_Q, K=HW_K, V=HW_V. Con n=40, questi sono Q,K,V∈R^{40×512}. Quindi per un grafo specifico
(4+36 o 20+20), puoi vedere q_i (cosa cerca ogni nodo), k_j (che chiave offre ogni nodo), v_j (che
contenuto offre ogni nodo). Però io non guarderei solo le heatmap 40×512. Guarderei anche:
A. Norme di q_i,k_i,v_i per nodo, plot lungo la chain — se gli endpoint o i nodi della piccola
chain hanno norme diverse, il modello li tratta come speciali. B. Cosine similarity tra query
cos(q_i,q_j) — dice se nodi lontani nella stessa componente fanno "domande" simili. C. Cosine
similarity tra key cos(k_i,k_j) — dice se certi nodi offrono chiavi simili. D. Cosine similarity
tra value cos(v_i,v_j) — dice se i contenuti trasportati dai nodi sono simili. E. Attention score
grezzo: obbligatorio, s_{ij}=q_i^T k_j/√512, heatmap 40×40, per layer 0 e layer 1. F. Attention
effettiva normalized-ReLU: nel vostro caso alpha_{ij} = (1/n)ReLU(s_{ij}). Questa non è softmax.
Quindi non somma a 1 per riga. Perciò devi anche plottare sum_j alpha_{ij} — questa quantità dice
quanta massa totale di attenzione prende il nodo i. È possibile che nel 4+36 non ci sia solo "dove
guarda", ma anche "quanto forte guarda".

7. La heatmap più importante non è solo S=QK^T, ma il contributo al messaggio. L'attention score
ti dice dove il nodo guarda. Però l'output attention è message_i=sum_j alpha_{ij}v_j. Quindi un
nodo j può avere attention alta ma value poco rilevante, o viceversa. Perciò devi plottare anche
una matrice di contributo: C_{ij} = alpha_{ij}||v_j||. Ancora meglio: C^out_{ij}=||alpha_{ij}v_j W_O||.
Questa dice: quanto il nodo j contribuisce davvero all'aggiornamento del nodo i? Questa è più
causale della sola attention. Per il fenomeno 4+36, la domanda diventa: i nodi lontani della
grande chain comunicano davvero tra loro, oppure il modello completa la componente grande senza
messaggi long-range? Se nella heatmap alpha vedi solo attenzione locale, ma il modello predice
tutta la lunga chain connessa, allora probabilmente non sta propagando davvero lungo 36. Sta usando
una scorciatoia nei residui/MLP/readout. Se invece vedi attenzione molto forte dentro tutta la
grande componente, allora potrebbe esserci una forma di global completion.

8. Esperimento fondamentale: fixed-label vs random-label. Questo secondo me è il test più
importante per i Report 5/6 sulle chain sbilanciate. Fai lo stesso split (a+(40-a)), ma in due
modi. Fixed-label: P_a={0,...,a-1}, P_{40-a}={a,...,39}. Random-label: costruisci lo stesso grafo
astratto, poi applichi una permutazione casuale pi ai nodi. Se il modello è algoritmico, dovrebbe
comportarsi uguale. Se il modello usa label shortcuts, crolla. Per ogni a, misura
Delta(a) = acc_fixed(a) - acc_random(a). Se Delta(a) è enorme proprio per 4+36, hai una prova forte
che il successo 4+36 dipende dalle etichette/posizioni. Questo è molto più convincente di guardare
solo i logits.

9. Esperimento: spostare la chain piccola. Altro test pulito. Per a=4, invece di mettere la
piccola chain sempre su {0,1,2,3}, mettila in posizioni diverse: {0,1,2,3}, {10,11,12,13},
{18,19,20,21}, {36,37,38,39}. La grande chain occupa gli altri nodi, collegati come path. Se la
performance cambia a seconda di dove sta la piccola chain, allora il modello non sta solo usando
"dimensione componente"; sta usando anche node labels specifici.

10. Esperimento: same lengths, different ordering. Per 4+36, puoi costruire la grande chain in
ordine naturale: 4-5-6-...-39, oppure con un ordine permutato: 4-17-9-31-.... Stessa componente,
stessa dimensione, stessa struttura astratta path, ma gli archi non seguono più l'ordine degli
indici. Se il modello risolve solo la chain ordinata, allora ha imparato una scorciatoia di
indice/path ordinato. Questo è molto importante perché il read-in vede coordinate fisse. Il
modello può imparare che "vicino nell'indice" spesso significa "vicino nel path".

11. Esperimento: component-size probe sugli embedding. Per ogni grafo a+(40-a), salva gli
embedding finali h_i^(2). Poi addestra un probe lineare, a modello congelato, per predire dal
singolo embedding h_i^(2): 1. dimensione della componente di i; 2. se i sta nella componente
grande; 3. distanza dall'endpoint più vicino; 4. posizione normalizzata nella chain; 5. degree del
nodo; 6. eccentricità nella componente. Se dal solo h_i puoi predire "sono nella componente grande"
molto bene, allora il modello sta codificando informazione globale di componente. Questo si
collega direttamente al linear readout: dato che z_{ij} usa solo h_i, tutto ciò che serve per
predire la riga i della connettività deve stare dentro h_i.

12. Esperimento: readout decomposition. Dato che z_{ij}=h_i^T w_j+b_j, puoi decomporre il logit
sulle coordinate: z_{ij}=sum_{r=1}^{512} h_{i,r}w_{j,r}+b_j. Per coppie interessanti, tipo 4+36
due nodi a distanza 35 nella grande chain predetti connessi, e 20+20 due nodi a distanza 19 nella
stessa chain predetti disconnessi, puoi guardare quali dimensioni r contribuiscono di più. Per
ogni coppia (i,j), calcola c_r = h_{i,r}w_{j,r}. Poi guarda top positive e top negative
coordinates. Domanda: le stesse coordinate spiegano il successo su 4+36 e il fallimento su 20+20?
Se sì, c'è un circuito comune. Se no, il modello cambia regime.

13. Esperimento: residual stream patching. Questo è più avanzato, ma molto forte. Prendi due
input: grafo A: 4+36, dove il modello funziona; grafo B: 20+20, dove il modello fallisce. Salva
gli stati H^(0),H^(1),H^(2). Poi fai interventi tipo: durante il forward su 20+20, sostituisco
l'embedding di alcuni nodi/layer con quello corrispondente dal 4+36. Per esempio: patch dopo
read-in; patch dopo layer 1; patch dopo layer 2; patch solo componente grande; patch solo
endpoint; patch solo MLP output; patch solo attention output. Se patchando H^(1) cambia molto la
predizione, il circuito rilevante nasce nel primo layer. Se cambia solo patchando H^(2), nasce nel
secondo layer/readout. Questo aiuta a rispondere: dove si forma la rappresentazione "componente
grande tutta connessa"?

14. Ablation: attention vs MLP vs readout. Dato che avete solo 2 layer e 1 head, le ablation sono
gestibili. Farei queste: A. Spegnere attention layer 0: metti attention output a zero nel primo
blocco, h ← LN(h+0). Guarda se 4+36 resta risolto. B. Spegnere attention layer 1: stessa cosa nel
secondo blocco. C. Spegnere MLP layer 0/layer 1: metti output MLP a zero. D. Spegnere solo W_O:
cioè attention calcolata ma output cancellato. E. Usare solo read-in + readout: passi
H^(0) → W_out senza transformer blocks. Se già questo risolve parzialmente 4+36, allora il modello
sfrutta moltissimo il read-in/readout e non una propagazione profonda.

15. Attention mask experiments. Anche se il modello è stato trainato con attention globale, puoi
fare interventi in inferenza. Per una chain, imponi maschere artificiali. A. Local-only attention:
permetti attenzione solo a nodi a distanza grafica ≤1 o ≤2 — se 4+36 resta corretto con attention
local-only, allora forse la connettività lunga non dipende da global attention; se crolla, allora
usava attenzione globale. B. Same-component-only attention: permetti attenzione solo dentro la
stessa componente — questo non è un test realistico, perché gli stai dando informazione della
target structure, ma serve come controllo. C. No-long-range attention: blocca attenzione tra nodi
con distanza maggiore di 9 — se 4+36 resta corretto anche senza attention long-range, allora il
superamento di 9 non viene da messaggi diretti long-range. D. Cut-only ablation: blocca attenzione
tra componenti diverse — se cambia molto, il modello stava usando anche segnali across-cut per
capire la separazione.

16. Per il similarity readout cambia tutto. Quando usate il similarity readout,
z_{ij}=s·cos(h_i,h_j)+b. Qui sì, il logit dipende direttamente da entrambi h_i,h_j. In questo caso
l'analisi più importante diventa cos(h_i,h_j) per distanza e per componente. Per il linear readout,
invece, la domanda è: h_i attiva w_j? Per il similarity readout, la domanda è: h_i e h_j diventano
simili se sono nella stessa componente? Quindi devi tenere separate le due varianti. Non mischiare
interpretazioni. Per linear readout: z_{ij}=h_i^T w_j+b_j. Per similarity readout:
z_{ij}=s cos(h_i,h_j)+b. Sono due meccanismi molto diversi.

17. Cosa implementerei concretamente adesso. Io farei una script unico tipo
mechanistic_audit_unbalanced_chains.py. Input: checkpoint; readout type; n; lista degli split
(a+(n-a)); numero di permutazioni; modalità: fixed, shifted, random-label, random-order. Output:
File 1: metrics.csv, una riga per grafo/split: checkpoint, split, mode, seed, exact, pairwise,
acc_short, acc_long, acc_cut, pred_pos_rate, mean_logit_short, mean_logit_long, mean_logit_cut.
File 2: pair_table.csv, una riga per coppia (i,j): split, mode, seed, i, j, component_i,
component_j, pair_type, distance, logit, prob, pred, target. File 3: cache.pt, per pochi grafi
selezionati: H0, layer0_q, layer0_k, layer0_v, layer0_scores, layer0_alpha, layer0_attn_out,
layer0_mlp_out, H1, layer1_q, layer1_k, layer1_v, layer1_scores, layer1_alpha, layer1_attn_out,
layer1_mlp_out, H2, logits. File 4: weights.pt / weights_summary.csv: salva W_in, W_q,W_k,W_v,W_o
per layer, MLP W1,W2 per layer, LayerNorm gamma,beta, W_out,b_out; e summary: norms, singular
values, cosine matrices, effective kernels.

18. Plot minimi da generare. Per ogni checkpoint e per split 4+36, 8+32, 12+28, 17+23, 20+20:
Logit/accuracy plots: 1. logit medio vs distanza; 2. accuracy vs distanza; 3. istogrammi dei
logits per short/long/cut; 4. positive-rate per split a. Attention plots, per layer 0 e layer 1:
1. heatmap S=QK^T/√d; 2. heatmap alpha=ReLU(S)/n; 3. row mass sum_j alpha_{ij}; 4. mean attention
vs graph distance; 5. contribution heatmap alpha_{ij}||v_j||; 6. contribution output
||alpha_{ij}v_j W_O||. Q/K/V plots, per layer 0 e layer 1: 1. heatmap Q,K,V (40×512);
2. norm ||q_i||,||k_i||,||v_i|| lungo i nodi; 3. cosine similarity tra q_i; 4. cosine similarity
tra k_i; 5. cosine similarity tra v_i; 6. PCA 2D di q_i,k_i,v_i, colorata per componente. Weight
plots: 1. heatmap W_in; 2. heatmap W_out; 3. cosine matrix delle righe di W_out; 4. cosine matrix
degli input embeddings da W_in; 5. matrice di allineamento E_in W_out^T; 6. heatmap grezze
W_Q,W_K,W_V,W_O; 7. singular values di W_Q,W_K,W_V,W_O; 8. effective node-space kernel:
E_in W_Q W_K^T E_in^T — questo kernel dice, già nello spazio dei node-labels, quali input-labels
tendono a produrre query/key compatibili.

19. La domanda scientifica da mettere al centro. Secondo me la domanda giusta non è solo "perché
4+36 funziona e 20+20 no?" ma più precisamente: nel 4+36, il modello sta davvero propagando
informazione lungo una chain di lunghezza 36, oppure sta usando una rappresentazione shortcut
della grande componente? Con questa architettura, le ipotesi principali sono tre. Ipotesi 1: vera
propagazione — il modello costruisce embedding che propagano localmente informazione lungo la
path; dovresti vedere attention prevalentemente locale ma layer dopo layer gli embedding
incorporano informazione più lontana, il comportamento regge a random relabeling, non dipende
dalla posizione della piccola chain, W_in e W_out non mostrano forti scorciatoie di label — questa
ipotesi mi sembra possibile ma non la più probabile. Ipotesi 2: large-component completion — il
modello riconosce che esiste una componente grande e completa tutta quella componente come
connessa; vedresti 4+36 alto, 20+20 basso, embedding della componente grande molto simili o molto
allineati ai target w_j della componente grande, performance sensibile alla dimensione della
piccola chain, forse attention non necessariamente long-range, h_i^T w_j>0 per quasi tutti i,j
nella componente grande — questa per me è l'ipotesi più plausibile. Ipotesi 3: label/order
shortcut — il modello sfrutta il fatto che la chain è ordinata e i nodi hanno label fissi; vedresti
performance alta su fixed-label, crollo su random relabeling, crollo se la chain viene permutata
internamente, pattern forti in W_in, W_out, E_in W_out^T, cosine similarity di W_out ordinata per
indice — questa è molto importante da escludere.

20. Quindi: cosa farei davvero, in ordine. 1. Sweep a+(40-a) con fixed, shifted, random-label,
random-order. 2. Metriche separate short/long/cut/distance. 3. Cache interno per pochi casi
chiave: 4+36, 8+32, 17+23, 20+20. 4. Heatmap attention score S e attention effettiva alpha.
5. Q/K/V node-level analysis: norme, cosine, PCA, heatmap 40×512. 6. Readout analysis: W_out,
cosine w_j, decomposizione h_i^T w_j. 7. Read-in/readout alignment: E_in W_out^T. 8. Ablation
attention/MLP/readout. 9. Solo dopo: evoluzione durante training. Quindi sì: guardare i logits
serve, ma nel vostro caso non basta. La parte più interessante è probabilmente l'interazione fra
W_in, W_Q,W_K,W_V, H^(2), W_out. E soprattutto il fatto che, con linear readout, z_{ij}=h_i^T w_j+b_j,
quindi il modello deve trasformare ogni nodo i in una rappresentazione che contiene quasi tutta
la riga i della matrice di connettività. Questa è la chiave interpretativa per capire cosa succede
sulle chain sbilanciate.
```

**Nota su come è stata gestita la Richiesta 2**: NON è stata implementata alla lettera al 100%
(sarebbe stato uno scope enorme, es. punto 15 "attention mask experiments" e punto 11 "linear
probe sugli embedding" non sono stati fatti). È stata **fusa** con il mio piano iniziale, potata a
un set gestibile (Tier 1/2/3, vedi §14.3/14.3b), con una correzione importante fatta subito
all'utente: il punto 8 (fixed-vs-random-label) **era già coperto** dalla metodologia esistente di
`eval_asym_chains.py` (permuta già ogni grafo a caso), quindi non è stato rifatto da zero ma solo
confermato con un confronto esplicito (Tabella `tab:r7relabel` nel `.tex`). Il punto 9 (shifted
small chain) è STATO ASSORBITO dalla stessa osservazione (la permutazione casuale già copre
implicitamente "dove sta la piccola chain"). I punti 13 (patching) e 14 (ablation) SONO stati
fatti (§14.3b). Il punto 16 (similarity readout) NON è stato fatto (checkpoint da verificare, vedi
§14.6). I punti 17-18 (script/plot unico con tutte le heatmap) sono stati implementati come
`mechanistic_heatmaps.py` + `plot_mechanistic_heatmaps.py` (§14.4).

**Richiesta 3 (dopo che il Tier 1 era già stato consegnato con dati reali su n=40 — l'utente ha
chiesto di completare TUTTO quello descritto nella Richiesta 2, non solo il Tier 1):**
```
vorrei che fai tutti gli altri esperimetni di cui avevamo parlato. rileggi la chat riga per riga
e fai gli esperimetni di cui avevamo parlato. rileggi e completa con tutte le tier.
inoltre, voglio che mostri bene i pesi, le attenzioni, le matrici dei pesi, le matrici delle
attention scores con delle heatmap.
```
Questa è la richiesta che ha prodotto Tier 2 (ablation) + Tier 3 (patching) + tutte le heatmap
(§14.3b) nella 2a sessione.

**Poi, correzione dura dell'utente (2a sessione, DA NON DIMENTICARE MAI — vedi anche §2 e la
memoria persistente `feedback_hpc_access_always_available`):**
```
non ti azzardare a scrierla mai piu perchè noi abbiamo sempre accesso all'hpc bocconi. scrivi su
istruzioni questa cosa
```
(riferito a frasi tipo "non abbiamo accesso al training" che erano finite nel `.tex` — RIMOSSE,
vedi §2 per la regola permanente).

**Poi, 3a sessione — generalizzazione a n=64:**
```
vorrei che rifacessi le cose per n64 con la stessa distribuzione e anche per ER n64.
```
→ Tier 1/2/3 ripetuti interamente a n=64 su path_union E su ER (checkpoint già esistenti,
nessun training nuovo) — esito in §14.3c, **diverso e più ricco** di n=40 (il modo di fallire si
inverte: a n=64 path_union il reach non crolla ma il cut sì, per sovra-connessione).

### 14.1 Da dove viene

Report 5 (verdetto) e Report 6 (outlook) dicevano entrambi la stessa cosa: tutta l'evidenza fin
qui è **comportamentale** (output del modello contro oracoli/dati), mai **meccanicistica**
(attention weights, matrice di read-out, embedding). `model.py` espone da sempre
`attention_maps(x)` e `hidden_states(x)`, ma un grep sul repo conferma che `attention_maps` non
era MAI stato usato in nessuno script prima di questa sessione. Report 7 apre il trunk, puntato
sul puzzle lasciato aperto dal Report 6 Thread B (§13.8): stesso modello, stessi pesi, stesso
`n=40` — uno split two-chains `(a, 40-a)` con `a` piccolo (es. `4+36`) viene risolto **esatto**
fino a ~38 hop, uno bilanciato (`17+23`, `20+20`) si ferma esatto al muro `3^L=9` e poi è **zero**.

**Nota tecnica imparata in questa sessione (utile per chiunque tocchi la distribuzione
disjoint-paths/path_union — es. per capire cosa vede davvero il modello a training time):**
`generate_path_union_graph(n, rng, max_paths=4)` in `data.py` funziona così: (1) il numero di
componenti `k` è estratto **uniformemente a caso tra 1 e 4** (`k ~ Uniform{1,2,3,4}`), quindi
circa un quarto delle volte il training vede un singolo path che copre tutti gli n nodi; (2) se
`k>1`, si scelgono `k-1` punti di taglio **uniformemente a caso senza rimpiazzo** tra le `n-1`
posizioni interne — le lunghezze dei path risultanti **non sono fisse né uniformi tra loro**
(sono gli "spacing" fra punti uniformi casuali, quindi tendenzialmente sbilanciate, non ~n/k
ciascuna); (3) i `k` path **partizionano tutti gli n nodi** (nessun nodo isolato/di padding);
(4) la permutazione dei nodi (per rompere scorciatoie di indice) avviene FUORI dal generatore, a
chiamata. Questo spiega perché, con questa distribuzione, il modello vede spesso split molto
sbilanciati (un pezzo minuscolo + uno enorme) ma anche 4 pezzi comparabili, mai in modo
sistematico/fisso — è la base empirica per l'ipotesi "il modello impara a gestire bene i casi con
una componente piccola risolvibile" del Report 7.

### 14.2 Le tre ipotesi (dichiarate PRIMA di guardare i dati, per restare falsificabili)

1. **Propagazione genuina**: il modello traccia davvero la distanza; dovrebbe reggere a
   relabeling casuale, a dove sta la componente corta, all'ordine interno del path lungo.
2. **Default "componente corta risolta"**: il modello risolve per intero la componente
   abbastanza piccola da stare in capacità (`short_len≲9`) e mette "connesso" di default a
   tutto il resto — una scorciatoia corretta finché c'è **esattamente** una componente piccola.
3. **Scorciatoia di label/ordine**: il modello sfrutta l'indice fisso dei nodi invece della
   struttura astratta (indebolita in partenza: `eval_asym_chains.py`, quello che ha prodotto
   `tab:basplit` del Report 6, **permuta già** ogni singolo grafo di test in modo indipendente e
   casuale prima di darlo al modello — quindi i numeri di Report 6 sono GIÀ media su centinaia di
   relabeling indipendenti).

### 14.3 Cosa dicono i dati (Tier 1, 4 seed `n40_path_union` di Report 6, TUTTO fatto)

- **Ipotesi 3: REFUTATA.** Fixed-label vs random-label danno lo stesso identico esito a ogni
  split (Report 7 tab 2). La matrice di read-out non ha struttura per-indice (norme ‖w_j‖
  uniformi, cosine tra righe ~0.06); lo skip-connection diretto `E_in @ W_out^T` (bypassa i
  transformer block) porta un segnale ~100× più debole di quello che decide davvero la
  predizione — qualunque cosa succeda, passa per i due layer, non per un lookup sull'indice.
- **Ipotesi 2: corretta ma in forma più STRETTA del previsto.** Il test decisivo è la
  **falsificazione a tre componenti** (Report 7 tab 4, `data.py::generate_three_way_split_graph` +
  `eval_three_way_split.py`, NUOVI): un grafo con una componente piccola + due componenti grandi
  **non connesse fra loro**. Se l'ipotesi 2 fosse "tutto ciò che non è la piccola è connesso", il
  modello dovrebbe fondere le due grandi. **Non lo fa**: `cut(L1,L2)=1.000` sempre, ogni seed, ogni
  taglia. MA il reach dentro ciascuna grande componente NON recupera (resta a 0.71–0.86, identico
  al reach di uno split bilanciato a due componenti della stessa taglia) — il "completamento
  extra" è specifico di un canvas a **esattamente due** componenti, non un default generico.
- **Ipotesi 1: corretta quando il segnale di completamento non è attivo.** Oltre la zona di
  transizione (`a≳11`), la frazione complessiva di coppie predette connesse combacia quasi esatto
  con quella di un oracolo puro a distanza `≤9` (0.36 modello vs 0.35 oracolo, calcolato in forma
  chiusa) — cioè il modello torna a comportarsi come matrix-powering puro (Report V) non appena il
  trucco a-due-componenti non si applica.
- **Il logit grezzo `h_i^T w_j` (pre-bias) decade IN MODO LISCIO**, non a gradino: saturato e
  fortemente positivo per `a≤7`, attraversa lo zero proprio in `a=8..11` (dove crolla l'accuracy),
  poi risale piano fino `a=20` — lo strappo comportamentale a `a=8` è solo la soglia binaria
  applicata a un segnale continuo. Il logit di cut invece resta **sempre negativo** (mai cambia
  segno), solo si affievolisce: la decisione "componente diversa" degrada in confidenza ma non in
  segno, coerente col cut-accuracy ≈1.0 per tutto lo sweep.
- **Attention rollout (2 layer, `α_2 @ α_1`, PRIMO uso reale di `attention_maps` nel progetto):**
  per un nodo a metà del path lungo l'attention raggiunge TUTTI i nodi della componente lunga sia
  ad `a=4` che ad `a=20` — non è "troppo corto raggio". Quello che cambia è quanta di quella massa
  **fuoriesce** sulla componente sbagliata: 0.3% ad `a=1` → 31% ad `a=20`, salita liscia e
  monotona, strettissima sui 4 seed (<1 punto percentuale di spread).

### 14.3b Tier 2/3 — FATTO (2a sessione, 2026-07-08). Heatmap, ablation, patching.

Su richiesta esplicita dell'utente ("fai tutti gli altri esperimenti... mostra pesi e attention con
heatmap"), completati anche Tier 2 e Tier 3 (tranne i due punti non ancora fatti, vedi §14.6 — non
per mancanza di accesso a HPC, che il progetto ha sempre tramite l'utente, ma perché richiedono un
training nuovo mai lanciato in questa sessione). Tutto rilanciato sui 4 checkpoint reali dopo un
secondo giro di `scp` (i `.pt` erano stati cancellati a fine 1a sessione, come da regola).

- **Heatmap (PRIMO uso reale in tutto il progetto sia di `attention_maps` sia di una vera
  visualizzazione dei pesi):** `S=QK^T/√d_h` e α per layer 0/1 mostrano che layer 0 è identico fra
  `a=4` e `a=20` (banda locale lungo la diagonale, rotta esattamente al confine componente) — la
  differenza è TUTTA in layer 1: ad `a=4` il rollout a 2 layer è un blocco quasi uniforme su TUTTA
  la componente lunga; ad `a=20` è visibilmente **block-diagonal** (due blocchi separati, zona
  centrale attenuata) — esattamente la stessa cosa mostrata numericamente dalla leak-fraction. Le
  norme Q/K/V per nodo sono IDENTICHE in forma fra `a=4` e `a=20` (la differenza è nella direzione,
  non nella magnitudo). `W_out`/`W_in` grezzi: nessuna struttura per-indice visibile (rumore),
  confermando ulteriormente la refutazione dell'ipotesi 3.
- **Ablation whole-layer (dissociazione pulita, decisiva):** spegnere QUALSIASI dei due layer di
  attention collassa il reach a ~0.06–0.29 A OGNI split (anche `a≤7`) — il segnale di completamento
  NON è un circuito separato, usa la STESSA macchina di attention del reach ordinario oltre-muro.
  Spegnere l'MLP layer 1 lascia il reach quasi intatto (0.87–0.99) ma fa CROLLARE il cut da ~1.0 a
  0.26–0.62 — **reach e cut dipendono da parti diverse del trunk** (cut ← MLP1, reach ← entrambi i
  layer di attention). Bypass completo (niente transformer block) → reach flat ~0.33 a ogni split
  (niente completamento possibile senza mixing). Risultato pulito su tutti e 4 i seed.
- **Activation patching (esplorativo, ESITO ONESTAMENTE INCONCLUSIVO):** patchare l'embedding
  finale da un donor risolto (`a=7`) in un recipient fallito (`a=10` o `a=20`) porta SEMPRE il reach
  a 1.0 — ma lo fa ANCHE un controllo random (patch di posizioni non allineate al confine) quasi
  altrettanto bene (0.89–1.0): è un artefatto del read-out per-riga (`z_ij` dipende solo da `h_i`),
  non una prova di localizzazione. Il patch PRIMA del layer 1 (block 0) è più informativo perché
  NON sempre aiuta: per lo split adiacente `7→10` rescue quasi completo (0.98); per lo split
  mismatched `4→20` **peggiora** (0.31, sotto il baseline 0.71) — block 1 si confonde con uno stato
  ibrido costruito da due grafi di forma diversa. Riportato nel `.tex` come negativo/esplorativo, non
  come evidenza decisiva.

### 14.3c Generalizzazione a n=64 — FATTO (3a sessione, 2026-07-08). path_union E ER, esito diverso da n=40.

Su richiesta esplicita dell'utente ("rifai le cose per n64 con la stessa distribuzione e anche per ER
n64"). Checkpoint **già esistenti** su HPC da Report 6 Thread A (`runs/report6/a1_train/
n64_{path_union,er}_roberta_linear_lam0_seed{1000..4000}/last.pt`) — **NESSUN training nuovo**, solo
eval-only con gli stessi script Tier 1/2/3, sweep `a=1..32` (`n//2` per n=64). Risultato: **la storia
non si ripete identica, si arricchisce in modo genuinamente nuovo.**

- **ER a n=64: NESSUN segnale di completamento, mai.** Reach sale liscio da 0.21 (a=1) a 0.39 (a=32),
  **sempre** entro pochi punti dall'oracolo a distanza `≤9` (calcolato in forma chiusa) — zero
  scogliera, zero pavimento, zero recupero. Cut resta `≥0.997` all'intero sweep, **anche allo split
  perfettamente bilanciato**. Prova più pulita del report: il segnale di completamento **non è
  un'inevitabilità architetturale** — è qualcosa che la distribuzione path_union insegna
  specificamente; un modello che non l'ha mai vista non lo acquisisce, a NESSUNO split.
- **path_union a n=64: il reach regge (anzi non crolla mai), ma il CUT crolla — il modo di fallire si
  INVERTE rispetto a n=40.** Reach decade liscio da ~1.0 (a=1) a ~0.82 (a=32), **mai** un crollo netto
  come a n=40 (dove a=10 scendeva a 0.65) — il modello a n=64 tende a **sovra-connettere** invece di
  sotto-connettere oltre capacità. Di conseguenza il cut crolla da ~0.9 a ~0.24 da a=1 a a=20-32 — **il
  contrario esatto di n=40** (dove cut restava ~1.0 per tutto lo sweep). Il logit grezzo di cut
  **cambia segno** (diventa positivo) oltre a≈13, non solo si affievolisce come a n=40. L'exact-match a
  `a` piccolo è rumoroso fra seed (a=4: 0.43/0.55/0.79/0.00) — coerente con la nota già in §13.8 che
  n64 path_union è più seed-lottery di n40.
- **Il test a tre componenti SI ROMPE a n=64 path_union (ma non a n64 ER).** ER: `cut(L1,L2)≥0.998`
  a ogni taglia, identico a n40 — la discriminazione "quale componente" regge SEMPRE con training ER,
  a qualunque n. path_union: a `small=1`, il modello fonde le due componenti grandi (31 e 32 nodi) il
  **91% delle volte** (`cut(L1,L2)=0.093`) — esattamente la scorciatoia grezza dell'ipotesi 2 che n=40
  aveva refutato! L'errore si riduce salendo `small` (0.09→0.79 da small=1 a 10), stessa direzione
  del cut che degrada nello sweep a due componenti.
- **Ablation a n=64: conferma che il layer-1 di attention è un meccanismo a doppio taglio.** Spegnere
  l'attention layer 1 collassa il reach (0.05-0.10, come a n40) **ma FIXA il cut** (torna a ≥0.93,
  molto sopra il baseline degradato) — è la STESSA macchina che costruisce sia il completamento utile
  sia la sovra-connessione dannosa; spegnerla scambia un fallimento per l'altro, non risolve nessuno
  dei due indipendentemente. Cut-dipende-da-MLP1 resta valido a n64 (sia path_union che ER).
- **Patching a n=64:** stesso esito qualitativo di n40 (patch finale ≈ controllo random, artefatto
  del read-out per-riga); la coppia mismatched (4→32) qui NON peggiora col patch after-block0 (a
  differenza di n40) — non lo leggiamo come una differenza sostanziale, dato il caveat architetturale
  già stabilito, solo come prova che il "peggioramento" visto a n40 non è una proprietà generale del
  design di patching.

**File/comandi nuovi (3a sessione):** stesso set di script Tier 1/2/3, ora generalizzati con
`--tag_glob`/`--tag_prefix`/`--suffix`/`--n`/`--title_tag` per aggregare/plottare condizioni diverse
senza sovrascriversi (`plot_mechanistic_asym_chains.py`, `plot_mechanistic_heatmaps.py`,
`plot_ablation_patch.py` — vedi le firme aggiornate negli script). Output in `runs/report7/{mechanistic,
three_way,heatmaps,ablation,patch}/n64_{pathunion,er}_seed{S}/`, figure in `runs/report7/report7_figs/
r7_*_n64_{pathunion,er}.png`. Scp usato (stesso pattern §14.5, download in `/tmp/r7_ckpts_n64/`,
cancellato a fine sessione):
```
mkdir -p /tmp/r7_ckpts_n64
for s in 1000 2000 3000 4000; do
  scp hpc:~/transformer-for-graphs/runs/report6/a1_train/n64_path_union_roberta_linear_lam0_seed${s}/last.pt \
      /tmp/r7_ckpts_n64/pathunion_seed${s}.pt
  scp hpc:~/transformer-for-graphs/runs/report6/a1_train/n64_er_roberta_linear_lam0_seed${s}/last.pt \
      /tmp/r7_ckpts_n64/er_seed${s}.pt
done
```

### 14.4 File nuovi (Tier 1, tutti smoke-testati prima di toccare checkpoint reali)

- `data.py::generate_three_way_split_graph(n, small_len, large_split=None)` — tre path disgiunti,
  uno piccolo + due grandi NON connessi fra loro.
- `eval_three_way_split.py` (eval-only): il test di falsificazione, output
  `runs/report7/three_way/<tag>/three_way_split.json`.
- `mechanistic_asym_chains.py` (eval-only, IL file principale): sweep denso comportamentale
  `a=1..20` + confronto random/fixed-label, decomposizione `h_i^T w_j` per tipo-coppia/distanza/
  split, geometria `W_out`/`W_in` (norme, cosine, allineamento skip-connection), attention rollout
  + row-mass + contributo-al-messaggio su split rappresentativi. Contiene `run_with_cache`, un
  forward pass manuale di `RobertaGraphTransformer` che tiene Q/K/V/attention/output — **validato
  numericamente** contro `model.forward_and_embeddings` (assert nello script, `--skip_selftest`
  per saltarlo). Output per checkpoint: `metrics.csv`, `readout.csv`, `weights_summary.json`,
  `attn_cache.npz` in `runs/report7/mechanistic/<tag>/`.
- `plot_mechanistic_asym_chains.py` (locale, no-GPU): aggrega i 4 seed, produce
  `runs/report7/report7_figs/r7_sweep_and_logit.png` e `r7_attention_leak.png` + stampa la
  tabella three-way.
- **⚠️ Trappola già presa e fissata in questa sessione**: la prima versione di
  `readout_decomposition` disallineava le coordinate — spermutava le embedding per nodo (`h`) ma
  le confrontava contro `W_out`, che è fisso nelle coordinate di RETE (permutate), non in quelle
  base. Fix: calcolare `h_i^T w_j` PRIMA in coordinate di rete, spermutare SOLO il risultato
  `[n,n]` come si fa già per `pred` (`np.ix_(inv,inv)`), mai spermutare `h` da solo contro un
  `W_out` fisso. Verificato: dopo il fix, `frac_positive` aggregato per `within_long` combacia
  esatto con `reach_long` calcolato indipendentemente dal path comportamentale (0.650 vs 0.648 a
  `a=10`, ecc.) — usare questo cross-check se si riestende lo script.

**File nuovi Tier 2/3 (2a sessione), stesso pattern (eval-only, smoke-testati su modello
giocattolo prima dei checkpoint reali):**
- `mechanistic_heatmaps.py` (eval-only): estrae, in coordinate BASE e mediate su molte
  relabeling casuali (statisticamente stabile, non un'istanza rumorosa), le matrici grezze —
  `scores`/`alpha`/`contrib` per layer 0/1, rollout, Q/K/V per-nodo — su split rappresentativi,
  più i pesi grezzi `W_in`/`W_out`/`W_Q`/`W_K`/`W_V`/`W_O` per layer e i loro valori singolari.
  Riusa `run_with_cache` da `mechanistic_asym_chains.py` (import diretto, niente duplicazione).
  Output: `runs/report7/heatmaps/<tag>/{heatmap_data.npz,raw_weights.npz}`.
- `plot_mechanistic_heatmaps.py` (locale, no-GPU): produce le 8 figure heatmap in
  `runs/report7/report7_figs/r7_heatmap_*.png` (un seed rappresentativo, dato che il pattern è
  già dimostrato strettissimo sui 4 seed dalla leak-fraction di Tier 1 — mediare le ATTENTION
  PATTERN fra seed diversi confonderebbe basi imparate diverse, a differenza di una media di
  scalari).
- `ablation_asym_chains.py` (eval-only): `forward_ablated(model, x, condition)` reimplementa a
  mano il forward di `RobertaGraphTransformer` con un branch (attention o MLP, layer 0 o 1)
  azzerato, o bypass totale. Validato: `condition="baseline"` combacia esatto con `model.forward`.
  Output `runs/report7/ablation/<tag>/ablation.csv`.
- `patch_asym_chains.py` (eval-only, Tier 3): allineamento donor/recipient per "hops dal confine
  componente" (unica corrispondenza sensata fra split di taglia diversa), sotto labeling FISSO
  (canonico) per un allineamento esatto — non permutato come il resto del progetto, di proposito.
  Output `runs/report7/patch/<tag>/patch.json`.
- `plot_ablation_patch.py` (locale, no-GPU): aggrega ablation su 4 seed →
  `runs/report7/report7_figs/r7_ablation_reach_cut.png`; stampa il riassunto patching (nessun
  json aggregato dedicato, i numeri vanno letti/ricopiati dai singoli `patch.json` o dallo stdout
  dello script).

### 14.5 Come sono stati presi i checkpoint (scp manuale, non è un limite di accesso a HPC)

Il sandbox in cui gira Claude non fa `ssh`/`scp` interattivi (DNS non risolve) — è un limite
tecnico di trasferimento file in QUESTA sessione, non un limite di accesso a HPC (vedi la regola
in §2: il progetto ha sempre accesso completo a HPC, training incluso, tramite l'utente). Per gli
eval/analisi locali su un checkpoint specifico, il download resta comunque manuale: l'utente fa lo
`scp` (richiede VPN Bocconi attiva) verso `/tmp`, MAI nel repo, e Claude li cancella subito dopo
l'uso (regola già in §13.7, riconfermata qui):
```
mkdir -p /tmp/r7_ckpts
for s in 1000 2000 3000 4000; do
  scp hpc:~/transformer-for-graphs/runs/report6/a1_train/n40_path_union_roberta_linear_lam0_seed${s}/last.pt \
      /tmp/r7_ckpts/seed${s}.pt
done
```
**NB (2a sessione):** i `.pt` erano stati cancellati a fine 1a sessione (regola del progetto), quindi
questo scp è stato rilanciato una seconda volta dall'utente (serve VPN Bocconi attiva) — capiterà di
nuovo in ogni chat futura che tocchi i checkpoint reali. Cancellati di nuovo a fine 2a sessione.

### 14.6 Stato finale (3a sessione) e cosa resta davvero aperto

**Tier 1 + Tier 2 + Tier 3 a n=40 E la generalizzazione completa a n=64 (path_union + ER) sono
FATTI**, con dati reali su 4 seed per ogni condizione (16 checkpoint in totale: 4 n40 + 8 n64 + 4
n40 riusati), integrati nel `.tex` (§4.1-4.9 = n40, §4.10 = n64, Verdetto e Honest limits
aggiornati con entrambi). Report 7 compila pulito, **20 pagine**, 0 ref indefinite, 0
overfull/underfull.

Restano aperti due punti — **non lanciati in questa sessione, non "fuori portata"** (regola §2: il
progetto ha sempre accesso a HPC/training tramite l'utente):
- **Retrain con loss riequilibrata** (proposto dall'outlook del Report 6): richiede un training
  job nuovo su HPC (non solo eval) — si lancia con `sbatch` esattamente come ogni altro training
  del progetto (§2). `experiments2/train_families_n20.py` non ha un flag per pesare diversamente
  le coppie within/cut nella loss — andrebbe aggiunto (piccola modifica, `BCEWithLogitsLoss` →
  maschera per tipo-coppia) prima di poter lanciare l'sbatch. Se una chat futura vuole farlo:
  aggiungere il flag, preparare il comando `sbatch` e consegnarlo all'utente, poi (quando il
  training è finito) rieseguire `mechanistic_asym_chains.py` sul nuovo checkpoint.
- **Stesso audit con il read-out similarity**: `§13.10` registra che un giro di checkpoint
  similarity (`n40_path_union_roberta_similarity_...`, e possibilmente `n64_..._similarity_...`)
  potrebbe già esistere su HPC dal training "Onda 1" del Report 6 — da controllare (con l'utente, o
  con un `ls`/`find` su HPC) prima di procedere. Se una chat futura lo riprende: verificare che i
  checkpoint esistano, poi ri-puntare `mechanistic_asym_chains.py`/`mechanistic_heatmaps.py` (già
  generici, auto-rilevano il read-out da `eval_families.load_model`, ma NB:
  `run_with_cache`/`forward_ablated` in questa sessione assumono `readout=="linear"` e sollevano
  `NotImplementedError` sul similarity — andrebbero estesi prima).
- Verifica sotto lo stream **mixed** (Report III-V): non fatta, citata come limite onesto nel
  `.tex`. La generalizzazione fin qui ha testato canvas size (n40 vs n64) e distribuzione da zero
  (path_union vs ER), non lo stream mixed opaco (che comunque questo report evita di default per
  principio, §13.2).
- Push del `.tex`/figure/json + tutti gli script nuovi (Tier 1 §14.4 + Tier 2/3 §14.3b + n64
  §14.3c), nessun `.pt` da committare (restano solo su HPC, mai lasciati stabilmente in locale).

### 14.7 Il "rollout" $\alpha_2\alpha_1$ era una scorciatoia troppo grezza — sostituito con lo
Jacobiano vero (4a sessione, 2026-07-09). **NON ancora rilanciato su checkpoint reali quando
questa nota è stata scritta** — vedi "stato lancio" sotto.

**Il problema.** Il rollout usato in Tier 1 #5 e nelle heatmap (§14.3b/§14.4) approssimava "quanto
l'embedding finale del nodo $i$ deriva dal nodo $k$" moltiplicando **solo le due matrici di
attenzione** ($\alpha_2\alpha_1$), scartando $V$, $W_O$, il residuo e la MLP — un proxy grezzo,
mai usato dal modello vero. L'utente l'ha bocciato e ha chiesto il calcolo vero.

**La sostituzione.** Nuova quantità $C_{ik}=\|\partial h_i^{(2)}/\partial h_k^{(0)}\|_F$: lo
Jacobiano ESATTO (autograd reale attraverso il forward pass vero — $V$, $W_O$, residuo, MLP,
derivata vera di LayerNorm, nulla approssimato) fra l'embedding di read-in del nodo $k$ e
l'embedding finale del nodo $i$. Implementata in `mechanistic_asym_chains.py::exact_contribution`
(un backward batched per riga-query $i$, via `is_grads_batched`), con selftest
(`_selftest_exact_contribution`, gira di default in ogni chiamata, confronta contro un doppio
loop non-batched indipendente + differenze finite — combacia a 1e-6/1e-7). Sostituisce il campo
`rollout`/`rollout_mean` ovunque: in `attention_probe` (Tier 1 #5, alimenta la figura leak-fraction)
e in `mechanistic_heatmaps.py::heatmap_probe` (alimenta la heatmap), campo rinominato
`contrib_exact`/`contrib_exact_mean`. Aggiornati `plot_mechanistic_asym_chains.py` (label/titolo,
niente più "attention rollout") e `plot_mechanistic_heatmaps.py` (stesso). Il `.tex` §sec:setup è
stato riscritto per definire $C_{ik}$ correttamente (ho anche corretto lì un errore mio: il
read-in vero applica un LayerNorm dopo `read_in`, che la prima stesura di questo paragrafo aveva
omesso).

**⚠️ COSTO: molto più caro delle quantità precedenti.** $C_{ik}$ costa un backward per riga-query
(≈15–20s per l'intera matrice $n{\times}n$ a $n{=}40,d{=}512$, misurato su Mac CPU/MPS) — contro
$<1$s per $\alpha$/message-contribution/row-mass. Per questo `attention_probe`/`heatmap_probe`
prendono ora un parametro **separato** `--contrib_n_graphs` (piccolo di default, 8) dal
`--n_graphs`/`--attn_n_graphs` usato per le quantità economiche. **L'utente ha chiesto
esplicitamente PIÙ grafi mediati, non pochi** ("sticazzi se ci mette un attimo di più") e di
valutare l'HPC (CPU) invece del Mac in locale.

**Nuovo sbatch, CPU-only:** `scripts/r7_exact_contribution.sbatch` (nuovo, **CPU non GPU** — è solo
autograd/matmul, nessun training) — `medium_cpu`, `--array=0-3` (i 4 seed), `--contrib_n_graphs 64`,
`--cpus-per-task=16` con `OMP_NUM_THREADS=16` ecc. (a differenza degli altri sbatch eval-only del
progetto, che pinnano i thread BLAS a 1 perché girano su GPU — qui il calcolo è tutto CPU, quindi
il multi-threading BLAS va lasciato attivo). Per `mechanistic_heatmaps.py` lo sbatch passa
`--splits 4 20` (gli unici due split che le figure heatmap usano davvero, `PAIR=(4,20)` in
`plot_mechanistic_heatmaps.py` — inutile spendere il calcolo caro sugli altri 6). Stima:
≈2.6h/seed per `attention_probe` (8 split × 64 grafi) + ≈40min per `heatmap_probe` (2 split × 64
grafi) ≈ 3.2h/seed, `--time=06:00:00` per margine, i 4 seed girano in parallelo (medium_cpu ha 8
nodi). Scrive **nelle stesse cartelle output** già esistenti da sessioni precedenti
(`runs/report7/mechanistic/n40_pathunion_seed{S}/`, `runs/report7/heatmaps/n40_pathunion_seed{S}/`)
— sovrascrive `attn_cache.npz`/`heatmap_data.npz` col nuovo campo, gli altri file
(metrics/readout/weights) sono rigenerati identici (non toccati dal cambio).

**STATO: FATTO (4a sessione, completato).** Job `574086` (path_union, n40) girato su HPC, pull
locale fatto (dopo aver rimosso vecchi `.npz`/`.csv`/`.json` locali non tracciati del run
rollout-based, che bloccavano il `git pull` — stesso pattern dell'errore 32/14), figure
rigenerate, **§sec:res-attention e §sec:res-heatmaps riscritte con i numeri veri**. Risultato
onesto: la leak-fraction **non è liscia/monotona** come suggeriva il vecchio rollout (0.3%→31%) —
i numeri veri sono $5.8\%$ ($a{=}1$) → $11\%$ ($a{=}4$) → $21$–$24\%$ ($a{=}7$–$8$) → **picco al
$42.5\%$ proprio ad $a{=}10$** (subito dopo il collasso comportamentale) → scende al $27\%$
($a{=}14$) → risale al $31$–$36\%$ ($a{=}17$–$20$); il picco è riproducibile sui 4 seed, non
rumore. L'heatmap non è un "blocco uniforme" ma una **banda vicino-diagonale che decade con la
distanza**, tagliata nettamente al confine di componente ad $a{=}20$ ma non ad $a{=}4$ (componente
corta troppo piccola per formare un blocco distinto); **scoperta nuova, verificata numericamente**:
l'estremità lontana di ogni path è una sorgente (non ricevitore) di contributo $3$–$4\times$ sopra
la media, a ogni split — coerente con un segnale "questo path si è chiuso" trasmesso verso
l'esterno. Compila pulito, **21 pagine**.
**Nota sul numero di grafi:** `attention_probe` usa `min(contrib_n_graphs, attn_n_graphs)` e lo
sbatch non sovrascrive `--attn_n_graphs` (resta al default 40) → l'`attention_probe` userà 40
grafi/split, non i 64 richiesti (il `heatmap_probe` invece userà i 64 pieni, il suo `n_graphs`
di default è 80). Deciso con l'utente di **non rilanciare** solo per questo, 40 è già un campione
solido.

**⚠️ Sezione patching RIMOSSA dal report (richiesta utente, 4a sessione, 2026-07-09).** L'utente,
dopo aver capito cosa faceva davvero l'esperimento (stesso checkpoint, due grafi di test diversi
mai visti in training, patch fra loro SOLO a inferenza — non due training diversi, la confusione
iniziale), ha chiesto di toglierlo (`toglilo direttamente`): probabilmente perché l'esperimento
era già dichiarato "esplorativo/inconcludente" nel testo (l'artefatto del read-out per-riga rende
il confronto random-vs-allineato poco informativo). **Rimossa l'intera sottosezione
§sec:res-patch** (Question/Setup/Result/tabella `tab:r7patch`), il paragrafo "Patching, briefly"
sotto §sec:res-n64, e i riferimenti in Verdetto/Honest limits — compila pulito, ora **20 pagine**
(era 21). **NON cancellati** (solo non più citati nel `.tex`): lo script `patch_asym_chains.py`,
la sua parte in `plot_ablation_patch.py`, e i dati `runs/report7/patch/**` — restano su disco,
nessuna azione richiesta a meno che l'utente non chieda esplicitamente di cancellarli.

File da pushare (in più rispetto a quanto sopra): `report/7/transformer_for_graphs_7.tex` e
`report/7/transformer_for_graphs_7.pdf` (già aggiornati con la sezione patching rimossa),
`istruzioni.md`.

### 14.8 Richiesto un condizione parallela ER a n=40 per §4.1/§4.4/§4.5/§4.6/§4.7/§4.8 (4a
sessione, 2026-07-09). **NON ancora lanciato.**

**Cosa ha chiesto l'utente.** Finora tutto il Report 7 (§4.1–§4.8, cioè sweep, geometria del
read-out, attention/heatmap, falsificazione a tre componenti, ablation) è fatto **solo sui
checkpoint n40 path\_union-trained**. L'utente vuole la **stessa identica batteria** ripetuta sui
checkpoint **n40 ER-trained** (mai visto path/multi-componenti in training — lo stesso ruolo di
cross-check che l'ER gioca già a n64 in §sec:res-n64, ma qui aggiunto anche a n40), e vuole i
risultati aggiunti nel `.tex` come sottosezioni **"b"** subito dopo ogni "a" esistente (es. §4.1a
= sweep path\_union esistente, §4.1b = sweep ER nuovo), per avere **due esempi affiancati** in
ogni capitolo. **Esplicitamente NON richiesti**: §4.2 (relabelling) e §4.3 (decomposizione
readout, che non ha una tabella propria) — solo §4.1, §4.4, §4.5, §4.6, §4.7, §4.8.

**Checkpoint:** `runs/report4/families_n40/n40_er_roberta_linear_lam0_seed{1000..4000}/last.pt`
(fallback non-bucketato `runs/families_n40/...`, stesso pattern `find_ckpt` di
`scripts/r6_a1_eval.sbatch`) — **esistenti, nessun training nuovo**.

**Script (tutti già generici, verificato leggendo i CLI args — nessuna modifica di codice
necessaria, solo un nuovo sbatch):**
- `mechanistic_asym_chains.py` → copre §4.1 (sweep, `tab:r7sweep`-analogo), §4.4 (geometria
  read-out, `tab:r7wout`-analogo), §4.5 (attention/leak, ora con la contribution esatta).
- `mechanistic_heatmaps.py` → §4.7 (heatmap, `--splits 4 20`, gli unici usati nelle figure).
- `eval_three_way_split.py` → §4.6 (falsificazione a tre componenti).
- `ablation_asym_chains.py` → §4.8 (ablation per-layer).

**Nuovo sbatch `scripts/r7_er_n40.sbatch`** (CPU-only, `medium_cpu`, `--array=0-3`, stessa
struttura di `scripts/r7_exact_contribution.sbatch` ma con tutti e 4 gli script in sequenza per
seed, output taggato `n40_er_seed{S}` in `runs/report7/{mechanistic,heatmaps,three_way,ablation}/`).
**Fix rispetto al lancio precedente**: qui `--attn_n_graphs 64` è esplicito (non lasciato al
default 40) insieme a `--contrib_n_graphs 64`, così la exact-contribution usa davvero 64 grafi
(nel lancio path\_union precedente era rimasta cappata a 40 per lo stesso motivo — vedi §14.7,
lì si è deciso di non rilanciare, ma qui che il job è nuovo si è corretto subito).
`--time=05:00:00` (tre pezzi extra — three\_way/ablation — sono comportamentali, veloci, pochi
minuti; il grosso resta `mechanistic_asym_chains.py`/`mechanistic_heatmaps.py`).

**STATO: FATTO (4a sessione, completato).** Job `574311` girato (con un fix-up
`scripts/r7_threeway_ablation_fix.sbatch` per `eval_three_way_split.py`/`ablation_asym_chains.py`,
mai pushati prima — vedi nota sotto), pull fatto, **§4.1b/§4.4b/§4.5b/§4.6b/§4.7b/§4.8b scritte
nel `.tex`** con dati reali. Risultati chiave, onestamente diversi da quanto ci si aspettava
per simmetria con path\_union:
- **§4.1b (sweep):** reach resta alto ovunque (0.90–0.98, NESSUN floor-and-recovery), ma **cut
  degrada costantemente** da 1.0 (a=1) a 0.15 (a=20) — il **fallimento speculare** di path_union
  (che aveva cut≈1.0 sempre e reach che crollava). Il logit di cut **cambia segno** verso a≈14
  (a differenza di path_union, che restava sempre negativo).
- **§4.4b (geometria read-out):** stessa storia di path_union (norme uniformi, cosine piccola,
  skip-alignment debole) — refuta l'ipotesi 3 anche per ER.
- **§4.5b (leak-fraction):** **liscia e monotona** (0%→30%), a differenza dello strano picco ad
  a=10 di path_union — prova che quel picco è specifico di path_union, non generico.
- **§4.6b (falsificazione a 3 componenti) — IMPORTANTE, corregge un'affermazione precedente nel
  Verdetto:** a n=40 il modello ER **NON refuta** la scorciatoia grezza dell'ipotesi 2
  (cut(L1,L2)=0.01–0.09 a small piccolo!) — il contrario esatto di quanto scritto prima
  ("ER non mostra mai questo fallimento, a nessuna taglia"). Riscritto il paragrafo Ipotesi 2 nel
  Verdetto: non è "ER vs path_union" fisso, è se QUEL checkpoint ha una tendenza a
  sovra-connettere (path_union→sì solo a n64, ER→sì solo a n40).
- **§4.7b (heatmap):** stessa banda vicino-diagonale ma **il taglio netto al confine c'è già ad
  a=4** (non solo ad a=20 come path_union), e **l'effetto "estremità = sorgente" È ASSENTE** —
  quindi quell'effetto è specifico del training path_union, non generale.
- **§4.8b (ablation):** **stesso meccanismo "a doppio taglio"** già visto a n64 per path_union,
  ma qui a **n=40** per ER — spegnere l'attention FIXA il cut e distrugge il reach. Prova che il
  meccanismo è generale, non specifico di canvas/training.
File pushati in più: `data.py`, `eval_three_way_split.py`, `ablation_asym_chains.py` (mai
committati prima — bug scoperto: gli sbatch li chiamavano ma silenziosamente fallivano,
`COMPLETED` fittizio; fixati anche gli sbatch con controllo esplicito dell'exit status).

### 14.9 Terza taglia canvas, n=20, ER e path\_union entrambi (4a sessione, 2026-07-09). **NON
ancora lanciato.**

**Cosa ha chiesto l'utente.** Verificare "per correttezza" che lo stesso approccio (exact
node-to-node contribution, stessa batteria §4.1/§4.4/§4.5/§4.6/§4.7/§4.8) funzioni anche a
**n=20**, sia ER sia path\_union (disjoint paths $1$–$4$), come **ulteriore riscontro**
(terza taglia canvas dopo n40 e n64) — se i risultati sono sensati, va aggiunto come **nuova
sezione** nel report (probabilmente affiancata a §sec:res-n64, la generalizzazione-taglia
esistente, non necessariamente fusa con quella).

**⚠️ Aspettativa da tenere a mente (nota già presente da Report VI Thread B, §13.8 qui sopra):**
a $n{=}20$ **anche lo split più bilanciato $(10,10)$ resta dentro la capacità $3^L{=}9$**
(distanza massima interna $9$, non oltre) — a differenza di $n{=}40/64$ dove lo split bilanciato
supera nettamente la capacità. Quindi il muro comportamentale potrebbe **non mordere allo stesso
modo** a $n{=}20$ (quasi tutto risolto, come già notato per Report 6 Thread B). Questo è
esattamente il motivo per cui vale la pena testarlo (una vera terza taglia, non solo una riconferma),
non un motivo per aspettarsi lo stesso identico risultato di n40/n64.

**Split**: il codice calcola già i range giusti da `mcfg.n` senza bisogno di override — a $n{=}20$,
`n//2=10` dà sweep completo $a{=}1..10$ e split rappresentativi $\{1,4,7,8,10\}$ (la formula
`sorted({s for s in (1,4,7,8,10,14,17,n//2) if s in splits})` scarta automaticamente $14,17>10$).

**Checkpoint (esistenti, nessun training nuovo):**
- path\_union: `runs/report6/a1_train/n20_path_union_roberta_linear_lam0_seed{1000..4000}/last.pt`.
- ER: `runs/repro_paper_n20_roberta/n20_p008_unrestricted_seed{1000..4000}/best.pt` (fallback
  bucketato `runs/report3/repro_paper_n20_roberta/...` — **NB errore 34: il bucketato ha SOLO
  json/png, il `.pt` vero è nel path non-bucketato**, messo per primo nel `find_ckpt` di
  `scripts/r6_a1_eval.sbatch` ma l'ordine non conta, `find_ckpt` prova entrambi).

**Nuovo sbatch `scripts/r7_n20.sbatch`** (CPU-only, `medium_cpu`, `--array=0-7` = 2 distribuzioni
× 4 seed, stessa struttura a 4 pezzi di `r7_er_n40.sbatch`). Due scelte deliberate diverse dal
lancio n40 (imparate dagli errori lì):
- `--attn_n_graphs 64` esplicito insieme a `--contrib_n_graphs 64` **fin dal primo lancio** (a
  differenza del primo giro n40 path\_union, dove `attn_n_graphs` era rimasto al default 40 —
  vedi §14.7).
- `mechanistic_heatmaps.py` gira su **tutti e 5** gli split rappresentativi (non una coppia
  fissa `--pair`): a $n{=}20$ non si sa ancora, prima di vedere lo sweep comportamentale, quale
  coppia "risolto vs fallito" sia la più informativa (vedi nota sopra: potrebbe non essercene
  una netta). La scelta della coppia per le figure si fa DOPO, con `plot_mechanistic_heatmaps.py
  --pair A B` (già supportato, nessuna modifica di codice).

**STATO: FATTO (4a sessione, completato).** Job `574336` girato, pull fatto, nuova
sottosezione §sec:res-n20 scritta subito dopo §sec:res-n64, nel Verdetto e nell'abstract del
piano esperimenti aggiornati di conseguenza. Esattamente come previsto **il muro NON morde per
path_union** (exact ≈1.0 a ogni split, $10{,}10$ incluso — tutto entro capacità) — ma questo si
è rivelato comunque informativo: la **leak-fraction continua a salire lisciamente (0.4%→36%)
anche con comportamento perfetto ovunque**, prova che il segnale meccanicistico non è solo un
sintomo del muro comportamentale. **ER a n20 invece NON risolve bene a nessuno split** (exact
picco 46% ad a=3, poi 0 ad a=6–9) per una ragione che non è il muro-capacità (niente lo supera
qui) — legge come sensibilità OOD generica ai grafi a due componenti a questa taglia. Falsificazione
a 3 componenti **pulita per ENTRAMBI** (cut(L1,L2)≥0.95) — conferma pulita che il merge grezzo
richiede di superare la capacità (coerente col fix del Verdetto in §14.8). Ablation: pattern
"a doppio taglio" presente ma più debole/distribuito a n20 (reach più robusto alle ablation
rispetto a n40/64). Report compila pulito, **28 pagine**, 0 ref indefinite.

**File da pushare (sessione completa):** `report/7/transformer_for_graphs_7.tex` +
`transformer_for_graphs_7.pdf` (tutte le sezioni "b" + §sec:res-n20 + Verdetto/Piano
aggiornati), `istruzioni.md`. Nessun altro file di codice nuovo in questa fase (tutti gli
script erano già pronti/pushati nelle fasi precedenti di questa sessione).

**Due fix successivi allo §sec:setup (stessa sessione, dopo §14.9):** (1) l'`align*` con
equazione+commento affiancati (Q/K/V/S/α/AttnOut/Ĥ/H) mandava alcune righe **fuori dal margine
destro della pagina** (testo tagliato, es. "does no[t]...") — nessun warning "Overfull" in log,
scoperto solo guardando il PDF pagina per pagina. **Non fidarsi del log da solo per l'overflow
matematico**: renderizzare la pagina (`pdftoppm`) e guardarla. Fix: equazioni pulite in `align*`
(solo `=`, niente testo affiancato), spiegazioni spostate in prosa sotto (va a capo normale).
(2) Aggiunto, su richiesta utente, il calcolo **entry-per-entry** (non solo a livello di
matrice) per ogni quantità del forward pass — verificato riga per riga contro `model.py`
(bias inclusi per Q/K/V/W_O/FFN, prima omessi nell'equazione a matrice; LayerNorm con
varianza non corretta/biased come fa PyTorch; GELU esatta $x\Phi(x)$). Un refuso di battitura
(pedice `_\cdot` mal posizionato, sembrava un punto sporco) trovato e corretto nella stessa
passata. Report ora **29 pagine**.

### 14.10 Richiesta di rifare tutto anche col readout SIMILARITY (4a sessione, 2026-07-13).
**NON ancora lanciato.**

**Richiesta dell'utente, VERBATIM (come richiesto esplicitamente di riportare):**
```
voglio intanto che rifai tutti quanti gli esperimenti (dal 4.1 in poi tutti quanti) però per i
checkpoint con readout similarity. voglio capire se è una cosa che dipende solo dal readout
oppure se anche per il similarity impara lo stesso. poi aggiungiamo al report una sezione per
questo. aggiorna istruzioni e report per giustificare questa cosa e spiegare i dubbi.
```
Contesto immediatamente precedente (perché la domanda nasce lì): l'utente ha notato che con
il readout lineare $z_{ij}=h_i^\top w_j+b_j$, il vettore $w_j$ deve imparare qualcosa di
generale su ogni nodo $j$ che vada bene combinato con QUALSIASI $h_i$ — un readout "a lookup
per nodo", non una funzione simmetrica $g(h_i,h_j)$ dei due embedding. L'utente vuole scrivere
un paper e non vuole che i risultati siano un artefatto di questo readout particolare. Ha
chiesto se esistono readout più generali (risposta data in chat: dot-product/GAE, cosine —
**già nel codice come "similarity"**, bilineare $h_i^\top M h_j$, concat+MLP). Chiesto
**scope**: l'utente ha scelto **il disegno 2×2 completo** (path_union E ER, non solo
path_union) — stessa struttura a coppie già usata per linear a n40.

**Perché questo era GIÀ un limite dichiarato nel report:** §sec:limits (Honest limits) diceva
già "the same audit repeated with the similarity read-out, whose logit is a direct function
of both endpoints' embeddings and may not show the same asymmetric, target-specific
completion signal at all" — questa sessione lo trasforma da limite aperto a esperimento in
corso.

**⚠️ Problema scoperto SUBITO (bug preesistente, non di questa richiesta): 3 dei 4 script
usati in tutto il Report 7 rifiutavano ESPLICITAMENTE i checkpoint similarity** (`raise
NotImplementedError` se `readout != "linear"`), perché la loro ultima riga chiamava
`model.read_out(h)` direttamente invece di gestire anche il ramo `similarity` (che non ha
`read_out`, solo `sim_scale`/`sim_bias`). **Fixato (verificato smoke-test su checkpoint
giocattolo similarity prima di toccare checkpoint reali, stesso pattern del progetto):**
- `mechanistic_asym_chains.py::run_with_cache` — ultima riga ora fa il branch
  linear/similarity (rispecchia `model.forward_and_embeddings`); guardia in cima rimossa
  (era ridondante con quella in fondo). **NUOVE funzioni**
  `readout_decomposition_similarity` (usa $\cos(h_i,h_j)$ al posto di $h_i^\top w_j$ — la
  quantità che il readout similarity guarda DAVVERO, dato che non c'è un $w_j$ fisso) e
  `weights_geometry_similarity` (**solo $E_{in}$** — non esiste un $W_{out}$ per questo
  readout, solo due scalari `sim_scale`/`sim_bias`, riportati anch'essi). `main()` sceglie la
  coppia giusta di funzioni in base a `readout` rilevato da `load_model`; `weights_summary.json`
  ora include sempre `readout_kind` per chiarezza a valle.
- `ablation_asym_chains.py::forward_ablated` — le due chiamate dirette a `model.read_out(h)`
  sostituite da un helper `_apply_readout()` che fa lo stesso branch; guardia rilassata
  (blocca solo `arch != "roberta"`, non più il readout).
- `mechanistic_heatmaps.py::raw_weights` — salta `W_out`/`b_out`/`sv_W_out` per similarity
  (non esistono), salva `sim_scale`/`sim_bias` al loro posto; guardia rilassata allo stesso modo.
- `eval_three_way_split.py` — **NESSUNA modifica necessaria**, era già agnostico al readout
  (usa solo `model.forward()`, che già smista internamente).
- **Script di plot aggiornati per non crashare** su output similarity:
  `plot_mechanistic_asym_chains.py::fig_sweep_and_logit` (autodetect colonna `mean_hTw` vs
  `mean_cos` in `readout.csv`, etichette di conseguenza) e
  `plot_mechanistic_heatmaps.py::fig_weight_geometry`/`fig_qkvo_raw_weights` (versione
  ridotta — solo pannelli $E_{in}$ — quando `weights_summary.json` dichiara
  `readout_kind=="similarity"`, e il pannello dei valori singolari salta `W_out` se assente).

**Checkpoint (esistenti, nessun training nuovo — confermato da §13.10 e dal path non-bucketato
già noto per families_n40):**
- path_union-similarity n40: `runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed{1000..4000}/last.pt`
  (allenati in Report 6, 16a sessione, "Onda 1", job 551391).
- ER-similarity n40: `runs/report4/families_n40/n40_er_roberta_similarity_lam0_seed{1000..4000}/last.pt`
  (fallback `runs/families_n40/...`).

**Nuovo sbatch `scripts/r7_similarity_n40.sbatch`** (CPU-only, `medium_cpu`, `--array=0-7` = 2
distribuzioni × 4 seed, stessa struttura a 4 pezzi con controllo esplicito dell'exit status
già usata per `r7_er_n40.sbatch`/`r7_n20.sbatch`). `--attn_n_graphs 64`/`--contrib_n_graphs 64`
fin da subito, `--splits 4 20` per le heatmap (stessa coppia rappresentativa del linear, per
confronto diretto).

**STATO: LANCIATO, parzialmente completato.** Job `583272` (`--array=0-7`, `medium_cpu`,
`--time=05:00:00`): 5/8 task `COMPLETED` (seed 2000 di path_union-similarity + **tutti e 4** i
seed ER-similarity — "COMPLETED" qui è affidabile, grazie al controllo esplicito dell'exit
status aggiunto dopo il bug del §14.8, quindi vuol dire davvero che tutti e 4 i pezzi sono
riusciti). **3/8 in `TIMEOUT`** (path_union-similarity seed 1000/3000/4000, indici array 0/2/3):
nodi condivisi più lenti del solito (già visto altre volte, es. errore 18) — a giudicare dai
log, servono **~7h** contro le 5h richieste, sopra il cap di `medium_cpu` (6h10).
**Rilanciati sulla partizione `compute`** (3 nodi, cap 3 giorni, nessun rischio di ritaglio):
`sbatch -p compute --time=20:00:00 --array=0,2,3 scripts/r7_similarity_n40.sbatch` (rifà da zero
i 3 seed mancanti, sovrascrivendo l'eventuale output parziale — nessuna logica di skip nello
script, corretto e atteso). **Da fare poi (chat che riprende, quando anche questi 3 sono
finiti):** pull, poi scrivere la nuova sezione **§sec:res-similarity** (dopo §sec:res-n20,
prima del Verdetto) con prosa+tabelle+figure sui dati veri — confrontare esplicitamente contro
le sezioni "a"/"b" lineari già scritte per rispondere alla domanda dell'utente (dipende dal
readout, o il similarity impara la stessa cosa?); aggiornare Verdetto/Honest-limits di
conseguenza (questo era uno dei due punti aperti lì). **Non scrivere niente finché i dati reali
non sono arrivati.**

**Richiesta successiva dell'utente (stessa sessione): le tabelle §sec:res-n20 (tab:r7n20sweep,
tab:r7n20sweeper) mostravano solo 5 split rappresentativi ($a{=}1,4,7,8,10$) — chiesto
esplicitamente **tutti e 10** ($a{=}1..10$, cioè $(1,19),(2,18),\dots,(10,10)$), per entrambe
le distribuzioni. **FATTO**: ricalcolati con precisione dai `metrics.csv` (non a memoria/da
numeri già scritti prima — richiesta esplicita "dai risultati corretti"), entrambe le tabelle
ora hanno le 10 righe complete; tolta la prosa "not tabulated" per $a{=}3$ (ER), ora in
tabella. Compila pulito, **29 pagine** (invariate).

### 14.11 §4.7 — aggiunta la matrice $C_{ik}$ pura come figura standalone (5a sessione, 2026-07-14).

**Richiesta dell'utente:** voleva $C_{ik}$ come heatmap $40\times40$ subito sotto la Figura 3
(la leak-fraction aggregata, §sec:res-attention) — non solo dentro la figura 2×3 di §4.7
(`fig:r7rollout`, che la mostra già ma impacchettata con message-contribution e row-mass).
**FATTO**: nuova funzione `plot_mechanistic_heatmaps.py::fig_contrib_matrix_only` (nessun nuovo
calcolo, riusa `contrib_exact` già in `heatmap_data.npz`) → figura dedicata
`r7_heatmap_contrib_matrix{suffix}.png`, un pannello per split ($a{=}4,20$), senza altri pannelli.
Inserita nel `.tex` **subito sotto** `fig:r7attn` (nuovo `fig:r7contribmat`) e sotto
`fig:r7attner` (nuovo `fig:r7contribmater`) — non dentro §sec:res-heatmaps. Compila pulito, **30
pagine**.

### 14.12 Sezione §sec:res-similarity SCRITTA con i dati veri (5a sessione, 2026-07-14). Risultati
sorprendenti: il completamento non è un artefatto del readout lineare, ma il readout decide
quanto costano le imperfezioni del trunk.

**Stato di partenza:** i 3 task TIMEOUT del job `583272` (relanciati su `compute`,
`--array=0,2,3`) sono tornati `COMPLETED` (verificato con `sacct`) — tutti e 8/8 task
similarity-n40 completati. Pull fatto (commit HPC coi dati, poi `git pull` locale): 32 nuove
cartelle in `runs/report7/{mechanistic,heatmaps,three_way,ablation}/n40_{pathunion,er}_similarity_seed{1000..4000}/`.

**Numeri chiave (letti direttamente da `metrics.csv`/`readout.csv`/`weights_summary.json`/
`attn_cache.npz`/`three_way_split.json`/`ablation.csv`, MAI a memoria):**
- **Sweep path\_union-similarity:** reach (long) non crolla MAI sotto $0.91$ (nessun floor a
  $0.58$--$0.65$ come il lineare) e recupera a $0.995$ ad $a{=}20$; cut $\ge0.995$ ovunque.
  Exact match a $a{=}8$--$11$ = $0.500$ **nasconde bimodalità netta per seed** (2 semi a 1.000,
  2 a 0.000, verificato riga per riga nei `metrics.csv` — NON un degrado uniforme): scritto
  esplicitamente in prosa, non lasciato nella media (regola preferenze §3).
- **Sweep ER-similarity:** reach parte GIÀ imperfetto ad $a{=}1$ ($0.716$) e sale
  MONOTONICAMENTE con $a$ (ordine opposto a ogni altra condizione del report); cut $=1.000$
  sempre; exact $=0$ ovunque (troppi pair per un match esatto anche con reach alto).
- **Geometria readout:** niente $W_{out}$ (solo $E_{in}$ + due scalari `sim_scale`/`sim_bias`
  globali). Nessuna scorciatoia per-indice (norme uniformi, cosine bassa) — refuta ipotesi 3
  anche qui. **Osservazione nuova**: la soglia di decisione $\cos=-\text{bias}/\text{scale}$
  ($\approx0.15$ path\_union, $\approx0.25$ ER) non viene mai avvicinata dal cut-cosine
  nell'intero sweep testato — spiega perché il cut non degrada mai come nel lineare (dove il
  segno di un logit per-coppia illimitato è l'unica difesa).
- **Leak-fraction (contributo esatto):** **path\_union-similarity**: salta da $0\%$ ($a{=}1$) a
  un plateau **piatto** $\approx22$--$23\%$ da $a{=}4$ in poi (non sale/scende come il lineare).
  **ER-similarity**: **$\le0.2\%$ a OGNI split** — praticamente zero, contrasto nettissimo
  confermato visivamente dalle heatmap $C_{ik}$ (quasi perfettamente block-diagonal a $a{=}4$ E
  $a{=}20$, mentre path\_union-similarity ha una banda diffusa che attraversa tutta la matrice
  anche col comportamento quasi perfetto — **dissociazione reale meccanismo/comportamento**: il
  trunk mischia, il readout a coseno non se ne fa ingannare).
- **Falsificazione a 3 componenti:** **ENTRAMBE le distribuzioni quasi perfette**
  ($\text{cut}(L_1,L_2)\ge0.988$, spesso $1.000$) — nessuna traccia del merge grezzo che il
  modello ER-LINEARE mostrava a questo stesso canvas (§14.8, $0.01$--$0.09$).
  Il readout similarity sembra eliminare quel fallimento specifico a $n{=}40$.
- **Ablation:** dissociazione reach/cut molto più pulita che nel lineare. Il `bypass` (niente
  transformer, readout diretto sul read-in) dà GIÀ cut $0.95$--$1.000$ per entrambe le
  distribuzioni (il taglio è quasi "gratis" dalla sola geometria coseno del read-in) con reach
  basso ($0.10$--$0.20$, nessun mixing = nessun reach) — praticamente l'opposto del lineare
  (bypass lì dava cut mediocre $\approx0.66$--$0.69$). Per ER-similarity la dissociazione è
  quasi perfetta: `zero_attn1` uccide il reach ma lascia cut a $1.000$; `zero_attn0` lascia il
  reach quasi intatto ma uccide il cut.

**Risposta diretta alla domanda dell'utente ("dipende dal readout, o il similarity impara lo
stesso?"):** il **segnale di completamento a due componenti non è un artefatto del readout
lineare** — il trunk lo mostra anche con similarity (leak-fraction non nulla, readout risponde
allo split). Quello che CAMBIA è **quanto quel segnale può nuocere**: il floor duro del reach
lineare e la fragilità del cut (un solo segno di un logit illimitato) sono proprietà del
readout lineare, non del trunk — un margine coseno fisso e globale rende il cut robusto per
ENTRAMBE le distribuzioni a $n{=}40$ col readout similarity, incluso il caso ER che falliva
clamorosamente col lineare allo stesso canvas.

**File toccati:** `plot_mechanistic_heatmaps.py` (nuova `fig_contrib_matrix_only`, §14.11),
`report/7/transformer_for_graphs_7.tex` (nuova §sec:res-similarity con dati veri, Verdetto
esteso con un paragrafo di sintesi, Honest limits aggiornato — rimossa la voce "similarity non
ancora fatta", aggiunta la voce "similarity fatta solo a n40, non n64/n20"), `istruzioni.md`.
Compila pulito, **38 pagine**, 0 ref indefinite, 0 overfull (verificato anche visivamente
pagina per pagina via `pdftoppm`, incluse le figure affiancate e le tabelle a 8 colonne).

**Aperto per una sessione futura (non richiesto esplicitamente, ma coerente coi limiti onesti
appena scritti):** ripetere il confronto similarity-vs-linear a $n{=}64$ e $n{=}20$ per vedere
se la robustezza del cut regge anche dove il lineare fallisce peggio (path\_union a $n64$, ER a
$n40$ — già fatto qui — e la generica sensibilità OOD di ER a $n20$).

### 14.13 Ricompilazione/verifica post-blocco disco + cambio di destinazione del report (6a
sessione, 2026-07-17).

**Blocco disco (non un problema del progetto):** la sessione precedente non era riuscita a
ricompilare per `ENOSPC` sul disco temporaneo della sandbox (`/private/tmp/claude-501/...`), non
sul disco del Mac né sul repo. Questa sessione è iniziata verificando con un comando minimo
(`true`) che lo spazio si fosse liberato — sì, al primo tentativo.

**Ricompilato e verificato** `report/7/transformer_for_graphs_7.tex` (2 passate `pdflatex` da
`report/7/`): **38 pagine**, log ripulito con `grep -iE "error|undefined|overfull"` → **zero
righe**, quindi 0 errori/0 ref indefinite/0 overfull. Estratte a PNG (via `pdftoppm`) e ispezionate
a schermo le due pagine con le modifiche di caption fatte in una sessione precedente ma mai
confermate visivamente:
- **pag. 12 (Figura 4, §sec:res-attention)**: caption ora include `Test set: two-chain graphs at
  the split shown (a=4 left, a=20 right), 64 independently node-relabelled graphs`.
- **pag. 18 (Figura/Tabella 9, geometria read-out/read-in)**: caption ora include `Test set: none
  — static properties of the trained weights, not evaluated on any graph` (corretto: quella
  figura è sui pesi allenati, non valuta nessun grafo — differenza importante rispetto al
  template standard della regola 21, che qui va dichiarata esplicitamente come "none" invece di
  essere lasciata implicita).

Le stesse coppie gemelle ER/similarity di Figura 4 (e la Tabella 4 corrispondente) erano state
sistemate nella stessa sessione precedente con lo stesso pattern — non ricontrollate pagina per
pagina qui (nessuna modifica aggiuntiva fatta), solo il PDF intero ricompilato pulito conferma che
nessuna di quelle modifiche ha rotto la build.

**Nessuna scoperta tecnica nuova sui dati/modello in questa sessione** — solo verifica di build.
L'unica domanda di merito posta dall'utente (se `generate_path_union_graph` produce nodi isolati)
era già interamente documentata in **§14.1** (nessun nodo isolato: i `k` path partizionano SEMPRE
tutti gli `n` nodi, `k ~ Uniform{1,4}`), risposta data da lì senza bisogno di nuova indagine.

**⚠️ Cambio di contesto da ricordare per ogni sessione futura (vedi anche il callout in cima al
file e §3):** da questa sessione in poi il Report 7 **non va più trattato come il deliverable
finale per la prof** — l'utente lo userà come base per costruire a mano un PowerPoint. Continuare
comunque a rispettare tutte le regole di scrittura del report (§3, regole 20/21/55–60) finché
l'utente non chiede esplicitamente di allentarne qualcuna.

### 14.14 Feedback della prof sul PowerPoint del Report 7 (stessa sessione, 2026-07-17). Errori 61/62 + spiegazione similarity.

**Contesto:** l'utente ha mostrato alla prof il PowerPoint costruito a mano dal Report 7 (§14.13: da questa
sessione il report non è più il materiale diretto per la prof, ma resta la fonte da cui l'utente estrae le
slide). La prof ha dato tre osservazioni, tutte sul contenuto mostrato — trascritte e gestite qui.

**1. Error bars mancanti (ora errore 61).** La Figura 1 del report (`fig:r7sweep`, `r7_sweep_and_logit.png`) e
tutte le sue gemelle nelle altre condizioni (ER-trained, `n64`, `n20`, similarity — 8 figure in totale) mostravano
solo la media sui 4 seed, senza error bars. **Fix FATTO**: `plot_mechanistic_asym_chains.py::fig_sweep_and_logit`
ora calcola anche lo std sui seed per ogni punto e disegna error bars (`ax.errorbar`) su ogni curva in entrambi i
pannelli (sweep comportamentale sopra, logit grezzo sotto). Nessun nuovo dato necessario — i valori per-seed erano
già in `metrics.csv`/`readout.csv`, prodotti dagli sbatch già girati, semplicemente lo script di plot ne calcolava
solo la media. **Le 8 figure sono state rigenerate in locale** (no GPU, `python plot_mechanistic_asym_chains.py
--tag_glob "n{...}_seed*" --n {...} --suffix "_{...}" --title_tag "..."` per ciascuna condizione — vedi i comandi
usati, uno per condizione, in `git log`/questa sessione): `r7_sweep_and_logit.png` (n40 path\_union),
`_n40_er`, `_n64_er`, `_n64_pathunion`, `_n40_pathunion_similarity`, `_n40_er_similarity`, più le due gemelle
`_n20_pathunion`/`_n20_er` (orfane, non referenziate nel `.tex` — rigenerate comunque per coerenza). Verificata
visivamente una figura (`r7_sweep_and_logit.png`): gli error bars confermano quanto già scritto in prosa (es. il
rumore di campionamento ad `a=3,6` è genuinamente piccolo/stretto sui 4 seed, non un singolo run rumoroso) e
aggiungono un'informazione nuova non ancora in prosa (il logit di cut ha uno spread per-seed ampio a `a` piccolo,
che si restringe salendo lo split). **Nessuna modifica al testo del `.tex` necessaria** (le caption/didascalie
restano valide, il contenuto qualitativo delle curve non cambia — solo si aggiunge l'informazione di dispersione).

**2. Buco nello sweep proprio sul punto interessante (ora errore 62).** La Figura 3 (`fig:r7attn`,
`r7_attention_leak.png`, la leak-fraction) ha uno spike riproducibile al $42.5\%$ ad $a{=}10$ (§14.7), ma lo split
successivo valutato era $a{=}14$: un buco di tre split ($11,12,13$) esattamente dove serviva capire se lo spike
scende subito dopo il picco o resta elevato. **Fix FATTO nel codice, NON ancora eseguito su HPC**:
- `mechanistic_asym_chains.py`: il default di `attn_splits` (usato dall'`attention_probe` che alimenta
  `attn_cache.npz` → Figura 3 e le sue gemelle) ora è `{1,4,7,8,10,11,12,13,14,17,n//2}` (era
  `{1,4,7,8,10,14,17,n//2}`) — ogni run futuro copre automaticamente l'intorno di un punto notevole già noto.
- Aggiunta una logica di **merge**: `main()` ora, prima di scrivere `attn_cache.npz`, carica il file esistente (se
  c'è) e aggiunge/sovrascrive solo le chiavi degli split appena calcolati, invece di sovrascrivere l'intero file.
  Motivo: l'`exact_contribution` (Jacobiano, un backward per nodo-query) è la parte cara del probe — ricalcolare
  tutti gli 8 split già fatti solo per aggiungerne 3 sarebbe stato uno spreco.
- **Nuovo sbatch `scripts/r7_leak_splits_11_12_13.sbatch`** (CPU-only, `medium_cpu`, `--array=0-3`, i 4 seed della
  condizione PRIMARIA di Figura 3, `n40` path\_union): chiama `mechanistic_asym_chains.py --attn_splits 11 12 13
  --contrib_n_graphs 64` sugli stessi checkpoint/output-dir di sempre (`runs/report7/mechanistic/n40_pathunion_seed{S}`)
  — grazie al merge, aggiunge SOLO i 3 split mancanti, costo ≈3/8 del job originale (§14.7, era ~2.6h/seed per 8
  split → questo dovrebbe stare abbondantemente dentro le 2h richieste). **Da lanciare:**
  `sbatch scripts/r7_leak_splits_11_12_13.sbatch`. **Dopo il pull**: rigenerare la figura con
  `python plot_mechanistic_asym_chains.py` (nessuna modifica allo script di plot necessaria — `fig_attention_leak`
  già itera su tutte le chiavi `aXX` presenti nell'npz, incluse quelle nuove).
- **Le altre gemelle di Figura 3 hanno lo stesso buco** (stesso default pre-fix): ER-trained a `n40`
  (`fig:r7attner`), path\_union/ER a `n64` (`fig:r7n64...` leak, dentro §res-n64), path\_union/ER a `n20`
  (`fig:r7n20leak`), path\_union/ER con readout similarity (`fig:r7simleak`) — **NON ancora ricalcolate**, non
  richiesto esplicitamente dalla prof (solo Figura 3 è stata nominata). Per estenderle: stesso pattern dello
  sbatch sopra, cambiando `CKPT`/`TAG` (e per la similarity, verificare che `mechanistic_asym_chains.py` gestisca
  già il ramo `similarity` per l'`attn_splits` — sì, è agnostico al readout dal §14.10). Non prioritario finché
  l'utente non lo chiede.

**3. Perché il readout similarity ha reach alta per seed ma exact match 0 (spiegato in chat, non richiesto scritto
qui — ma la spiegazione ESISTE GIÀ nel `.tex`).** Riferito a Tabella `tab:r7simsweeper` (§sec:res-similarity,
il modello **ER-trained con readout similarity** a $n{=}40$): il reach (pairwise, sulla componente lunga) è alto
per ogni seed ($0.72$--$0.99$ a seconda dello split) ma l'exact-match (l'INTERA matrice giusta) è $0.000$ a ogni
split. Non è un bug né un'incoerenza: exact-match richiede che **tutte** le centinaia di coppie nella componente
lunga siano corrette **simultaneamente** nello stesso grafo, mentre il reach è una media coppia-per-coppia. Se
ogni singola coppia ha probabilità $\approx0.8$--$0.95$ di essere giusta ma le coppie non sono perfettamente
correlate fra loro (il modello non è ideale), la probabilità che TUTTE lo siano insieme crolla rapidamente con il
numero di coppie — con centinaia di coppie anche un reach per-coppia del $95\%$ implica una probabilità
dell'intera riga/matrice vicina a zero. Il `.tex` lo dice già esplicitamente (§sec:res-similarity, dopo
`tab:r7simsweeper`): *"Exact match is 0 throughout, not because the model does something qualitatively wrong, but
because a reach of 0.7–0.99 spread over hundreds of long-component pairs almost never lands on a perfectly exact
whole-graph match."* Nessuna azione richiesta: è già scritto correttamente, la domanda della prof era di
comprensione, non un errore da correggere.

**File toccati questa sessione:** `mechanistic_asym_chains.py` (default `attn_splits` + merge logic),
`plot_mechanistic_asym_chains.py` (error bars in `fig_sweep_and_logit`), `scripts/r7_leak_splits_11_12_13.sbatch`
(nuovo), le 8 figure `runs/report7/report7_figs/r7_sweep_and_logit*.png` (rigenerate), `istruzioni.md`. **Nessuna
modifica al `.tex`** del Report 7 in questa sessione (le caption restano valide).

---

## 15. Report 8 — dove/come viene combinata l'informazione di connettività

> Blocco aggiunto all'apertura del Report 8 (2026-07-21). **Resta valido per OGNI nuova chat sul
> Report 8.** Poco tempo, obiettivo dichiarato dall'utente: **concludere il paper**.

### 15.0 Richiesta originale dell'utente (trascritta VERBATIM — leggere prima di tutto)

Come già fatto per il Report 7 (§14.0), l'utente ha chiesto di trascrivere qui parola per parola gli
appunti presi durante l'incontro con la prof, così una chat futura può rileggerli per intero invece
che affidarsi a un riassunto:

```
lei plotterebbe anche gli h(1) e poi gli h(2) cosi da vedere se li si vedono i blocchi e se si vede
a che punto c'è già un assetto diciamo. plotterebbe anche la relu.
lei pensa che c'è una parte della rete che dice se sei vicino a questo, sei vicino a quest'altro e
allora sei connesso a quel nodo oppure no: l'obiettivo di questo report è capire come vengono
combinate queste informazioni. forse nell'mlp. provo a plottare attention out/h(l), magari vengono
già combinati li. di base comunque è interessante vedere a che punto le informazioni sono già
combinate. prendi questo come titolo e come idea chiave per questo report.
non le è chiaro come trova informazioni sulla connettività, come vengono combinate.
attenzione, la professoressa mi ha detto che non le piace la misura di node to node contribution
perchè si chiedeva se effettivamente cambiasse prendere h_i(0) o direttamente la colonna del nodo
i, nel senso che può essere che non sia piu correlato strettamente al nodo i. la professoressa mi
ha detto che non le piace la misura di node to node contribution perchè si chiedeva se
effettivamente cambiasse prendere h(0), nel senso che può essere che non sia piu correlato
strettamente al nodo j. cioè ha piu senso rifare quegli esperimenti usando anzichè h(0) già
modificato, A' capito? cioè ha piu senso vedere come cambia il final embedding del node I se
cambio da 0 ad 1 o viceversa un edge. quindi appunto anzichè prendere le colonne di h, prendere
quelle di A'. parliamo se ha senso.
quindi ho già pensato ad un espeirmento come soluzione, però te lo dico in un nuovo messaggio.
intanto volevo solo spiegarti le cose da mettere nel nuovo report 8 e le cose che implenteremo.

poi:
nell'esperimento con la vecchia contribution sembrava che gli estremi avessero tanta attenzione.
per verificare se effettivamnete ciò che capisce è 'questa path è closed' dagli estremi dei
segmenti (come si vede nella contrivution matrix) una cosa che posso fare è fare:
testing di quel modello però quei grafi di due chain li chiudo (chiudo a cerchio le due catene):
anzichè avere due chain faccio due cicli (sempre con catene di (ad esempio) 4 e 36 (ma da fare
sweep come sempre) che diventano due cicli) e vedere:
se impara -> i boundary/estremi del segmento non influenzano e sono solo nodi che si comportano
in modo diverso.
se invece non riesce mai ad imparare allora vuol dire che quei nodi di estremi del segmento sono
super influenti

come modello uso sempre il readout similarity. è giusto dire: 'abbiamo visto che funziona molto
meglio cosi, allora abbiamo deciso di usare questo readout'. una cosa però: non abbiamo messo nel
report 7 la matrice di attenzione plottata. io quindi rifarei in una sezione del report 8
esattamente il plot di tabella 1, figura 1 sia sopra che sotto (con gli estremi che indicano la
varianza però), tabella 4, le weights matrixes visualized: W_out,Win, readout/radi norms,
cos(wi,wj), la figure con le attention scores and normalized relu alpha per layer 0 ed 1 con a = 4
e a =20, il real node to node contribution mass reaching the wrong component con la nuova
contribution, l'exact node to node contrivution matrix per un seed, il layer ablation reasults, il
falsification test. insomma guarda attentamente le slides che ho messo dentro a report 7 (il pdf
trasnfomrer for grphas 7.pdf) e voglio mettere tutti quegli esperimenti in quel modo.
infatti è meglio un setting unico in cui mostrare le cose.
partiamo con n40 union path, in caso poi capiamo per il resto in un secondo momento.

quindi diciamo preapra il report 8 adattandolo agli altri report, prepara poi le istruzioni
incollando esttamente il testo che ti ho scritto fin qui, in modo che ogni chat di claude abbia in
mente il focus del report 8. poi implementa l'esperimento di testing con due cicli (mantieni
sempre quella contribution iniziale, poi faremo la nuova)
mettiti dei goal cosi fai tutto
```

### 15.1 Come è stata interpretata la richiesta (mappatura verbatim → azioni)

- **"Tabella 1, Figura 1 sopra/sotto con estremi/varianza", "Tabella 4", "weights matrixes... W_out,
  Win, readout norms, cos(wi,wj)", "attention scores/alpha layer 0/1 a=4,a=20", "node to node
  contribution mass... l'exact node to node contribution matrix", "layer ablation", "falsification
  test"** → questi numeri fanno riferimento alla numerazione di Tabelle/Figure del **Report 7**:
  Tabella 1 = `tab:r7sweep`, Figura 1 = `fig:r7sweep`, Tabella 4 = `tab:r7wout` (geometria
  read-out/read-in), poi `fig:r7wout`, `fig:r7attnscores`, `fig:r7attn`+`fig:r7contribmat`,
  `tab:r7ablation`, `tab:r7threeway`. La prof vuole che questa batteria intera venga rifatta **con
  il readout similarity come UNICO setting** (non più sparsa fra linear-primario e similarity-a-parte
  come nel Report 7) — **"è meglio un setting unico in cui mostrare le cose"**.
- **Scoperta chiave di questa sessione**: per `n40_pathunion_similarity` questi dati/figure **esistono
  già** su disco da quando il Report 7 ha girato `mechanistic_asym_chains.py`/`mechanistic_heatmaps.py`
  sui checkpoint similarity (§14.10) — **incluso** `r7_heatmap_attention_scores_n40_pathunion_similarity.png`,
  che infatti esiste come FILE ma **non era mai stato messo nel `.tex` del Report 7** (l'utente lo nota
  esplicitamente: "non abbiamo messo nel report 7 la matrice di attenzione plottata" — verificato,
  vero). Quindi §sec:battery del Report 8 **non ha richiesto nuovo calcolo**: solo riorganizzare i dati
  già validati del Report 7 in un'unica sezione, con la nuova figura mai mostrata aggiunta.
- **"con la nuova contribution"** (menzionato una volta, a proposito del leak-fraction): letto come
  l'intenzione finale una volta che la nuova misura (vedi sotto) esisterà, NON come richiesta per
  questa sessione — l'istruzione esplicita e finale dell'utente è **"mantieni sempre quella
  contribution iniziale, poi faremo la nuova"**. §sec:battery-contrib e il test a due cicli usano
  quindi entrambi la contribution ESISTENTE (`C_ik`, Jacobiano esatto, Report 7 §sec:setup), con una
  nota esplicita nel `.tex` che verranno rifatti quando la nuova misura sarà definita.
- **La nuova misura di contribution (A' invece di h^(0))**: l'utente ha detto esplicitamente che la
  spiega in **un messaggio successivo** — NON è ancora stata specificata. **Non inventare il design**:
  aspettare quel messaggio. Annotato come primo item aperto in §sec:outlook del `.tex` e qui sotto.
- **Plot di h^(1)/h^(2) direttamente (non solo quantità derivate), plot della ReLU, plot di
  attention\_out/h(l) per capire dove si combina l'informazione**: idee esplicitamente indicate come
  parte del "cosa mettere nel report 8" ma **nessun "implementa" esplicito** per queste — a differenza
  del test a due cicli, per cui l'utente ha scritto **"poi implementa l'esperimento di testing con due
  cicli"** come comando diretto. Trattate come **prossimo passo pianificato** (§sec:outlook nel `.tex`,
  vedi anche §15.4 qui sotto), non implementate in questa sessione per restare dentro l'unico comando
  di implementazione esplicito e il tempo limitato dichiarato dall'utente.
- **"come modello uso sempre il readout similarity... abbiamo visto che funziona molto meglio così"**:
  standardizzato da questa sessione in poi. Motivazione da scrivere nel report (fatto, §sec:setup del
  `.tex`): Report 7 §sec:res-similarity ha mostrato che il readout similarity rende il comportamento
  molto più robusto alle imperfezioni del trunk (reach non crolla mai sotto 0.91 a n40 vs il floor
  lineare 0.58–0.65; cut non scende mai sotto 0.995 vs il collasso lineare a 0.15/0.01–0.09 a seconda
  della distribuzione) pur mostrando lo stesso segnale meccanicistico graduale — quindi si fissa il
  readout a similarity per concentrarsi sul meccanismo, non sull'artefatto del readout lineare.
- **"partiamo con n40 union path, in caso poi capiamo per il resto in un secondo momento"**: scope di
  QUESTA sessione = solo `n40`, training `path_union` (disjoint-paths), readout similarity. ER/n64/n20
  per la batteria unificata e per il test a due cicli sono lasciati esplicitamente a dopo (annotato in
  §sec:outlook del `.tex`).
- **"prepara il report 8 adattandolo agli altri report"**: fatto, `report/8/transformer_for_graphs_8.tex`
  — stesso preambolo/stile/regole di scrittura dei Report 4–7 (§3, regole 20/21/55–60), compila pulito.
- **"prepara le istruzioni incollando esattamente il testo"**: fatto, è questa sezione (§15.0 sopra).
- **"implementa l'esperimento di testing con due cicli (mantieni sempre quella contribution iniziale,
  poi faremo la nuova)"**: fatto (codice + smoke-test), sbatch pronto **NON ancora lanciato** — vedi
  §15.3.

### 15.2 Titolo/idea chiave del report (dalla richiesta della prof)

Il Report 7 ha trovato *che* esiste un segnale di completamento e *quanto* dipende dal readout, ma non
ha mai chiesto **DOVE nel forward pass l'informazione viene combinata** — cioè: a che punto la rete ha
già deciso "sei vicino a questo nodo, quello è vicino al target, quindi sei connesso (o no)"? La
domanda guida del Report 8, presa **verbatim come titolo/idea chiave** su richiesta esplicita
dell'utente ("prendi questo come titolo e come idea chiave per questo report"): capire **a che punto le
informazioni sono già combinate** — se già dopo il layer 1, o solo dopo il layer 2, e se la
combinazione avviene nell'attention o nell'MLP. Titolo scelto per il `.tex`: *"Part VIII: Where Is
Connectivity Information Combined?"*.

### 15.3 Stato implementazione (questa sessione, 2026-07-21)

**Codice, FATTO e smoke-testato** (su un checkpoint giocattolo, readout linear E similarity, prima di
toccare checkpoint reali — stesso pattern del progetto):
- `data.py::generate_split_cycles_graph(n, short_len)` — analogo a due cicli di
  `generate_split_chains_graph`: stesso split `(a, n-a)`, ogni segmento chiuso a ciclo invece che path
  aperto (richiede `short_len>=3` e `n-short_len>=3`, un ciclo semplice ha bisogno di almeno 3 nodi).
- `mechanistic_asym_chains.py`: aggiunto `--topology {chain,cycle}` (default `chain`, **retrocompatibile
  — verificato che il comportamento di default non cambia**) che fa da dispatch fra i due generatori in
  tutte e quattro le funzioni che costruivano il grafo (`behavioural_sweep`, `readout_decomposition`,
  `readout_decomposition_similarity`, `attention_probe`); per `--topology cycle` lo split di default
  parte da `a=3` (non `a=1`, un ciclo non può avere 1-2 nodi).
- `mechanistic_heatmaps.py`: stesso `--topology {chain,cycle}` aggiunto a `heatmap_probe`.
- `plot_mechanistic_asym_chains.py` / `plot_mechanistic_heatmaps.py`: aggiunto `--report_root` (default
  `report7`, retrocompatibile) così gli stessi script di plot servono sia `runs/report7/...` sia
  `runs/report8/...`; i nomi dei file figura ora derivano da `--report_root` (`report8` → prefisso
  `r8_`) invece di avere `r7_` hardcoded.
- Tutti e quattro gli script **compilano e girano end-to-end** (smoke-test locale con un checkpoint
  giocattolo `RobertaGraphTransformer` linear E similarity, sia `--topology chain` sia `cycle`,
  incluso il test che il merge dell'`attn_cache.npz` esistente funziona — vedi errore 62).
- `scripts/r8_two_cycles.sbatch` (nuovo, CPU-only `medium_cpu`, `--array=0-3`, i 4 seed
  `n40_pathunion_similarity`): chiama `mechanistic_asym_chains.py --topology cycle` +
  `mechanistic_heatmaps.py --topology cycle --splits 4 20` sugli stessi checkpoint del Report 7
  (`runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed{S}/last.pt`), output
  `runs/report8/{mechanistic,heatmaps}/n40_pathunion_cycle_seed{S}/`. **Da lanciare**:
  `sbatch scripts/r8_two_cycles.sbatch`.
- **Deliberatamente NON incluso in questo giro** (fuori scope della richiesta esplicita): il
  falsification test a 3 componenti e il layer ablation per la topologia a cicli (richiederebbero un
  generatore "tre cicli" e non erano nella richiesta — solo il comportamento/sweep + la contribution
  esistente sul test a due cicli erano richiesti).

**Report, FATTO**: `report/8/transformer_for_graphs_8.tex`, compila pulito (**10 pagine**, 0 ref
indefinite, 0 overfull/underfull). Struttura: §1 recap I–VII + domanda guida; §2 setup (standardizza su
similarity, spiega perché); §3 `sec:battery` — la batteria unificata (sweep+logit con error bars,
geometria read-in, attention scores/alpha — NUOVA, mai mostrata prima —, contribution leak-fraction +
matrice, layer ablation, falsification test), **tutta scritta con dati reali già esistenti** (nessun
nuovo calcolo, riusa `runs/report7/report7_figs/*_n40_pathunion_similarity.png` via
`\includegraphics`/`\figorbox` — riferimento cross-report, pattern già in uso dal Report 4 che legge
dati dal Report 3); §4 `sec:cycles` — il test a due cicli, ipotesi dichiarate PRIMA dei dati (endpoint
incidentali vs load-bearing), **stub "data pending"** (job non ancora lanciato); §5 `sec:outlook` — le
tre idee non ancora implementate (nuova contribution su A', plot diretto di h^(1)/h^(2), plot
dell'attention output), più l'estensione a ER/n64/n20 lasciata a dopo.

**Da fare in una chat futura (in ordine)**:
1. `sbatch scripts/r8_two_cycles.sbatch`, poi pull, poi rigenerare le figure con
   `python plot_mechanistic_asym_chains.py --tag_glob "n40_pathunion_cycle_seed*" --n 40 --topology...`
   — **nota**: il tag_glob seleziona le cartelle per nome, non serve passare `--topology` al plot
   script (l'informazione è già nei dati salvati); usare
   `--report_root report8 --suffix _n40_pathunion_cycle --title_tag "disjoint-paths-trained, closed into cycles"`.
   Sostituire il paragrafo "[Data pending]" in `report/8` §sec:cycles con la tabella/figura vere e il
   verdetto (endpoint incidentali o load-bearing).
2. Quando l'utente manda il messaggio con il design della nuova contribution (basata su `A'`, non
   `h^(0)`): NON improvvisare prima di quel messaggio. Implementarla come nuova funzione (probabilmente
   in `mechanistic_asym_chains.py`, accanto a `exact_contribution`), poi rifare §sec:battery-contrib e
   il test a due cicli con la nuova misura.
3. Plot diretto di `h^{(1)}`/`h^{(2)}` (proiezione bassa dimensionalità colorata per componente) e
   dell'attention output per layer: nuovi script, non ancora iniziati.
4. Estendere §sec:battery e il test a due cicli a ER/n64/n20 (checkpoint già esistenti dal Report 7,
   nessun training nuovo).

### 15.4 La nuova contribution causale (edge-ablation), 2026-07-22 — design ricevuto, implementata

**Contesto**: l'utente ha mandato il design della nuova misura (quella annunciata al punto 2 di §15.3,
"lo dico in un nuovo messaggio") — un agente esterno l'ha elaborato in dettaglio. Richiesta esplicita:
implementarla per **entrambi** gli esperimenti in corso — la condizione attuale di §sec:battery (n40
path\_union, similarity, topologia chain) **e** dentro lo sbatch dei due cicli (§15.3), che non era
ancora stato lanciato quindi c'era tempo per modificarlo. **La vecchia contribution (Jacobiano,
`exact_contribution`) NON va eliminata** — va **rinominata/reinterpretata** come misura di *sensibilità
interna* (non causale), tenuta così com'è; la nuova è il complemento causale da usare per affermazioni
sul ruolo di un edge o della componente corta.

**Il design (riassunto, dettaglio completo nel docstring di `edge_ablation_contribution.py`):** invece
di derivare rispetto a `h_k^{(0)}` (già un embedding interno, costruito dall'intero vicinato di `k`, non
strettamente legato al solo nodo `k`), si interviene **direttamente su `A'`**: per ogni edge non
orientato `{u,v}` del grafo si azzera in entrambe le direzioni (self-loop sulla diagonale intatti — il
grafo/modello sono simmetrici, non si tocca una sola entry), si rifà il forward pass, e si misura per
ogni nodo `i` (1) il cambio nel suo embedding finale (norma L2), (2) il cambio medio assoluto su tutta
la sua riga di logit, (3) lo stesso ma ristretto alle coppie within-long a distanza `>9` (le coppie
beyond-capacity del puzzle Report VI/VII). Aggregazione per nodo `k`: **media** (non somma) sugli edge
incidenti a `k` — motivazione esplicita dell'utente: un estremo di path ha grado 1, un nodo interno
grado 2, sommare confonderebbe "più edge" con "più influenza". Risultato: tre matrici $40\times40$
$C^{\mathrm{edge}}, C^{\mathrm{logit}}, C^{\mathrm{far}}$, stessa forma/convenzione (componente corta
prima, poi lunga; ogni relabeling rimappato all'ordine base prima di aggregare) delle heatmap esistenti.
Più una **edge-leak fraction** analoga a quella vecchia (frazione di massa causale della componente
lunga attribuibile agli edge della componente corta), calcolabile sia su $C^{\mathrm{edge}}$ sia su
$C^{\mathrm{far}}$. **Caveat esplicito dell'utente, tenuto nel report in una frase sola**: su una
path/ciclo OGNI edge è un ponte — rimuoverlo cambia anche il vero target di connettività per quella
coppia, quindi la misura è l'effetto di una perturbazione strutturale reale, non un contributo
additivo indipendente. Un controllo con l'aggiunta di una *chord* (edge dentro la stessa componente, il
target vero non cambia) è indicato dall'utente come **naturale prossimo passo, non richiesto ora**.

**Implementato, smoke-testato** (checkpoint giocattolo, linear e similarity, chain e cycle, prima di
qualunque dato reale):
- `edge_ablation_contribution.py` (NUOVO): `edge_ablation_probe(...)` fa tutto il calcolo (nessun
  backward pass — solo forward, quindi molto più economico del Jacobiano esatto: un batch di
  `1+n_edges` forward pass per grafo, niente autograd); riusa `_build_split_graph` da
  `mechanistic_asym_chains.py` per il dispatch chain/cycle (stesso `--topology` delle altre due). Un
  self-test (`_selftest`) verifica: (a) la riga baseline del batch combacia con un forward pass a
  parte; (b) i self-loop sopravvivono a un'ablazione di edge; (c) l'insieme $F(i)$ (coppie
  beyond-capacity within-long) è confinato alla componente lunga. Output `edge_contrib.npz` per
  checkpoint (stesso pattern chiavi-appiattite `aXX__...` di `attn_cache.npz`).
- `plot_edge_ablation.py` (NUOVO, locale no-GPU): le tre heatmap $C^{\mathrm{edge}}/C^{\mathrm{logit}}/
  C^{\mathrm{far}}$ a due split rappresentativi (default 4 e 20, un seed rappresentativo — stesso motivo
  delle altre heatmap: sono pattern, non scalari, non si mediano fra seed con basi diverse) + la curva
  edge/logit/far-leak-fraction sull'intero sweep, **con error bars (std sui seed) — errore 61**,
  aggregata su tutti i seed che matchano `--tag_glob`. Supporta `--report_root` come gli altri plot
  script di questa sessione.
- `scripts/r8_two_cycles.sbatch` **modificato** (non era ancora stato lanciato, in tempo per
  modificarlo come richiesto): aggiunto un terzo step `edge_ablation_contribution.py --topology cycle`
  per ognuno dei 4 seed, output `runs/report8/edge_contrib/n40_pathunion_cycle_seed{S}/`.
- `scripts/r8_edge_contrib_n40_pathunion.sbatch` (NUOVO): la stessa misura per la condizione chain di
  §sec:battery (n40 path\_union, similarity) — **da lanciare separatamente**, non tocca gli altri due
  step già fatti per quella condizione (mechanistic/heatmaps già completi da §15.3).
- `report/8/transformer_for_graphs_8.tex` **aggiornato**: §sec:battery-contrib riformulata (ora dice
  esplicitamente "sensibilità interna, non ancora una prova causale", rimanda alla nuova sezione); nuova
  §sec:battery-edge (subito dopo) con il design spiegato e uno stub "data pending" (le predizioni da
  verificare: ad `a=4` gli edge della componente corta dovrebbero avere un effetto reale su
  $C^{\mathrm{far}}$ della componente lunga, ad `a=20` l'effetto dovrebbe restare vicino alla diagonale);
  §sec:cycles aggiornata per menzionare che userà anche la nuova misura, con la nota sul diametro dei
  cicli (metà di quello di una path della stessa lunghezza — la soglia beyond-capacity per $C^{far}$
  nei cicli serve un segmento lungo più lungo, ~19 nodi contro ~10 per una path); §sec:outlook
  riscritta: tolto il bullet "nuova contribution" (ora implementata, non più un'idea vaga), aggiunto il
  controllo chord come nuovo follow-up esplicito dell'utente. Compila pulito, **11 pagine**, 0 ref
  indefinite, 0 overfull.

**Da fare in una chat futura (in ordine, aggiorna/sostituisce il punto 2 di §15.3 che è ora FATTO)**:
1. `sbatch scripts/r8_edge_contrib_n40_pathunion.sbatch` (condizione chain, aggiorna §sec:battery-edge)
   e `sbatch scripts/r8_two_cycles.sbatch` (ora include anche l'edge-ablation per i cicli, aggiorna
   §sec:cycles) — **nessuno dei due è stato ancora lanciato**.
2. Dopo il pull: `python plot_edge_ablation.py --tag_glob "n40_pathunion_seed*" --report_root report8
   --title_tag "disjoint-paths-trained"` (chain) e con `--tag_glob "n40_pathunion_cycle_seed*" --suffix
   _n40_pathunion_cycle --title_tag "disjoint-paths-trained, closed into cycles"` (cycle); più
   `python plot_mechanistic_asym_chains.py`/`plot_mechanistic_heatmaps.py` per i cicli (punto 1 di
   §15.3, ancora valido). Sostituire i due paragrafi "[Data pending]" nel `.tex` con tabelle/figure vere
   e il verdetto.
3. Il controllo chord (§sec:outlook) e i punti 3–4 di §15.3 (plot h^(1)/h^(2), attention output,
   estensione a ER/n64/n20) restano aperti, non iniziati. **Aggiornamento (§15.5): il plot di
   h^(1)/h^(2) è ora coperto (in forma più ricca, con tutti gli stadi intermedi) — vedi sotto.**

### 15.5 Dove viene combinata l'informazione: probe stagewise (Priorità 1), 2026-07-22

**Contesto**: l'utente ha mandato una specifica tecnica enorme in 12 sezioni (probabilmente elaborata
da un agente esterno, come per l'edge-ablation di §15.4) per una nuova domanda del report: **a che
punto del forward pass l'informazione di connettività viene combinata** — già l'idea chiave del titolo
del Report 8 (§15.2). L'utente stesso ha dato un **ordine di priorità esplicito**: Priorità 1 (stati
intermedi, cosine heatmaps, intermediate read-out, margin per stadio, ΔZ per sotto-blocco) = "gli
esperimenti essenziali"; Priorità 2 (Q/K/V/αV/AttnOut norms, message-contribution matrix, causal
short-to-long attention masking); Priorità 3 (stagewise edge ablation, chord-addition control, PCA).
Dato il tempo limitato, **questa sessione implementa SOLO la Priorità 1**, seguendo l'ordine
esplicitamente richiesto — Priorità 2/3 restano da fare, elencate sotto come prossimi passi.

**Design (riassunto — dettaglio completo nel docstring di `stagewise_diagnostics.py` e nella nuova
sezione `sec:stagewise` del `.tex`, scritta con lo stesso livello di formule del Report 7/della
sezione edge-ablation):** un forward pass diagnostico (non tocca il comportamento standard del
modello) salva, per ciascuno dei due layer, non solo $H^{(1)}/H^{(2)}$ ma anche
$\mathrm{AttnOut}^{(\ell)}$, $H_{\mathrm{attn}}^{(\ell)}$ (dopo attention+residuo+LayerNorm, prima
dell'MLP) e $\mathrm{FFNOut}^{(\ell)}$ — 5 "stadi principali" in tutto:
$H^{(0)}\to H_{\mathrm{attn}}^{(1)}\to H^{(1)}\to H_{\mathrm{attn}}^{(2)}\to H^{(2)}$. Per ciascuno:
(1) la matrice coseno $G^X_{ij}=\cos(x_i,x_j)$ (esattamente ciò che il readout similarity usa); (2) un
probe "intermediate read-out" — applica scale/bias già allenati (solo per $H^{(2)}$) a $G^X$ come se il
modello si fermasse lì, e misura exact/reach short/reach long (near/far)/cut/positive-rate; (3) il
margine **threshold-free** $M_{\mathrm{far}}^X=\mu_{\text{long-far}}^X-\mu_{\text{cross}}^X$ (non
dipende da scale/bias allenati solo per l'ultimo stadio); (4) i quattro $\Delta Z$ per sotto-blocco
(differenze consecutive dei logit intermedi: attn1, mlp1, attn2, mlp2), aggregati per categoria
(within-short/within-long-near/within-long-far/cross). Ipotesi dichiarata PRIMA dei dati (non
assunta, come richiesto esplicitamente dall'utente — "non assumere in anticipo... riportare ciò che
emerge"): attention 1 raccoglie info locale, MLP 1 comincia a separare le componenti, attention 2 alza
la similarità delle coppie lontane, MLP 2 converte questo in decisione finale — una fra le letture
possibili, da verificare con $M_{\mathrm{far}}$ e le medie per categoria, non da assumere.

**Implementato, smoke-testato** (checkpoint giocattolo similarity, `n=20`, prima di qualunque dato
reale — **due bug di layout trovati e corretti durante lo smoke-test, non solo un test passivo**):
- `stagewise_diagnostics.py` (NUOVO): `run_with_stages(...)` (il forward diagnostico, analogo a
  `run_with_cache` ma con gli stadi intra-layer in più), un self-test che verifica $H^{(2)}$/logit
  contro `model.forward_and_embeddings`, e `stagewise_probe(...)` che calcola tutto quanto sopra per
  ogni split. Solo readout **similarity** (coerente con lo standard del Report 8, §15.1). Nessun
  backward pass — solo forward, economico come `edge_ablation_contribution.py`. Output
  `stagewise_geometry.npz` (matrici) + `stagewise_{metrics,margins,deltaz}.csv` (scalari).
- `plot_stagewise_diagnostics.py` (NUOVO, locale no-GPU): heatmap coseno 5 pannelli, heatmap ΔZ 4
  pannelli, curve far-reach/margin-vs-stadio (error bars su seed — errore 61), tabella categoria×branch
  per ΔZ. **Bug trovati e corretti durante lo smoke-test**: (1) un backslash-spazio superfluo in due
  titoli (`"vs.\\ stage"`) che matplotlib renderizzava letteralmente fuori da un blocco `$...$`; (2)
  `fig.colorbar(im, ax=axes, ...)` insieme a `fig.tight_layout()` con un `suptitle` a due righe causava
  sovrapposizione fra il suptitle e i titoli dei pannelli, e la colorbar finiva sopra l'ultimo pannello
  — corretto passando a un colorbar-axis dedicato (`fig.add_axes` dopo `tight_layout(rect=...)`)
  invece di lasciare che `colorbar` rubi spazio dalla lista di assi. Verificato visivamente prima e
  dopo il fix.
- `scripts/r8_stagewise_n40_pathunion.sbatch` (NUOVO, CPU-only `medium_cpu`, `--array=0-3`, i 4 seed
  `n40_pathunion_similarity` — stessa condizione di §sec:battery): **da lanciare**,
  `sbatch scripts/r8_stagewise_n40_pathunion.sbatch`. Time limit 2h (stesso ordine di grandezza
  dell'edge-ablation, nessun backward pass — dovrebbe bastare ampiamente).
- `report/8/transformer_for_graphs_8.tex`: nuova sezione `sec:stagewise` ("Where is the information
  combined? A stagewise probe") fra il test due-cicli e l'outlook, con tutte le formule (stessa
  precisione del Report 7), le ipotesi dichiarate prima dei dati, stub "data pending". Compila pulito,
  **15 pagine**, 0 ref indefinite, 0 overfull.

**Deliberatamente NON implementato in questa sessione** (Priorità 2/3, esplicitamente rimandate):
Q/K/V/αV/AttnOut norms lungo la path, message-contribution matrix $T^{(\ell)}_{ij}$, causal
short-to-long attention masking (condizioni A–E); stagewise edge-ablation (la contribution causale di
§15.4 estesa a tutti e 5 gli stadi, non solo $H^{(2)}$); chord-addition control; PCA/UMAP. Da fare in
questo ordine quando si riprende.

**Da fare in una chat futura (in ordine)**:
1. `sbatch scripts/r8_stagewise_n40_pathunion.sbatch` — non ancora lanciato.
2. Dopo il pull: `python plot_stagewise_diagnostics.py --tag_glob "n40_pathunion_seed*" --heatmap_seed
   1000 --title_tag "disjoint-paths-trained"`. Sostituire lo stub "[Data pending]" in `report/8`
   §sec:stagewise con le figure vere e il verdetto sull'ipotesi (quale stadio muove di più
   $M_{\mathrm{far}}$?).
3. Poi, in ordine: Priorità 2 (Q/K/V/message-contribution/causal masking), Priorità 3 (stagewise edge
   ablation, chord control, PCA) — nessuna delle due iniziata.
4. Restano aperti anche i punti già noti: il controllo chord di §15.4/outlook, l'estensione di
   §sec:battery/§sec:cycles/§sec:stagewise a ER/n64/n20.

### 15.6 Entrambi i job tornati, sezioni completate con dati reali — 2026-07-22 (stessa sessione)

**Tempi reali (utili per calibrare `--time` in futuro):** `r8_two_cycles.sbatch` (job 598932, rilanciato
su `compute` dopo il timeout a 5h su `medium_cpu` — vedi sopra) ha impiegato **5h33m–6h32m per seed**
(stessa scala di `mechanistic_asym_chains.py`/Jacobiano esatto già vista per il Report 7, §14.7/§14.10);
`r8_stagewise_n40_pathunion.sbatch` (job 599790) ha impiegato **~6–7 secondi per seed** — confermato
economico come previsto (nessun backward pass).

**§sec:cycles (test due-cicli) SCRITTA — risultato SORPRENDENTE, diverso da entrambe le letture
pre-registrate.** Non "risolve come le chain" né "degrada". Il modello, ad **ogni** split da `a=3` a
`a=20` e **ogni** seed, predice l'**intero grafo connesso** (predicted-positive rate = 1.000 esatto,
cut = 0.000 esatto, exact = 0.000 esatto — identico a 3 decimali su tutti gli split e seed). La lettura
meccanicistica si spacca in due: la vecchia contribution (Jacobiano) resta quasi perfettamente
block-diagonal (leak ≈0 da `a=7` in poi — separazione interna pulita), mentre la nuova edge-ablation
mostra un leak che sale fino a ~0.48 (più alto che per le chain) con matrici quasi uniformemente alte
su TUTTO il 40×40, senza taglio al confine — coerente con un readout che ha smesso di discriminare le
componenti. **Limite onesto aggiunto esplicitamente**: il training è SOLO su unions di path (mai un
ciclo), quindi chiudere in cerchio non isola "togliere gli estremi" — cambia anche il grado locale di
ogni nodo (tutti grado 2) ed è una struttura mai vista. Il collasso uniforme assomiglia di più
all'over-connessione OOD già documentata su canvas non familiari (Report VI Thread A a n64, Report VII
§res-n64) che a un test pulito sull'ipotesi degli estremi. **Prossimo passo pulito indicato nel testo
(non implementato)**: chiudere in cerchio SOLO una delle due componenti, lasciando l'altra aperta, per
separare "collasso locale alla forma non familiare" da "collasso globale alla vista di un ciclo
qualsiasi".

**§sec:stagewise (probe Priorità 1) SCRITTA — risultato chiaro, corregge l'ipotesi pre-registrata su
QUALE stadio fa il lavoro.** La cosine geometry mostra una banda locale stretta fino a $H^{(1)}$, poi
un allargamento drastico a un blocco denso su tutta la componente **fra $H^{(1)}$ e $H_{\mathrm{attn}}^{(2)}$**
— costruito dall'**attention layer 2**, non dall'MLP. Il far-reach (accuracy soglia-dipendente) conferma:
≤0.10 fino a $H^{(1)}$, salta a 0.72 ($a=4$)/0.99 ($a=20$) a $H_{\mathrm{attn}}^{(2)}$, rifinito da MLP2 a
1.00/0.98. **Nota onesta**: il margine threshold-free $M_{\mathrm{far}}$ CONCORDA con questa lettura ad
`a=20` (salta anch'esso a $H_{\mathrm{attn}}^{(2)}$) ma NON ad `a=4` (resta basso fino a MLP2) — le due
misure (soglia vs. margine medio) divergono in un caso, riportato onestamente invece di scegliere quella
più pulita. La scomposizione $\Delta Z$ per sotto-blocco aggiunge il pezzo mancante: attention 2 alza la
similarità in modo abbastanza indiscriminato (all'interno di qualunque componente un nodo si trovi), ma
è **MLP 2** il sotto-blocco che separa nettamente connesso/disconnesso (a `a=4`: within-long-far +11.2,
cross −16.3, il valore più estremo di tutta la tabella). **Verdetto sull'ipotesi pre-dichiarata**: parzialmente
sbagliata — "attention 2 alza la similarità delle coppie lontane nello specifico" è troppo stretto (la alza
in modo generico, non selettivo); "MLP 2 converte la geometria nella decisione finale" è vero ma
specificamente per il **cut**, non per il completamento (che attention 2 ha già quasi risolto).

**Report**: entrambe le sezioni ora scritte con dati reali (non più "[Data pending]"). Compila pulito,
**21 pagine**, 0 ref indefinite, 0 overfull (verificato anche visivamente pagina per pagina).

**File da pushare (in più rispetto a §15.4/§15.5):** `report/8/transformer_for_graphs_8.tex` +
`.pdf` aggiornati, tutte le nuove figure in `runs/report8/report8_figs/` (sweep/leak/heatmap per il
test due-cicli, cosine/ΔZ/curve per lo stagewise probe).

**Resta aperto (nessun cambiamento rispetto a §15.4/§15.5)**: Priorità 2/3 del filone stagewise, il
controllo chord dell'edge-ablation, l'estensione di tutte le sezioni a ER/n64/n20, e il follow-up
"chiudi solo una componente" suggerito sopra per il test due-cicli.

Dopo questo punto, due piccole revisioni tabellari a richiesta utente (nessun nuovo calcolo): Tabella 4
(`tab:r8ablation`) integrata con il blocco "reach (componente corta)" oltre a "reach (componente lunga)"
e "cut"; Tabella 5 (`tab:r8threeway`) integrata con le colonne `|L1|`/`|L2|` (dimensioni delle due
componenti grandi, prima mancanti — la tabella era illeggibile senza, vedi errore 67 in §4).

### 15.7 Handoff di chiusura — Report 8 sostanzialmente finito (2026-07-23)

**Stato a questo punto**: il Report 8 è considerato dall'utente **sostanzialmente concluso**. Tutte le
sezioni (§sec:battery, §sec:battery-edge, §sec:cycles, §sec:stagewise) sono scritte con dati reali, il
`.tex` compila pulito (21 pagine, 0 ref indefinite, 0 overfull), le due tabelle segnalate dall'utente
(Tabella 4 e Tabella 5) sono state corrette. **Non ci sono altri task noti in sospeso su questo report.**

**Cosa succede ora**: l'utente costruisce a mano un PowerPoint per la prof a partire dal contenuto del
Report 8 (stesso pattern già seguito per il Report 7, vedi cambio banner in cima al file e memoria
`project_report7_audience_change`). Questa chat non proseguirà il lavoro sul report a meno di richiesta
esplicita; per lo stesso motivo (chat troppo lunga) eventuali modifiche future potrebbero arrivare da
una **nuova chat Claude**, che farà l'onboarding leggendo questo file per intero + tutti i `.tex` dei
report (regola di lettura in cima al file) — non serve altro contesto per ripartire.

**Per la prossima chat, in sintesi**: leggere il banner in cima, poi questo §15 per intero (in
particolare §15.6 per i risultati finali), poi il `.tex` del Report 8. Le §13/§14 restano lo storico
completo di Report 6/7 se serve contesto più profondo su thread precedenti (es. l'origine del puzzle
Report 6 Thread B che il Report 7 ha risolto). Gli errori 1–67 in §4 vanno riletti per intero prima di
scrivere qualunque cosa, come da regola di lettura non negoziabile in cima al file.

**(SUPERATO da §15.8 sotto — la sessione del 2026-07-24 ha aggiunto altro contenuto reale al Report 8,
quindi questa non è più la chiusura finale. Lasciato qui come storico di come si è arrivati fin lì.)**

### 15.8 Sessione di rifinitura post-chiusura (2026-07-24) — Q&A + nuove figure meccanicistiche sui
due cicli. Handoff finale prima di aprire una NUOVA chat per il Report 9.

**Contesto**: dopo la chiusura di §15.7, l'utente ha riaperto la stessa chat (non ancora una nuova) per
una serie di domande di comprensione sul Report 8 già scritto, che hanno prodotto piccoli fix e — alla
fine — due nuove figure reali con un risultato nuovo. **Questa è la chiusura AGGIORNATA**: la prossima
chat sarà una chat NUOVA e lavorerà su un **Report 9** (argomento non ancora specificato dall'utente in
questa sessione — va chiesto/deciso all'apertura di quella chat, non assumerlo).

**Cosa è stato fatto, in ordine:**

1. **Tabella 5 (`tab:r8threeway`, il falsification test a 3 componenti) — aggiunta la colonna `exact`.**
   L'utente ha chiesto exact-match accuracy per riga; letta direttamente dal campo `exact` già presente
   in `runs/report7/three_way/n40_pathunion_similarity_seed{1000..4000}/three_way_split.json` (verificato
   PRIMA che le altre colonne — reach/cut — combaciassero esattamente coi valori già pubblicati, prova
   di essere sulla tabella giusta). Risultato: exact molto più basso a `small=1,2` (0.012, 0.380)
   nonostante reach/cut già ≥0.988 lì — spiegato in una frase nel corpo (statistica congiunta su 300+
   coppie, converge da `small=4` in poi). **Confermato via codice** (non solo dati): esistono SOLO
   queste 6 combinazioni `(small, L1, L2)` — `eval_three_way_split.py:115-125` ha un default hardcoded
   `(1,2,4,7,8,10)` e lo sbatch che ha lanciato questo run (`scripts/r7_similarity_n40.sbatch`) non passa
   `--small_lens`, quindi non esiste già da qualche parte uno sweep più fitto — andrebbe rilanciato con
   `--small_lens` esplicito se mai richiesto.
2. **Figura 9 (`fig:r8cyclesweep`, sweep due-cicli) — aggiunta la soglia di decisione coseno.** L'utente
   ha chiesto conferma che il readout similarity classifica "connesso" sse
   `cos(h_i,h_j) > -bias/scale` (confermato leggendo `model.py:112-145/314-347`: il logit è
   `scale·cos+bias`, soglia a logit`>0`). Poi ha chiesto di disegnare quella soglia sul grafico.
   **Fatto in codice**: nuova `load_cos_threshold(tag_glob)` in `plot_mechanistic_asym_chains.py` (legge
   `sim_scale`/`sim_bias` da `weights_summary.json`, media sui seed), `fig_sweep_and_logit()` ora accetta
   `cos_threshold` e disegna una linea tratteggiata nel pannello inferiore quando il readout è
   similarity; `main()` la calcola e la passa. Rigenerata `r8_sweep_and_logit_n40_pathunion_cycle.png` —
   la linea (≈0.146) mostra visivamente perché il cut è sempre 0.000: anche la curva cross-cycle (cut)
   sta sempre ben sopra la soglia (0.72–0.90). Caption aggiornata nel `.tex`.
3. **Chiarito nel testo cosa vuol dire "far" in `M_far` (§sec:stagewise).** L'utente ha notato che la
   definizione ("pairs beyond capacity, d>9, dentro la componente lunga" — la stessa `F(i)` della
   sezione edge-ablation) stava in un paragrafo precedente e non era ripetuta dove si introduce `M_far`.
   Aggiunto un richiamo inline esplicito nel paragrafo "A threshold-free margin".
4. **Chiarite due domande di comprensione, nessuna modifica di codice**: (a) il range teorico di `M_far`
   è `[-2,2]` (differenza di due coseni, ciascuno in `[-1,1]`), i valori osservati (0.03–0.9) restano
   ben dentro; (b) negli heatmap `C^{edge}`/`C^{logit}`/`C^{far}` (Figura `r8edgeheat`) l'asse x è `k`
   (nodo ablato, cresce andando a destra) e l'asse y è `i` (nodo query) ma con `origin='upper'` di
   matplotlib di default — quindi **andando verso l'alto `i` DECRESCE** (riga 0 in cima), l'opposto della
   convenzione cartesiana; nessun `origin=`/`extent=` esplicito nello script, quindi vale il default.
5. **Perché il readout similarity ha `scale`/`bias` e non il coseno nudo — spiegato, nessuna modifica.**
   Il coseno grezzo in `[-1,1]` passato a una sigmoide non potrebbe mai esprimere confidenza vicina a
   0/1 (serve per l'exact-match su centinaia di coppie); `scale` (impara fino a ≈32.6) allarga il range,
   `bias` libera la soglia di decisione da un vincolo arbitrario `cos=0`. Non è una scelta "meno
   corretta" del coseno puro, è lo standard (come il temperature scaling di ArcFace/CosFace/contrastive
   heads) — senza, la loss non potrebbe mai scendere vicino a zero.
6. **NUOVO script `plot_cosine_raw_examples.py`** (scritto, smoke-testato su checkpoint giocattolo
   similarity — sia il fix iniziale delle chiavi `model_config`/`model_state_dict` di `load_model`, sia
   il fix del layout colorbar+tight_layout, stesso pattern errore 65b — **MAI ancora lanciato su dati
   reali**, resta un task aperto). Scopo: matrice di cosine similarity GREZZA (non mediata su tante
   relabelling come le altre heatmap) per pochi grafi individuali a un dato split, colorata rispetto
   alla soglia di decisione (rosso sopra = predetto connesso, blu sotto = predetto disconnesso) così gli
   errori del modello si vedono direttamente confrontando col confine di componente. Uso preparato:
   `python plot_cosine_raw_examples.py --checkpoint /tmp/r8_ckpt/seed1000.pt --n 40 --split_a 20
   --n_examples 5 --out runs/report8/report8_figs/r8_cosine_raw_examples_a20.png` (checkpoint già
   scaricato in questa sessione, vedi punto 8).
7. **`stagewise_diagnostics.py` esteso con `--topology {chain,cycle}`** (stesso pattern già usato in
   `mechanistic_asym_chains.py`/`mechanistic_heatmaps.py`/`edge_ablation_contribution.py`): la funzione
   `stagewise_probe()` ora prende un parametro `topology` invece di avere `"chain"` hardcoded al posto
   di chiamare `_build_split_graph`. Smoke-testato su checkpoint giocattolo (chain retrocompatibile +
   cycle), poi **lanciato per davvero** su HPC→locale: checkpoint reale
   `runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed1000/last.pt` scaricato via `scp` in
   `/tmp/r8_ckpt/seed1000.pt` (VPN Bocconi, comando dato all'utente), poi
   `python stagewise_diagnostics.py --checkpoint /tmp/r8_ckpt/seed1000.pt --output_dir
   runs/report8/stagewise/n40_pathunion_cycle_seed1000 --topology cycle --splits 4 20 --n_graphs 64`
   girato **in locale sul Mac** (economico, nessun backward pass, pochi secondi — stessa scala già nota
   dal caso chain, §15.6). **Un solo seed (1000), non tutti e 4** — se in futuro serve estendere le
   curve margin/far-reach/deltaz aggregate con error bars (regola 61) per i cicli, servono anche i seed
   2000/3000/4000 (stesso comando, altro checkpoint).
8. **Nuove figure aggiunte al `.tex` in §sec:cycles** (dopo il paragrafo "Two mechanistic readings...",
   prima di "Honest limits"):
   - **Figura `fig:r8cycleattnscores`**: le attention scores/α reali layer 0/1 per `a=4,20` sui cicli —
     riusa un file **già calcolato** in una sessione precedente (`r8_two_cycles.sbatch` aveva già girato
     `mechanistic_heatmaps.py --topology cycle`) ma **mai mostrato nel `.tex`** (stesso pattern già visto
     per la chain condition in §sec:battery-attn, errore-tipo già noto). Nuovo paragrafo: layer 0
     identico al caso chain (banda locale + banda di wrap-around del ciclo); layer 1 mostra un pattern a
     scacchiera dovuto alla periodicità, e l'`α` reale resta quasi zero cross-ciclo **anche ad `a=20`**
     (l'attenzione grezza non attraversa il confine più che nel caso chain).
   - **Figure `fig:r8cyclestagecosine4`/`20`**: le due heatmap layerwise cosine geometry generate al
     punto 7 (dati REALI, seed 1000).
   - **Nuovo paragrafo con la scoperta chiave** (numeri letti da
     `runs/report8/stagewise/n40_pathunion_cycle_seed1000/stagewise_margins.csv`, non a memoria): ad
     `a=20`, allo stadio `H_attn^{(2)}` (dopo attention 2, PRIMA di MLP 2) il modello separerebbe quasi
     perfettamente i due cicli — `mu_long_far=0.790`, `mu_cross=-0.118` (**negativo**), margine
     `M_far=0.908`, dieci volte la soglia di decisione (0.146). **MLP 2 poi cancella quasi tutto**: a
     `H^{(2)}` `mu_cross` salta a `0.929` (quasi quanto `mu_long_far=0.964`), il margine crolla a
     `0.034`, producendo il collasso "tutto connesso". È l'**opposto esatto** di quanto MLP 2 fa sulle
     chain (dove *rifinisce* il cut, §sec:stagewise/§15.6) — qui lo **distrugge**. Ad `a=4` lo stesso
     ordine qualitativo ma molto più debole (margine 0.036 a `H_attn^{(2)}`, 0.133 a `H^{(2)}`).
   Ricompilato: **22 pagine** (era 21), 0 errori, 0 ref indefinite, 0 overfull — verificato anche
   visivamente pagina per pagina (`pdftoppm` + ispezione, non solo il log).

**File toccati/nuovi in questa sessione** (oltre a `istruzioni.md`): `report/8/transformer_for_graphs_8.
{tex,pdf}`; `plot_mechanistic_asym_chains.py` (soglia coseno); `stagewise_diagnostics.py` (`--topology`);
`plot_cosine_raw_examples.py` (NUOVO, mai lanciato su dati reali); `runs/report8/stagewise/
n40_pathunion_cycle_seed1000/*` (nuovo, dati reali, un solo seed); `runs/report8/report8_figs/
r8_sweep_and_logit_n40_pathunion_cycle.png` (rigenerata con soglia), `r8_stagewise_cosine_a{4,20}_cycle.
png` (nuove, nel report), `r8_stagewise_{deltaz_a4,deltaz_a20,far_reach,margin}_cycle.png` (nuove,
generate come sottoprodotto ma **non ancora messe nel `.tex`** — solo un seed, niente error bars).

**⚠️ Checkpoint ancora in `/tmp/r8_ckpt/seed1000.pt` — NON cancellato a fine sessione** (a differenza
della regola standard §13.7/§14.5: scarica in `/tmp`, usa, cancella subito). È rimasto lì perché
`plot_cosine_raw_examples.py` (punto 6) non è ancora stato lanciato e potrebbe servire ancora nella
stessa sessione. **Se una chat futura lo trova ancora lì e non sa perché**: è sicuro da cancellare
(`rm -rf /tmp/r8_ckpt`), è solo una copia locale di un checkpoint che vive comunque su HPC.

**Cosa resta aperto per il Report 8** (nessun cambiamento rispetto a §15.6, +2 nuovi):
- Il plot `cosine_raw_examples` (punto 6) non ancora generato su dati reali.
- Le curve margin/far-reach/deltaz aggregate per i cicli servono altri 3 seed per avere error bars.
- Tutto il resto invariato da §15.6: Priorità 2/3 del filone stagewise, il controllo chord
  dell'edge-ablation, l'estensione di ogni sezione a ER/n64/n20, il follow-up "chiudi solo una
  componente" per il test due-cicli.

**⚠️ Per la PROSSIMA chat (sarà una chat NUOVA — l'utente lo ha detto esplicitamente in questa
sessione): il lavoro sarà un Report 9, argomento NON ancora specificato.** Non assumere il topic — va
chiesto o atteso dall'utente all'apertura di quella chat. Onboarding standard: banner in cima al file +
questo intero §15 (specialmente questo §15.8) + il `.tex` del Report 8, come da regola di lettura in
cima al file. Se il Report 9 riprende uno dei fili "resta aperto" sopra, va scritto come `report/9/...`
nuovo (non innestato dentro l'8) seguendo lo stesso schema di apertura degli altri report (recap
Reports I–VIII in un elenco puntato, poi la nuova domanda).

---

## 16. Report 9 — Oltre il diametro: generalizzazione OOD su path, e il ruolo degli estremi

> Blocco aggiunto all'apertura del Report 9 (2026-07-24, stessa sessione che ha chiuso il Report 8 in
> §15.8). **Resta valido per OGNI nuova chat sul Report 9.** In questa sessione si scrive SOLO il piano
> (questo blocco + lo scheletro `report/9/transformer_for_graphs_9.tex`) — **nessun esperimento,
> nessun training, nessuno script nuovo**: l'utente ha chiesto esplicitamente "non creare ancora gli
> esperimenti. quello lo facciamo alla nuova domanda". Non implementare nulla di §16.3 finché l'utente
> non lo chiede esplicitamente, thread per thread.

### 16.0 Richiesta originale dell'utente (trascritta VERBATIM — leggere prima di tutto)

Come già fatto per i Report 7/8 (§14.0/§15.0), l'utente ha chiesto di incollare qui **parola per
parola, senza cambiare niente**, il testo ricevuto, così una chat futura può rileggerlo per intero
invece che affidarsi a un riassunto:

```
quello che vorremmo dire è che il diametro non è veramente la misura che dobbiamo sempre guardare perchè ad esempio c’è il caso delle path che può essere imparato oltre il 2*3^L. il problema è che questa cosa delle path è in distribution perchè il disjoint path ha anche ogni split di 2 segmenti nel training set. sarebbe interessante fare training set sempre path, però fare test su path che non sono in training. perchè al momento sembra che non riusciamo a generalizzare sui 2 cycles. 
primo esperimento (il piu importante) : cercare qualche esempio di path learning che sia out of distribution, perchè al momento riusciamo a dire solo che in distribution impara oltre il diametro di 2*9 perchè usa gli ednopoitns. 
ad esempio training set di union path, però includendoli solo fino ad un certo diametro tipo split di 2 chains senza andare oltre il diametro di 18 e vedere se generalizza oltre , cioè test set 2chains con diametro oltre 18

il training set cosi vario (disjoint path da 1 a 4) come lo abbiamo ora per lei sembra too much, forse non abbiamo bisiogno.  per me invece forse non riesce a capire questa cosa dell’utilizzo degli endpoints se non ha questa distribuzione di 1,2,3,4 chains, però è da testare sicurmaente per vedere cosa fa con meno tipologie di path. anche perchè se l’imparare sta cosa degli endpoints è specifica del training set, allora diventa meno interessante, si può comunque scrivere. 
un altro esperimento è che potremmo avere solo vari split di 2 chains, e poi faccio test sulle 2 chains togliendo un edge cioe diventano 3 chains non viste nel training (perchè al momento anche le tre chains le aveva gia viste per l’esperimento fatto ora). 

poi dobbiamo pensare ad altri esesmpi di training su path e test su path, capire come farli però. 
ad esempio potrei testare i chckpoints che già ho su 5/6/7 chains che sono OOD e vedere se è stato sufficiente aver imparato questa cosa degli endpoints


oppure vedo se i checkpoints trainati sui 2 chains (tra l’altro come al solito voglio 4 seeds) hanno imparato di nuovo sta cosa degli endpoints e in caso li testo su delle distribuioni particolari tipo 1 chain e 1 cliques, oppure 2 sorte di chains però i nodi all’interno delle chains (non gli endpoints) hanno anche degli edges tra di loro, in modo che rimangano 2 componenti che hanno degli estremi

secondo esperimento serve per verificare se questi estremi sono effettivamente necessari e quindi il nostro modello di trasnformer i cycles non li riesce proprio ad imparare oppure è solo una caratteristica specifica delle path
fare direttamente uguale il training che ho fatto sulle path, ma farlo anche per i cycles con i vari split per vedere cosa succede e come impara (test in disitrubtion). noi vogliamo studiare infatti settings in cui il modello impara e vogliamo capire perchè lo fa, il che è una cosa che non è stata tanto fatta.
quindi voglio un esperimento analogo a quello fatto per i path, ma fatto sui cycles. quindi prima di dire che gli endpoints sono importanti/che i cycles sono davvero piu difficili degli altri grafi oppure se anche loro possano essere un esempio che contraddice un po’ il diametro.
vedere anche qui un po’ gli attention score e i similiarity geometry per i vari step. insomma voglio che fai(leggi attentamente il pwp del report 8 (Empirical Analysis of Transformers on Graph Connectivity part 8.pdf)): general results, analisi del read in, attention scores, node to node contribution, layer ablation, 3 cycles test, 2 chains split test, layerwise cosine geometry 


mi ha detto di provare a fare training uguale a prima con le disjoint path e il test uguale (quindi diciamo il report 8) anche con n46 uguale ad adesso con similarity readout e lo stesso numero di samples e stesso identico modello ad adesso uguale, cambia solo n46 anzichè n40. riusa i codici. direttamente tutti i training che dovrai fare in questo nuovo report falli con n46. 
```

### 16.1 Da dove nasce la domanda

Il Report 8 (§15, in particolare §15.6/§15.8) ha stabilito che il modello path\_union-trained (readout
similarity, $n{=}40$) risolve uno split sbilanciato ben oltre il muro raddoppiato $2\cdot3^L=18$,
grazie a un segnale di completamento legato agli estremi (endpoints) di path — e che lo stesso modello
**collassa a "tutto connesso"** sul test a due cicli (§sec:cycles del Report 8), dove gli estremi non
esistono. Il problema, notato dalla prof: tutta questa evidenza di "generalizzazione oltre il muro" è
misurata **dentro** la distribuzione di training — `generate_path_union_graph` (§14.1) genera online
unioni di $1$–$4$ path che **coprono già ogni split a due segmenti possibile**, quindi un test a due
componenti non è mai davvero out-of-distribution nella sua *struttura*, solo eventualmente nella sua
*taglia esatta*. Il Report 9 nasce per separare due cose che finora sono rimaste intrecciate: (a) il
modello ha imparato un meccanismo generale legato agli estremi, che dovrebbe generalizzare a strutture
genuinamente mai viste; oppure (b) il modello ha semplicemente interpolato dentro una distribuzione di
training che già copre lo spazio dei test. Il secondo filone (cicli) chiede inoltre se il fallimento sui
cicli sia dovuto specificamente all'assenza di estremi, o se sia semplicemente che i cicli non sono mai
stati allenati nello stesso modo sistematico dei path.

### 16.2 Le due tesi centrali che il report vuole sostenere/verificare

1. **Il diametro non è sempre la misura giusta di difficoltà — ma va isolato un esempio pulito.** Il
   Report 8 suggerisce che un modello può imparare oltre $2\cdot3^L$, ma finché il test resta
   in-distribution questo non prova nulla di nuovo rispetto al muro. Serve un esperimento in cui il
   training è esplicitamente **limitato** in diametro/struttura e il test genuinamente lo supera.
2. **Gli estremi sono davvero necessari (load-bearing), o è una caratteristica del training sui path
   specificamente?** Se un training analogo sui cicli (che non hanno estremi) produce un fenomeno
   analogo di generalizzazione oltre il muro, allora gli estremi non sono la spiegazione causale, e i
   cicli diventano un secondo controesempio a "il diametro conta". Se invece i cicli restano
   inapprendibili anche con un training dedicato, l'ipotesi degli estremi si rafforza (ma resta da
   escludere che sia semplicemente un limite architetturale sui cicli indipendente dagli estremi).

### 16.3 Piano esperimenti — TUTTI da implementare in una sessione/richiesta successiva (NON ora)

#### Thread A — Generalizzazione OOD per i path (esperimento 1, il più importante)

- **A.1 — Training troncato in diametro, test oltre.** Allenare su path/union-di-path limitati a un
  diametro massimo esplicito (es. split di 2-chains senza mai superare diametro $18$), poi testare su
  2-chains con diametro **oltre** $18$, mai visto in training. Questo è il test decisivo per isolare la
  generalizzazione OOD dal completamento in-distribution.
- **A.2 — Meno varietà nel training set.** La distribuzione attuale (unione di $1$–$4$ path disgiunti)
  è, secondo la prof, "too much" — forse non necessaria. Allenare con **meno tipologie** (es. solo split
  di 2-chains, non $1$–$4$ componenti) e verificare se il modello impara comunque il completamento via
  estremi. Se sì: il fenomeno non dipende dalla ricchezza/varietà della distribuzione (più generale, più
  interessante). Se no: è specifico di quella distribuzione ricca (meno interessante, ma comunque un
  risultato da scrivere onestamente).
- **A.3 — 3-chains genuinamente OOD.** Allenare **solo** su vari split di 2-chains (mai $3+$
  componenti), poi testare togliendo un edge interno a una delle due componenti — che diventano così 3
  componenti **mai viste in training** (a differenza dell'esperimento già fatto finora, dove le 3-chains
  erano comunque già state viste in qualche forma dalla distribuzione $1$–$4$).
- **A.4 — Checkpoint esistenti su 5/6/7-chains OOD.** Usare i checkpoint già allenati (path\_union, che
  arriva solo a $4$ componenti) e testarli su unioni con $5,6,7$ componenti, mai viste in training, per
  vedere se il trucco degli estremi generalizza al *numero* di componenti oltre a quanto già coperto.
- **A.5 — Checkpoint 2-chains, test su distribuzioni mirate.** Allenare (4 seed, come sempre) su vari
  split di 2-chains puri, verificare che il modello impari di nuovo il segnale degli estremi, poi
  testarlo su strutture scelte per isolare cosa conta davvero: (i) un grafo con **1 chain + 1 clique**
  (due componenti con un solo estremo "vero" da un lato); (ii) due "chain" dove i nodi **interni** (non
  gli estremi) hanno anche archi fra loro, restando comunque 2 componenti con estremi ben definiti ma
  internamente più dense — per capire se serve la forma esatta di path o basta avere due componenti con
  estremi riconoscibili.

*(Nota: A.4/A.5 richiedono di identificare/riusare i checkpoint giusti già su HPC — path\_union $n{=}40$
di Report 6/7/8 per A.4; i nuovi checkpoint 2-chains-puri di A.5 vanno allenati a $n{=}46$, §16.4. Da
decidere in dettaglio quando si implementa il thread, non ora.)*

#### Thread B — Gli estremi sono davvero necessari? L'esperimento analogo sui cicli (esperimento 2)

- **B.1 — Stesso training dei path, ma sui cicli.** Ripetere esattamente lo stesso schema di training
  usato per i path (stesso spirito del Report 8: distribuzione esplicita, non un mixed opaco) ma con
  cicli al posto dei path, con vari split, e valutare **in-distribution** (analogo a come i path sono
  stati valutati) — capire se/come il modello impara a gestire i cicli quando li vede sistematicamente
  in training, invece di vederli solo come test OOD (come nel test a due-cicli del Report 8, dove il
  training non conteneva mai un ciclo).
- **B.2 — Ripetere l'intera batteria meccanicistica del Report 8 sui cicli-trained.** Rileggendo
  attentamente le slide del Report 8 (il PDF del PowerPoint), riprodurre sui checkpoint cicli-trained
  la stessa batteria: risultati comportamentali generali, analisi del read-in, attention scores, node-
  to-node contribution, layer ablation, il test di falsificazione a 3 cicli (analogo del three-way
  split test), il test di split a 2 cicli, e la layerwise cosine geometry (il probe stagewise).

Obiettivo di B: prima di concludere che gli estremi sono la causa del fallimento sui cicli (o che i
cicli sono intrinsecamente più difficili di altri grafi), verificare se un training dedicato e
sistematico (analogo a quello dei path) permette al modello di imparare un fenomeno simile sui cicli —
nel qual caso i cicli diventerebbero un **secondo controesempio** a "il diametro è l'unica misura di
difficoltà", non un limite architetturale legato specificamente all'assenza di estremi.

### 16.4 Nota tecnica non negoziabile: canvas size $n=46$

Riportata testualmente dalla richiesta (§16.0, ultimo paragrafo): **ogni training nuovo di questo
report va fatto a $n=46$** (non $n=40$) — stesso identico setup del Report 8 (training sulle
disjoint-paths, readout **similarity**, stesso numero di sample/step, stesso modello RoBERTa-faithful
$L{=}2$, single head, $d_{\mathrm{model}}{=}512$) — **cambia solo la dimensione del canvas**. Riusare il
codice di training/eval già esistente (già parametrico in `--n_nodes`/`--n`, vedi §5/§9/§14/§15), non
reimplementare nulla da zero. Questo vale per **tutti** i training del Report 9, inclusi quelli dei
Thread A e B sopra (A.1, A.2, A.3, A.5 — training nuovi — e B.1 — training nuovo sui cicli), non solo
per un'eventuale ripetizione 1:1 del Report 8. A.4 resta un'eccezione naturale: è eval-only su
checkpoint **già esistenti** a $n=40$ (path\_union di Report 6/7/8), quindi non si ri-allena a $n=46$.

### 16.5 Cosa NON fare ancora (stato della sessione di apertura, 2026-07-24)

Su richiesta esplicita dell'utente, la sessione di apertura del Report 9 (2026-07-24) ha prodotto
**solo** il piano (questo §16) e lo scheletro `report/9/transformer_for_graphs_9.tex`
(struttura/domande/paragrafi "planned", **nessun dato, nessuna tabella, nessuna figura reale**). Nessun
generatore dati nuovo, nessuno script di eval, nessun training, nessun `sbatch` in quella sessione.
**Superato in parte da §16.6 sotto**: la sessione successiva (2026-07-25) ha implementato codice per due
punti specifici del piano su richiesta diretta dell'utente. Tutto il resto del piano segue comunque lo
stesso pattern già rodato nei Report 5–8: un esperimento alla volta, codice smoke-testato su un
checkpoint/modello giocattolo prima di toccare pesi reali, dati reali pullati da HPC prima di scrivere
qualunque numero nel `.tex` (regola §4 errore 2), caption complete (regola 21), niente nomi di
file/codice nel testo renderizzato (regola 56), niente riferimenti a "advisor"/cronologia (regola 57),
error bars su ogni curva aggregata sui seed (regola 61), e verificare i punti vicini a ogni segnale
interessante in uno sweep discreto (regola 62).

### 16.6 Prima implementazione (2026-07-25): sanity-check $n=46$ + Thread A.4 (5/6/7-chain OOD)

**Richiesta dell'utente questa sessione**: partire con due esperimenti concreti prima del resto del
piano — (1) un sanity-check a due seed della condizione $n=46$ (per decidere se usarlo per tutto il
resto del report), (2) il test A.4 (checkpoint $n=40$ esistenti su unioni di 5/6/7 path). L'utente ha
anche chiesto esplicitamente se il test A.4 ha più senso su GPU o CPU — risposto e motivato sotto.

**(1) Sanity-check $n=46$ — due seed, due sbatch separati (come richiesto, non un array).** Training
IDENTICO alla condizione disjoint-paths di Report~VIII (famiglia \texttt{path\_union}, RoBERTa-faithful
$L{=}2$/single-head/$d_{\mathrm{model}}{=}512$, readout similarity, $10^6$ step, batch $1000$), unica
differenza $n{=}40\to46$. Riusato **senza modifiche** `experiments2/train_families_n20.py` (già
parametrico in `--n_nodes`), stessi iperparametri/tempo del run $n{=}40$ analogo di Report~VI ("Onda 1",
`gpunew`, 12h — qui alzato a 14h per margine, $n{=}46$ è solo leggermente più costoso). Dopo il training,
ogni sbatch lancia in automatico `eval_asym_chains.py` (esistente, non modificato) sul checkpoint appena
allenato — lo sweep split diretto che mostra/non mostra la firma di completamento (split piccolo risolto
ben oltre il muro, split bilanciato fermo al muro). Nuovi file:
`scripts/r9_n46_sanity_seed1000.sbatch`, `scripts/r9_n46_sanity_seed2000.sbatch` (due file separati,
job-name `r9n46s1`/`r9n46s2`). Output: checkpoint in
`runs/report9/n46_train/n46_path_union_roberta_similarity_lam0_seed{1000,2000}/`, eval in
`runs/report9/asym_chains_n46/n46_pathunion_seed{1000,2000}/asym_chains.json`.
**Da lanciare (l'utente):**
```
sbatch scripts/r9_n46_sanity_seed1000.sbatch
sbatch scripts/r9_n46_sanity_seed2000.sbatch
```
**STATO: preparato, NON lanciato.** Nessun dato ancora. Se il segnale riappare a $n=46$ come atteso, il
resto del piano (A.1, A.2, A.3, A.5, B.1) può procedere a $n=46$ senza ulteriori dubbi; se non riappare,
va discusso con l'utente prima di proseguire (potrebbe indicare che $n=46$ non è equivalente a $n=40$ per
qualche motivo non ovvio, es. feasibility degli split o tempo di convergenza).

**(2) Thread A.4 — nuovo generatore + nuovo eval, eval-only sui checkpoint $n=40$ già esistenti.**
- `data.py::generate_multi_path_split_graph(n, sizes)` (NUOVO): generalizza
  `generate_split_chains_graph`/`generate_three_way_split_graph` a un numero arbitrario $K=$
  `len(sizes)` di path disgiunti che partizionano tutti gli $n$ nodi, date le dimensioni ordinate dei
  componenti (`sizes`, che devono sommare a $n$). Nessuna modifica ai generatori esistenti.
- `eval_multiway_split.py` (NUOVO, eval-only): per $K\in\{5,6,7\}$ (il numero di path componenti — lo
  stream di training ne pesca solo $1$–$4$) costruisce **una componente lunga + $K{-}1$ corte**, con la
  dimensione delle componenti corte scelta in automatico (funzione `default_small_sizes`, sweep su
  poche taglie candidate `2..8`, filtrate) così che il diametro interno della componente lunga
  **superi $18$** (il muro raddoppiato) — override possibile via `--small_sizes`/`--dist_cutoff`.
  Metriche per cella (mai collassate): exact-match; reach nella componente lunga (aggregato +
  per-distanza, con i tre bucket espliciti $\le9$/$9$–$18$/$>18$ — quest'ultimo è quello decisivo);
  reach nelle componenti corte (aggregato, se hanno coppie interne); **cut lungo-corte** e **cut
  corta-corta** (aggregato su TUTTE le coppie di componenti corte diverse — la generalizzazione diretta
  della colonna decisiva `cut(L1,L2)` del test di falsificazione a tre vie, ora con più di due "altre"
  componenti mai viste insieme in training). Smoke-testato su un checkpoint giocattolo
  (`RobertaGraphTransformer`, similarity, $n{=}40$) prima di toccare pesi reali — verificato che per
  $n{=}40$ le taglie corte automaticamente feasibili sono $\{2,3,4,5\}$ ($K{=}5$), $\{2,3,4\}$
  ($K{=}6$), $\{2,3\}$ ($K{=}7$), tutte con diametro-lungo $>18$ per costruzione.
- **GPU o CPU? Risposta: CPU.** Questo è un eval **puramente forward** (nessun backward/Jacobiano, a
  differenza degli script meccanicistici di Report~VII/VIII) su un modello piccolo ($L{=}2$,
  $d_{\mathrm{model}}{=}512$) — stesso genere di `eval_asym_chains.py`/`eval_three_way_split.py`, che in
  passato hanno girato anche su GPU (`short_gpuh200`) quando non c'era altro in coda. Qui però ci sono
  **contemporaneamente** i due training $n{=}46$ sopra, che hanno bisogno di GPU e competono per il tetto
  di 4 GPU-concorrenti-per-utente (§6) — mettere anche questo eval su GPU sottrarrebbe uno slot ai
  training senza bisogno reale, dato che un forward-only su questo modello è già economico su CPU (pochi
  minuti per ~2700 forward pass totali, 9 celle × 300 grafi). **Scelta: `short_cpu`**, `--cpus-per-task=8`
  con BLAS multi-thread abilitato (pattern degli script CPU-only di Report VII, non quello a 1 thread
  degli script GPU). Nuovo file `scripts/r9_a4_multiway_n40.sbatch` (array di 4, un task per seed, sui
  checkpoint `runs/report6/a1_train/n40_path_union_roberta_similarity_lam0_seed{1000..4000}/last.pt`).
  Output: `runs/report9/multiway_split/n40_pathunion_seed{S}/multiway_split.json`.
  **Da lanciare (l'utente):**
  ```
  sbatch scripts/r9_a4_multiway_n40.sbatch
  ```
  **STATO: preparato, NON lanciato.** Nessun dato ancora.

**Report `.tex` aggiornato**: nuova §\ref{sec:planpre} ("Preliminary check") prima del Thread~A per il
sanity-check $n=46$; il paragrafo A.4 aggiornato per riflettere che il codice esiste (in attesa di
risultati); §Status aggiornata di conseguenza. Compila pulito, **5 pagine**, 0 errori, 0 ref indefinite,
0 overfull.

**File nuovi/toccati questa sessione**: `data.py` (nuovo generatore), `eval_multiway_split.py` (nuovo),
`scripts/r9_n46_sanity_seed{1000,2000}.sbatch` (nuovi), `scripts/r9_a4_multiway_n40.sbatch` (nuovo),
`report/9/transformer_for_graphs_9.{tex,pdf}`, `istruzioni.md`. Nessun checkpoint toccato/scaricato in
questa sessione (tutto smoke-testato su un checkpoint giocattolo locale, mai salvato nel repo).

**Da fare in una chat futura**: lanciare i tre sbatch sopra (l'utente), poi `git pull`, poi leggere i
json (`asym_chains.json` × 2 per il sanity-check, `multiway_split.json` × 4 per A.4) e scrivere le
sezioni corrispondenti nel `.tex` con i dati veri. Se il sanity-check conferma $n=46$, procedere con
A.1/A.2/A.3/A.5/B.1 (tutti ancora da disegnare in dettaglio/implementare) sempre a $n=46$.

### 16.7 Primo lancio reale (2026-07-25): 2/3 job falliti per la partizione `gpunew` sparita, fix fatto

**Cosa è successo.** L'utente ha lanciato i tre sbatch di §16.6. **`scripts/r9_a4_multiway_n40.sbatch`
(Thread A.4, CPU) è partito correttamente**, `Submitted batch job 604793` — nessuna azione necessaria,
in attesa che finisca. **I due training $n{=}46$ (`scripts/r9_n46_sanity_seed{1000,2000}.sbatch`) sono
FALLITI alla sottomissione stessa** (`sbatch: error: Batch job submission failed: User's group not
permitted to use this partition`, nessun job ID assegnato) — causa: la partizione `gpunew` **nuda**
usata da entrambi gli script non esiste più su questo cluster (errore 68/§6, scoperto proprio da questo
fallimento). **Fix fatto**: `--partition=gpunew` → `--partition=gpuh200` in entrambi gli script (stessa
identica riga, nessun altro cambiamento — stesso `--time=14:00:00`, ampiamente dentro il cap di 1 giorno
di `gpuh200`). Nessun altro file toccato.

**Da rilanciare (l'utente), dopo il pull della fix:**
```
sbatch scripts/r9_n46_sanity_seed1000.sbatch
sbatch scripts/r9_n46_sanity_seed2000.sbatch
```
Il job `604793` (Thread A.4) non va ri-lanciato, è già in coda/esecuzione con la configurazione giusta.

**Nota sulla manutenzione full-cluster del 27 luglio (§6):** questi due training richiedono fino a 14h;
se lanciati abbastanza in anticipo rispetto a lunedì 27/7 09:00 CEST finiranno prima della finestra di
manutenzione, altrimenti verranno cancellati a metà e andranno ri-sottomessi da capo dopo — controllare
la tempistica al momento del rilancio.

**File toccati questa sessione**: `scripts/r9_n46_sanity_seed{1000,2000}.sbatch` (fix partizione),
`istruzioni.md` (errore 68, aggiornamento §6, questo paragrafo).

### 16.8 Thread A.4 SCRITTO con dati veri (2026-07-25): reach generalizza, il "cut fra più altre
componenti" no — un nuovo asse di fallimento

**Dati pullati e verificati** (job `604793`, 4 seed, 9 celle ciascuno, tutti i `.out` puliti, nessuno
skip) — letti dai json grezzi, mai a memoria (regola §4 errore 2). Risultato scritto in
§sec:planA/paragrafo A.4 del `.tex`, con `tab:r9a4` (dati per-seed, mai collassati, regola 4/20/62).

**Esito, in breve:**
- **Il reach oltre-muro generalizza PERFETTAMENTE al numero di componenti mai visto in training.**
  Su tutte le $36$ combinazioni (cella × seed): reach nella componente lunga esattamente $1.000$ ---
  aggregato, nei tre bucket $\le9$/$9$–$18$/$>18$, e come block-exact. Stesso per le componenti
  corte. Zero eccezioni. Il training vede solo $1$–$4$ componenti totali; qui si arriva a $K=7$ senza
  che il reach vacilli mai.
- **Il fallimento è tutto nel *cut*, e si spacca in due pezzi con soglie diverse.** `cut(long,short)`
  (distinguere la lunga da UNA corta) resta $\ge0.994$ per 2 semi su 4 fino a $K=7$; gli altri 2 semi
  scivolano già a $K=5,s{=}5$ ($0.22$–$0.36$) e peggiorano. `cut(short,short)` (distinguere DUE
  componenti corte diverse fra loro) è molto più fragile: quasi intatto a $K=5$ ($0.54$–$1.00$) ma
  **collassa per OGNI seme entro $K=6$ o $K=7$** --- inclusi i due semi (1000, 3000) il cui
  `cut(long,short)` non fallisce mai: seed 1000 dà $0.000$ già a $K{=}6,s{=}2$ e a entrambe le celle
  $K{=}7$; seed 3000 dà $0.000$ a $K{=}6,s{=}2$. L'exact-match segue qualunque dei due cut collassi per
  primo.
- **Lettura**: il reach oltre-capacità compone bene con un numero arbitrario di componenti attorno; ma
  tenere DISTINTE fra loro più componenti "altre" simultanee non è la stessa skill di tenere distinta
  UNA sola componente grande da tutto il resto (il three-way falsification test di Report VII testava
  solo quest'ultima, a $K=3$). È un asse di generalizzazione genuinamente diverso da ogni risultato
  basato sulla distanza in questo progetto: degrada nel **numero di componenti simultanee**, non nel
  diametro di una singola componente, e fallisce anche per checkpoint che non mostrano nessun'altra
  debolezza.

**Fix di layout**: la tabella (7 colonne, numeri "a/b/c" per cella) era overfull di 9.7pt in `\small`
di default — risolto con `\setlength{\tabcolsep}{4pt}` sulla stessa tabella (nessun altro cambiamento).
Verificato anche visivamente (`pdftoppm` + ispezione, non solo il log) che la tabella stia dentro la
pagina. Abstract e §Status aggiornati per riflettere che A.4 ha dati veri (non più "[Data pending]").
Compila pulito, **7 pagine**, 0 errori, 0 ref indefinite, 0 overfull.

**Cosa resta aperto**: il sanity-check $n=46$ (§16.6/§16.7, rilanciato dall'utente dopo il fix di
partizione, ancora nessun dato tornato) e tutto il resto del piano (A.1, A.2, A.3, A.5, B.1, B.2) —
invariato rispetto a §16.6.

**File toccati questa sessione**: `report/9/transformer_for_graphs_9.{tex,pdf}` (A.4 scritta con dati
veri, abstract/status aggiornati), `istruzioni.md`. Nessun file di codice toccato (solo lettura dei
json già pullati).

---

### 16.9 Battery meccanicistica anche per il sanity-check n=46 (2026-07-25), gateata sul training

**Richiesta dell'utente**: i due job di training $n{=}46$ (§16.6/§16.7, `604831` seed 1000 in
esecuzione, `604832` seed 2000 in coda) fanno solo il training + lo sweep comportamentale base
(`eval_asym_chains.py`, già incluso). L'utente vuole ANCHE, sugli stessi due checkpoint risultanti:
la Tabella 1/Figura 1 del Report 8 (sweep + logit), la Figura 3 (attention scores/alpha), e la
layerwise cosine geometry (probe stagewise) — cioè la stessa identica batteria meccanicistica già
validata su $n=40$ nei Report VII/VIII, ora su $n=46$.

**Non si ri-lancia il training** (i due job già in coda/esecuzione restano intatti — cancellarli e
rifarli sprecherebbe i 40+ minuti già spesi da `604831`; inoltre lo script sbatch di un job già
sottomesso è comunque congelato da Slurm al momento della sottomissione, modificarlo ora non
cambierebbe nulla per quei due job). Invece: **nuovo sbatch eval-only, gateato sui due job di
training con `--dependency=afterany`** (non `afterok`, errore 44), che parte automaticamente non
appena entrambi i training finiscono (comunque vada).

**Nuovo file `scripts/r9_n46_mechbattery.sbatch`** (CPU-only, `medium_cpu`, array di 2 = i due
seed, `--time=02:00:00`): per ogni seed, sugli stessi tre script già usati/validati in Report
VII/VIII, **nessuna modifica di codice**, solo argomenti CLI diversi:
1. `mechanistic_asym_chains.py --attn_splits 4 23` (il pair rappresentativo "4, n//2" per $n=46$,
   l'analogo diretto di "4, 20" a $n=40$) → `metrics.csv`+`readout.csv` (Tabella 1/Figura 1). Lo
   sweep comportamentale pieno resta sul range di default (`1..n//2`=`1..23`) — è puro forward,
   economico indipendentemente dal numero di split; solo l'`attn_splits` (che innesca il Jacobiano
   della exact-contribution) è ristretto per restare economico.
2. `mechanistic_heatmaps.py --splits 4 23` (attention scores/alpha reali, Figura 3) — lasciati i
   default economici dello script (`contrib_n_graphs=8`, `n_graphs=80`), non i 64 usati per le
   figure finali del Report 8 (qui è un sanity-check, non un numero da pubblicare).
3. `stagewise_diagnostics.py --splits 4 23 --n_graphs 64` (layerwise cosine geometry) — nessun
   backward pass, economico a prescindere.
Output: `runs/report9/{mechanistic,heatmaps,stagewise}/n46_pathunion_seed{1000,2000}/`.

**Corretto su richiesta dell'utente: gate PER SEED, non su entrambi insieme.** L'eval di ogni seed
parte non appena FINISCE IL SUO PROPRIO training, non deve aspettare l'altro. Stesso script array
(`SEEDS=(1000 2000)`, indice 0/1), sottomesso DUE VOLTE, ogni volta con `--array` forzato a un solo
indice (un flag da riga di comando sovrascrive il default `#SBATCH --array=0-1` dello script) e la
propria dipendenza:
```
sbatch --dependency=afterany:604831 --array=0 scripts/r9_n46_mechbattery.sbatch   # seed 1000
sbatch --dependency=afterany:604832 --array=1 scripts/r9_n46_mechbattery.sbatch   # seed 2000
```
(job ID da riverificare con `squeue -u $(whoami)` prima di lanciare, se sono cambiati).
**STATO: preparato, NON lanciato.** Nessun dato ancora (né dal training n=46 né da questa batteria).

**Nota per dopo (plotting locale)**: `plot_mechanistic_asym_chains.py` e
`plot_mechanistic_heatmaps.py` supportano già `--report_root` (quindi `--report_root report9`
funziona senza modifiche). `plot_stagewise_diagnostics.py` invece ha `ROOT`/`OUT` **hardcoded** su
`runs/report8/...` — se si vuole rigenerare quella figura per il Report 9 servirà o un piccolo
edit locale (stesso pattern `--report_root` degli altri due) o puntare temporaneamente lo script
altrove; non ancora fatto, non blocca il lancio su HPC (che scrive comunque i dati grezzi in
`runs/report9/stagewise/...`, indipendenti dallo script di plot).

**File nuovi questa sessione**: `scripts/r9_n46_mechbattery.sbatch`, `istruzioni.md`.

---

### 16.10 Grande giro di implementazione (2026-07-25): rescale layerwise geometry + battery
K-way (5/6/7) + tre nuovi training a un seed (2-chains, 1+2-path, 2-cycles)

**Richiesta dell'utente, cinque pezzi in un colpo solo** (trascritta/riassunta, il verbatim
completo resta nel messaggio utente di questa sessione):
1. Ricolorare TUTTE le figure "layerwise similarity geometry" (in ogni report) con
   $Z=\mathrm{scale}\cdot\cos+\mathrm{bias}$ invece del coseno grezzo, così il confine
   connesso/disconnesso è sempre esattamente a $0$ (col coseno grezzo il confine è a
   $\cos\approx0.146$, i colori non si distinguono bene).
2. Sui 4 checkpoint $n{=}40$ path\_union-similarity già esistenti: la stessa batteria del
   Report~8 (Tabella~1, Figura~1, attention scores, layerwise similarity geometry) ma
   testata su unioni di **5, 6, 7** path (il Thread~A.4 del piano, ora con la batteria
   meccanicistica completa, non solo il comportamento base già scritto in §16.8). **CPU.**
3. Un nuovo training $n{=}46$, **1 solo seed**, su **soli 2-chains a split casuale**
   (readout similarity, stesso setup solito), testato in-distribuzione con tutta la solita
   batteria (tabella, Figura~1, attention, layerwise geometry) come in
   `scratch_n46_seed1000.pdf`.
4. Un nuovo training $n{=}46$, **1 solo seed**, su una miscela di **1-path (una sola catena
   da 46 nodi) e 2-path (split casuale)**, stessa batteria.
5. Un ultimo training, **1 solo seed**, su **2-cicli a split diversi** (stesso training
   "base" ma chiudendo le catene a cerchio), stessa batteria — per vedere se gli estremi
   sono davvero necessari o se anche i cicli, allenati per bene, imparano qualcosa di
   simile.

Tutto con le stesse caratteristiche di sempre (RoBERTa $L{=}2$/single-head/$d_{512}$,
readout similarity, $10^6$ step online, batch $1000$).

**(1) Rescale — FATTO, codice cambiato, dati esistenti da rigenerare.**
`stagewise_diagnostics.py` ora salva `scale`/`bias` (due scalari, letti direttamente da
`model.sim_scale`/`model.sim_bias`) dentro `stagewise_geometry.npz` — nessun'altra modifica
alla logica esistente (che già calcolava `Z=scale*G+bias` per le metriche, solo non lo
salvava per l'heatmap). `plot_stagewise_diagnostics.py::fig_cosine_heatmaps` ora carica
`scale`/`bias` dal npz e plotta $Z$ invece di $G$ (fallback automatico al coseno grezzo con
un warning se un npz vecchio non ha ancora `scale`/`bias`); titolo/colorbar aggiornati di
conseguenza. **Aggiunto anche `--report_root` a `plot_stagewise_diagnostics.py`** (era
hardcoded su `report8`, serviva per leggere i dati `report9`), stesso pattern già usato in
`plot_mechanistic_asym_chains.py`/`plot_mechanistic_heatmaps.py`, retrocompatibile (default
`report8` invariato). Smoke-testato su un checkpoint giocattolo (verificato visivamente: la
colorbar passa da $[-1,1]$ a $[-\mathrm{scale}{+}\mathrm{bias},\,\mathrm{scale}{+}\mathrm{bias}]$
esattamente come atteso).
**I dati ESISTENTI però non hanno ancora `scale`/`bias`** (le run precedenti salvavano solo
`G`): serve un RE-RUN, cosa economica dato che `stagewise_diagnostics.py` non fa backward
pass (~6-7s/seed, già misurato in Report~8). Nuovo sbatch
`scripts/r9_stagewise_rescale_refresh.sbatch` (CPU, `short_cpu`, array di 6): rigenera
`stagewise_geometry.npz` per i 4 seed chain $n{=}40$ (Report~8), il seed cycle $n{=}40$
(Report~8), e il seed $n{=}46$ (Report~9) — stessi identici checkpoint/split/topology delle
run originali, quindi metrics/margins/deltaz CSV escono identici, cambia solo l'npz.
**Da lanciare (l'utente):**
```
sbatch scripts/r9_stagewise_rescale_refresh.sbatch
```
**Dopo il pull**: rigenerare le figure con `plot_stagewise_diagnostics.py` per ciascuna
condizione (report8 chain: `--tag_glob "n40_pathunion_seed*"`, default `--report_root
report8`; report8 cycle: `--tag_glob "n40_pathunion_cycle_seed*" --suffix _cycle`; report9:
`--tag_glob "n46_pathunion_seed*" --report_root report9`) e, per il Report~8, SOSTITUIRE le
figure `fig:r8stagecosine4/20` e `fig:r8cyclestagecosine4/20` già nel `.tex` (stesso
contenuto concettuale, solo l'asse colore cambia — nessuna modifica di testo necessaria a
meno che i numeri descritti in prosa facciano riferimento a valori di coseno specifici, da
controllare quando si rigenera).

**(2) Thread A.4, battery meccanicistica completa — codice pronto, smoke-testato, NON
lanciato.** Tre file nuovi, tutti eval-only, che riusano SENZA MODIFICARLI i building block
generici già validati (nessun rischio per i Report~7/8 già finiti):
- `mechanistic_kway.py` (NUOVO): behavioural sweep + logit coseno grezzo per celle
  $(K,\text{small\_size})$ — l'analogo diretto di Tabella~1/Figura~1 del Report~8, ma con
  **una lunga + $(K{-}1)$ corte compattate (pool) in un unico indice "corto"** invece del
  singolo split a due vie. Riusa `_device`/`_selftest`/`weights_geometry_similarity` da
  `mechanistic_asym_chains.py` (import, zero modifiche) e `default_small_sizes`/
  `generate_multi_path_split_graph` da `eval_multiway_split.py`/`data.py` (Thread~A.4
  comportamentale di §16.8). Celle derivate in automatico: $K{=}5{:}\{2,3,4,5\}$,
  $K{=}6{:}\{2,3,4\}$, $K{=}7{:}\{2,3\}$ (stessa feasibility di §16.8).
- `mechanistic_kway_heatmaps.py` (NUOVO): attention scores/$\alpha$/Q/K/V reali (Figura~3
  equivalente) per le stesse celle — riusa `run_with_cache`/`exact_contribution`
  (`mechanistic_asym_chains.py`) e `raw_weights` (`mechanistic_heatmaps.py`), import puro.
- `stagewise_kway.py` (NUOVO): layerwise cosine geometry per le stesse celle — riusa
  `run_with_stages`/`_cosine_batch`/`MAIN_STAGES`/`SUBBLOCKS`/`_selftest` da
  `stagewise_diagnostics.py` (import puro; **include già il fix (1)**, salva
  `scale`/`bias` dal primo lancio, nessun refresh necessario in futuro per questi dati).
  ⚠️ **Bug trovato e fissato durante lo smoke-test**: la prima stesura importava per errore
  `_selftest` da `mechanistic_asym_chains.py` (valida `run_with_cache`, che questo script
  non usa) invece che da `stagewise_diagnostics.py` (valida `run_with_stages`, quello
  davvero usato) — corretto prima di girare su qualunque checkpoint reale.
- `plot_mechanistic_kway.py` (NUOVO, locale no-GPU): tabella aggregata per (K, small\_size)
  su console (mean±std sui seed) + tre figure — sweep/logit sfaccettato per $K$ (un pannello
  per $K=5,6,7$, asse-x = taglia della componente corta), attention scores $\alpha$ (celle
  scelte via `--attn_cells`), layerwise cosine geometry (una cella scelta via
  `--cosine_cell`, con il fix (1) già incorporato). Tutti e tre gli script smoke-testati
  end-to-end su un checkpoint giocattolo con 2 "seed" finti (aggregazione verificata) prima
  di toccare pesi reali; nessun artefatto di test lasciato nel repo (girato in una working
  dir separata sotto `/tmp`).
- **Nuovo sbatch `scripts/r9_kway_n40_battery.sbatch`** (CPU-only, `medium_cpu`, array di 4
  seed, le 9 celle sopra passate esplicitamente a heatmaps/stagewise): **CPU per scelta
  esplicita dell'utente** — comunque coerente con la logica già usata per il Thread~A.4
  comportamentale (§16.6/§16.9): tutto qui è forward-only o backward economico a
  `contrib_n_graphs=8` di default, nessun bisogno di GPU.
  **Da lanciare (l'utente):**
  ```
  sbatch scripts/r9_kway_n40_battery.sbatch
  ```
  **STATO: preparato, NON lanciato.** Output atteso in
  `runs/report9/{mechanistic_kway,heatmaps_kway,stagewise_kway}/n40_pathunion_seed{S}[_kway]/`.

**(3)(4)(5) Tre nuovi training a un seed, $n=46$, readout similarity — codice pronto, NON
lanciato.**
- `experiments2/train_families_n20.py` **esteso con due nuove famiglie nominate**
  (principio dati §13.2, mai un mixed opaco): `split_chains` (uno split casuale a due vie
  ogni campione, `short_len` uniforme $1..n{-}1$, via `data.py::generate_split_chains_graph`
  — NUOVO rispetto alla `2chains` già esistente, che è sempre lo split bilanciato fisso
  $n/2$) e `split_cycles` (stesso split casuale ma chiudendo ogni segmento a ciclo,
  `short_len` uniforme $3..n{-}3$, via `generate_split_cycles_graph`). Lista famiglie note
  estesa di conseguenza. **Fix collaterale** (scoperto smoke-testando in locale):
  `DataLoader(prefetch_factor=..., persistent_workers=True)` va in errore con
  `--num_workers 0` (errore 37, mai durevolmente risolto in questo file) — ora
  `prefetch_factor`/`persistent_workers` sono passati solo se `--num_workers>0`,
  retrocompatibile (default 16 su HPC, invariato).
- Smoke-testato in locale (CPU, `--train_steps 8 --num_workers 0`): `split_chains`,
  `split_cycles`, e la lista esplicita `1chain,split_chains` (tag risultante
  `1chain+split_chains`) girano tutti senza errori.
- **`scripts/r9_n46_splitchains_seed1000.sbatch`** (NUOVO, seed $3$): train
  `--families split_chains --n_nodes 46 --readout similarity` (stesso $10^6$ step/batch
  $1000$ di sempre), poi in automatico `eval_asym_chains.py` +
  `mechanistic_asym_chains.py --attn_splits 4 23` + `mechanistic_heatmaps.py --splits 4 23`
  + `stagewise_diagnostics.py --splits 4 23` (nessuno script nuovo necessario: `split_chains`
  produce comunque uno split a due vie, quindi la batteria chain esistente si applica senza
  modifiche).
- **`scripts/r9_n46_onetwopath_seed1000.sbatch`** (NUOVO): train
  `--families 1chain,split_chains ...`, stessa batteria di eval del punto sopra.
- **`scripts/r9_n46_splitcycles_seed1000.sbatch`** (NUOVO): train `--families split_cycles
  ...`, poi `mechanistic_asym_chains.py`/`mechanistic_heatmaps.py`/`stagewise_diagnostics.py`
  tutti con `--topology cycle` (già supportato, Report~8) — **`eval_asym_chains.py` è
  saltato apposta** (costruito solo per catene aperte; lo sweep di
  `mechanistic_asym_chains.py` copre comunque la stessa tabella per i cicli).
  ⚠️ **Assunzione da confermare con l'utente**: la richiesta non specificava $n$ per questo
  training ("uguale al training base... chiudendo le chain") — assunto $n{=}46$ per
  coerenza col resto della sessione; se l'utente intendeva $n{=}40$ (il canvas dei
  Report~6/7/8 originali) va corretto.
- Tutti e tre gli script sono un solo job (training GPU + eval nello stesso allocation, non
  gateato separatamente): dato che è un singolo seed esplorativo per ciascuno, non vale la
  complessità di due job dipendenti come nel sanity-check a due semi (§16.6/§16.7).
  Partizione `gpuh200` (NON `gpunew`, sparita — errore 68), `--time=16:00:00` (margine sopra
  le 14h di training + l'eval).
  **Da lanciare (l'utente):**
  ```
  sbatch scripts/r9_n46_splitchains_seed1000.sbatch
  sbatch scripts/r9_n46_onetwopath_seed1000.sbatch
  sbatch scripts/r9_n46_splitcycles_seed1000.sbatch
  ```
  **STATO: preparati, NON lanciati.**

**Cosa NON è stato ancora fatto** (nessuna scrittura nel `.tex` del Report~9 o in un nuovo
scratch pdf): tutti e cinque i pezzi sopra restano dati-in-arrivo. Una volta tornati i
risultati, seguire lo stesso pattern già rodato con `scratch_n46_seed1000.pdf` (§messaggi
precedenti di questa sessione) — uno scratch `.tex`/`.pdf` per condizione (o uno unico con
più sezioni), MAI scritto direttamente nel Report~9 finché i dati non sono confermati.

**File nuovi/toccati questa sessione**: `stagewise_diagnostics.py`,
`plot_stagewise_diagnostics.py`, `experiments2/train_families_n20.py` (modificati);
`mechanistic_kway.py`, `mechanistic_kway_heatmaps.py`, `stagewise_kway.py`,
`plot_mechanistic_kway.py` (nuovi); `scripts/r9_stagewise_rescale_refresh.sbatch`,
`scripts/r9_kway_n40_battery.sbatch`, `scripts/r9_n46_splitchains_seed1000.sbatch`,
`scripts/r9_n46_onetwopath_seed1000.sbatch`, `scripts/r9_n46_splitcycles_seed1000.sbatch`
(nuovi sbatch); `istruzioni.md`. Nessun checkpoint toccato (tutto smoke-testato su
checkpoint giocattolo locali, mai salvati nel repo).

---

*Per aggiungere questo file a git (lo fa l'utente):*
`git add istruzioni.md && git commit -m "Update project handoff instructions" && git push origin main`
