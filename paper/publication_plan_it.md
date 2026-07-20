# Piano di pubblicazione per Brain Informatics

Data di congelamento iniziale: 2026-07-20

## 1. Decisione editoriale

Rivista target: *Brain Informatics*, articolo di tipo **Research**.

Titolo principale:

> **Do Frozen Neuroimaging Foundation Representations Add Diagnostic and
> Prognostic Value in Alzheimer Disease? A Longitudinal ADNI Audit**

Il titolo proposto e' efficace perche formula una domanda verificabile e include
sia l'utilita diagnostica sia quella prognostica. Il sottotitolo rende esplicito
il disegno longitudinale e riduce il rischio che il manoscritto venga letto come
una proposta di nuova architettura foundation.

Il fit piu naturale e' con:

- Track 3: Brain Big Data Analytics, Curation and Management;
- Track 4: Informatics Paradigms for Brain and Mental Health Research.

Il contributo deve essere presentato come **brain-informatics evaluation
framework**: una valutazione riproducibile, longitudinalmente consapevole e
clinicamente ancorata del valore incrementale di rappresentazioni MRI foundation
congelate. Non deve essere presentato come semplice applicazione di Neuro-JEPA a
un nuovo dataset e neppure come world model.

Nessun piano puo garantire l'accettazione. Per "pubblicazione sicura" si intende
qui una submission scientificamente robusta, trasparente e difficile da
respingere per leakage, baseline deboli, selezione post-hoc o claim eccessivi.

## 2. Domanda, ipotesi e claim

### Domanda primaria

Le rappresentazioni estratte da MRI T1w mediante Neuro-JEPA congelato aggiungono
valore diagnostico e prognostico rispetto a morfometria MRI, demografia e stato
clinico convenzionali nella malattia di Alzheimer?

### Ipotesi primaria

L'aggiunta di Neuro-JEPA a feature convenzionali riduce l'errore nella stima
same-visit della severita cognitiva come parte di una matrice core completa,
senza privilegiare un singolo endpoint.

### Ipotesi secondarie

- Il valore incrementale e' osservabile in alcuni endpoint a 12 mesi e nella
  previsione dello stato ippocampale futuro.
- Il valore diminuisce agli orizzonti piu lunghi.
- Neuro-JEPA non sostituisce lo stato clinico e morfometrico esplicito nella
  conversione MCI-to-AD e nel survival.

### Claim massimo consentito

> Frozen Neuro-JEPA representations provide complementary, endpoint-dependent
> information beyond conventional Alzheimer disease variables, with their most
> reproducible value expected for concurrent cognitive severity and selected
> short-horizon outcomes.

Non sono consentiti claim di superiorita generale, utilita clinica dimostrata,
causalita, external generalization, patient simulation o world-model utility.

## 3. Perche il lavoro e' originale

La novita non e' una nuova rete. E' la combinazione dei seguenti elementi:

1. valutazione incrementale oltre baseline MRI e cliniche separate;
2. copertura coerente di stato diagnostico, severita, previsione cognitiva,
   anatomia futura, conversione e time-to-event;
3. nested cross-validation esclusivamente subject-level, ripetuta tre volte;
4. confronto paired con bootstrap a livello partecipante;
5. gerarchia prespecificata di endpoint e controllo della molteplicita;
6. pubblicazione dei risultati positivi, nulli e negativi nello stesso audit;
7. distinzione esplicita tra informazione rappresentazionale statica e valore
   prognostico longitudinale.

L'Introduction e la cover letter devono spiegare che classificare AD-vs-CN non
risponde alla domanda piu importante: se una rappresentazione foundation aggiunga
informazione oltre i marker gia disponibili e se tale informazione persista nei
task longitudinali.

## 4. Evidenza gia disponibile

I valori seguenti sono **storici/esplorativi** e servono per disegnare il paper,
non per compilare le tabelle finali.

- AD-vs-CN: pooled Neuro-JEPA + covariate AUROC 0.953 contro 0.942.
- MCI conversion 36 mesi: covariate 0.819; pooled + covariate 0.809.
- First-MCI survival: pooled + covariate C-index 0.719 contro 0.695.
- Same-visit cognition, differenza paired di MAE combinato meno convenzionale:
  MMSE -0.282, ADAS11 -0.720, ADAS13 -0.859; gli intervalli esplorativi non
  attraversano zero.
- Cognizione a 12 mesi: piccoli miglioramenti per MMSE e ADAS11; ADAS13 incerto.
- Future hippocampus: piccolo miglioramento paired; ventricoli e cognizione di
  Task F non mostrano vantaggio consistente.
- Task G--J non isolano un vantaggio affidabile del rollout predetto sullo stato
  corrente.

Questa configurazione sostiene un audit con eterogeneita degli effetti. Non
sostiene un paper sulla superiorita uniforme di Neuro-JEPA.

## 5. Cosa manca prima dei run finali

### 5.1 Feature group canonici

Definire una sola volta, per ciascun task:

- `morphometry_only`: hippocampus/ICV, amygdala/ICV, lateral e inferior lateral
  ventricles/ICV, cerebral cortex/ICV; nessuna covariata di acquisizione;
- `clinical_demographic`: eta, sesso, educazione, APOE4, diagnosi corrente e stato
  clinico ammesso dal task;
- `conventional_full`: unione dei due gruppi precedenti;
- `neurojepa_only`: pooled embedding `[768]`;
- `neurojepa_conventional`: pooled embedding + `conventional_full`;
- analisi MRI-specifica: `neurojepa_morphometry` contro `morphometry_only`.

Produrre un dizionario dati con nome, unita, sorgente ADNI, timing rispetto
all'anchor, gestione missing e task nei quali la variabile e' ammessa. Il target
same-visit non deve mai essere tra gli input. Nei task futuri il valore corrente
del target e' ammesso ma deve essere dichiarato e confrontato con carry-forward.

### 5.2 Reporter statistico unico

Il reporter deve leggere prediction per-riga, verificare l'allineamento dei
soggetti e produrre:

- metrica e CI 95% di ogni modello;
- differenza paired `combined - conventional`;
- bootstrap subject-level a 10.000 repliche;
- MAE subject-equal quando un soggetto contribuisce piu righe;
- AUROC, AUPRC e metriche di calibrazione per classificazione;
- C-index e, se supportato, Brier score time-dependent per survival;
- `n_rows`, `n_subjects`, positivi/eventi e censurati;
- correzione Benjamini-Hochberg per gli endpoint secondari;
- output CSV/JSON machine-readable e tabella pronta per il paper.

### 5.3 Nested cross-validation bloccata

- I seed storici `20260623` e `20260718` sono ritirati.
- Tre nuovi seed deterministici definiscono 3 repeat × 4 outer fold; ogni outer
  development usa 5 inner fold per il tuning.

Codice, feature, griglia, endpoint e regole sono congelati prima delle predizioni
outer-test. Ogni participant e' outer-test una volta per repeat. I repeat misurano
la sensibilita allo split ma non sono coorti indipendenti.

### 5.4 Cohort flow e provenance

Versionare per repeat e outer fold:

- conteggi iniziali, esclusioni e motivi;
- sessioni con MRI ed embedding validi;
- train/validation/test per ogni task;
- classi, eventi, censura, follow-up mediano e IQR;
- hash di manifest, pair table, checkpoint e configurazioni;
- commit del codice, ambiente Python e dipendenze;
- modello/checkpoint Neuro-JEPA e preprocessing esatto.

## 6. Pacchetto sperimentale congelato

### Matrice core

Tutti gli endpoint diagnostici-state e prognostici prespecificati hanno un'unica
metrica core. Il confronto comune e' `neurojepa_conventional` contro
`conventional_full`; gli undici confronti condividono una famiglia BH globale.

### Endpoint secondari gerarchici

1. Same-visit MMSE e ADAS11.
2. ADAS11 e MMSE a 12 mesi, predicendo il cambiamento e confrontando
   carry-forward.
3. Hippocampus/ICV futuro horizon-matched a 12 mesi; 24/36 mesi sono sensitivity.
   Le coppie a 6 mesi sono escluse dal protocollo confermatorio.

### Endpoint di supporto e boundary conditions

- AD-vs-CN: sanity check diagnostico, non risultato longitudinale principale.
- ADAS13 a 12 mesi e ventricles/ICV futuro: secondari.
- MCI-to-AD entro 36 mesi: AUROC, AUPRC e calibrazione.
- First-MCI time-to-AD: C-index, eventi e follow-up.
- Cognizione a 24/36 mesi: sensitivity analysis.
- Task E: supplementare meccanicistico opzionale.

Task G--J, Stage 1, LeWM e generazione MRI non entrano negli esperimenti principali.
Possono essere citati solo come motivazione interna per la distinzione tra encoding
statico e dinamica predetta; risultati non pubblicati non vanno presentati come
evidenza esterna.

## 7. Analisi necessarie per una submission robusta

Obbligatorie:

- verifica automatica che nessun participant attraversi gli split;
- confronto sulla stessa coorte test matched per ciascuna coppia di modelli;
- CI paired subject-level sugli out-of-fold predictions e dispersione fold/repeat;
- calibrazione per MCI conversion (Brier score, calibration intercept/slope o
  curve calibrate) senza scegliere soglie sul test;
- analisi missingness ed esclusioni per endpoint;
- sensitivity one-row-per-subject o subject-equal;
- nessuna selezione o sostituzione di fold/repeat in base ai risultati;
- confronto futuro con carry-forward e, per anatomia, current-state persistence;
- audit di scanner/site almeno descrittivo per split.

Fortemente raccomandate prima della prima submission:

- confronto con almeno un secondo encoder MRI foundation o self-supervised, usando
  lo stesso protocollo congelato;
- external validation su una coorte accessibile, anche limitata agli endpoint
  diagnostici/same-visit compatibili;
- analisi per ADNI phase/site/scanner come stress test di generalizzazione;
- interpretabilita limitata e prespecificata, per esempio associazione delle
  dimensioni/feature con marker anatomici, evitando mappe post-hoc decorative.

Tra queste, il secondo encoder e' la difesa piu diretta dall'obiezione "single
model audit"; una coorte esterna e' la difesa piu forte dall'obiezione
"ADNI-only". Almeno una delle due estensioni renderebbe sensibilmente piu sicura
la submission a una rivista con Impact Factor elevato.

## 8. Struttura del manoscritto Brain Informatics

### Abstract, 150--250 parole

Background; Methods; Results; Conclusions. Inserire numeri solo dopo il freeze
dei due report finali. Evitare "clinically meaningful" se non supportato da una
soglia clinica o decision-curve analysis.

### Introduction

1. Foundation models per neuroimaging e limiti dei benchmark statici.
2. Esigenza Alzheimer-specifica: valore incrementale oltre morfometria/clinica.
3. Gap: mancano audit longitudinali multi-endpoint con baseline matched e
   incertezza paired.
4. Domanda, ipotesi e contributi.

### Related work

Integrare senza creare una sezione enciclopedica:

- neuroimaging foundation/self-supervised representations;
- Alzheimer MRI diagnosis/prognosis e marker convenzionali;
- evaluation methodology, leakage e generalization in medical AI;
- longitudinal cognition, conversion e survival.

La ricerca bibliografica deve essere aggiornata fino alla data di submission e
basata su fonti primarie. Serve una tabella di related work che confronti modello,
coorte, frozen/fine-tuned, baseline convenzionale, endpoint longitudinali,
external validation e uncertainty.

#### Framing bibliografico da mantenere

La narrativa non deve sostenere che i foundation model non siano mai stati usati
in Alzheimer o su ADNI. Questa affermazione sarebbe gia' falsa. Il framing
difendibile e':

1. il pretraining self-supervised ha prodotto rappresentazioni trasferibili in
   immagini naturali e video;
2. in medical imaging i primi risultati piu' visibili hanno riguardato la
   segmentazione, ma i foundation model MRI recenti coprono ormai anche diagnosi,
   regressione, prognosi e survival;
3. in Alzheimer esistono gia' modelli MRI task-specific, benchmark longitudinali,
   foundation encoder valutati su ADNI e predizione della cognizione futura;
4. resta poco studiato il **valore incrementale** di una rappresentazione
   foundation congelata oltre un modello convenzionale completo, interpretabile e
   clinicamente disponibile;
5. resta inoltre incerto se tale valore sia coerente tra severita concorrente,
   cognizione e anatomia futura, conversione e survival.

Il vantaggio pratico dei modelli convenzionali va formulato con precisione. Una
volta derivate le variabili, Ridge/logistic/Cox richiedono poco calcolo downstream,
sono semplici da validare e mantengono significato clinico/anatomico diretto. Non
affermare un minore impatto ambientale senza misurare energia e intensita' di
carbonio: riportare invece hardware, wall time, memoria GPU e parametri trainabili,
e parlare di minore requisito computazionale downstream.

#### Perche' JEPA e Neuro-JEPA

JEPA va presentato come una delle principali strategie contemporanee di
representation learning predittivo, non come "il SOTA" universale. I-JEPA ha
mostrato la predizione non generativa di rappresentazioni semantiche da immagini;
V-JEPA 2 ha esteso il paradigma a comprensione, anticipazione e planning video con
risultati state-of-the-art su specifici benchmark. Neuro-JEPA rende ora possibile
un test MRI realistico grazie al checkpoint pre-addestrato su circa 1,55 milioni di
scan e a risultati competitivi su benchmark neuroimaging eterogenei.

La motivazione corretta e': se una rappresentazione neuroimaging moderna,
large-scale e basata su un paradigma predittivo forte non aggiunge informazione
oltre morfometria e stato clinico, il risultato negativo e' informativo per la
valutazione e l'adozione dei foundation model. Se aggiunge informazione solo in
alcuni endpoint, l'eterogeneita' identifica il loro ruolo realistico.

#### Mappa iniziale delle evidenze vicine

| Lavoro | Sovrapposizione | Gap rispetto al presente audit |
|---|---|---|
| I-JEPA (CVPR 2023); V-JEPA 2 (2025) | latent predictive pretraining su immagini/video | non neuroimaging Alzheimer |
| SAM-Med3D | foundation model volumetrico medicale | segmentazione, non valore incrementale AD |
| BrainIAC; BrainDINO; AnatCL | foundation representation brain MRI, probing/fine-tuning, task ADNI in alcuni casi | prevalentemente confronto tra encoder; non audit incrementale multi-endpoint |
| Neuro-JEPA (2026) | ADNI AD, amyloid e MCI-to-AD time-to-event; confronto con altri encoder e CNN scratch | soprattutto fine-tuning e performance image-only; nessun confronto completo oltre morfometria/clinica |
| Ma et al., *Nature Aging* 2026 | diagnosi, cognizione corrente e futura da singola MRI; CNN scratch, MedicalNet ed external test | esclude cognizione/diagnosi corrente e biomarker regionali dagli input; domanda non incrementale |
| Demir et al., CVPRW 2026 | frozen foundation features, linear probe e progressione ADNI | usa piu' MRI storiche; niente baseline clinica completa, cognizione continua o survival |
| TADPOLE | previsione longitudinale di diagnosi, ADAS13 e ventricoli | non valuta frozen raw-MRI foundation representation |
| Modello clinico/MRI *Scientific Reports* 2026 | valore incrementale, calibrazione e declino a 12 mesi | nessuna rappresentazione foundation da MRI raw |

Fonti primarie iniziali da includere e verificare nella bibliografia finale:

- Assran et al., I-JEPA, CVPR 2023;
- Assran et al., V-JEPA 2, 2025;
- Wang et al., SAM-Med3D;
- Tak et al., BrainIAC, *Nature Neuroscience* 2026;
- Wu et al., BrainDINO, 2026;
- Barbano et al., AnatCL, *Pattern Recognition Letters* 2026;
- Huang et al., Neuro-JEPA, 2026;
- Ma et al., *Nature Aging* 2026;
- Demir et al., CVPRW 2026;
- Marinescu et al., TADPOLE;
- Lannelongue et al., Green Algorithms.

Claim di gap consigliato:

> Recent studies show that pretrained brain MRI representations can support
> Alzheimer disease classification and progression prediction. However, it
> remains unclear whether a frozen foundation representation provides incremental
> information beyond explicitly measured morphometric and clinical state, whether
> such value persists across longitudinal endpoints, and whether observed effects
> reproduce across participant-level data splits.

### Materials and methods

1. ADNI cohort, ethics/data-use acknowledgement e data sources.
2. MRI selection, preprocessing e frozen Neuro-JEPA extraction.
3. Feature groups e prevenzione del leakage.
4. Task e temporal anchoring.
5. Linear probes e validation-only selection.
6. Statistical analysis, multiplicity and locked internal testing.
7. Reproducibility and software availability.

### Results

1. Cohort flow e caratteristiche.
2. Diagnostic state e same-visit cognition.
3. Short-horizon cognitive prognosis.
4. Future anatomical state.
5. MCI conversion e survival come boundary conditions.
6. Replication consistency e sensitivity analyses.

### Discussion

Aprire con la risposta alla domanda del titolo. Interpretare l'eterogeneita come
risultato, non come fallimento. Separare informazione complementare, utilita
predittiva e utilita clinica. Discutere ridondanza con morfometria, attenuazione a
lungo termine, ADNI-only, singolo foundation model, endpoint correlati e rischio
di endpoint selection informata dai risultati storici.

### Declarations richieste

- Availability of data and materials.
- Competing interests.
- Funding.
- Authors' contributions.
- Acknowledgements.
- Authors' information, opzionale.

La data availability deve spiegare che MRI e tabelle cliniche ADNI non possono
essere redistribuite, che l'accesso si richiede ad ADNI e che codice, config,
metriche aggregate e artefatti non identificativi saranno archiviati con DOI.

## 9. Tabelle, figure e supplemento

Main paper:

- Figure 1: cohort e pipeline informatica, dalla T1w ai confronti paired.
- Table 1: caratteristiche di coorte e flow per partizione/task.
- Table 2: task, anchor, input consentiti, target, orizzonte, modello e metrica.
- Figure 2: forest plot degli effetti incrementali nel test bloccato.
- Figure 3: valore prognostico per orizzonte, incluso carry-forward.
- Table 3: conversione, calibrazione e survival.

Supplemento:

- data dictionary e criteri di inclusione completi;
- risultati del test bloccato, storia di tuning e tutti i CI;
- griglie iperparametri e scelte validation;
- ROC/PR/calibration curves;
- sensitivity per diagnosi, scanner/site e repeated visits;
- Task E opzionale;
- checklist di reporting TRIPOD+AI/CLAIM adattata allo studio.

Non inserire tabelle come immagini. Figure: PDF vettoriale quando possibile,
larghezza finale 85 o 170 mm, circa 300 dpi per raster, font incorporati e file
individuali sotto 10 MB secondo le istruzioni correnti della rivista.

## 10. Sorgenti canoniche del repository

### Cohort e split

- `data/neurostar_adni_t1w_manifest.csv`
- `data/neurostar_adni_subject_splits.csv`
- `data/neurostar_adni_subject_splits_summary.json`
- `data/neurostar_adni_t1w_progression_pairs.csv`
- `data/neurostar_adni_t1w_state_transition_pairs_horizon_matched.csv`
- `scripts/data/build_neurostar_adni_manifest.py`
- `scripts/data/build_neurostar_subject_splits.py`
- `scripts/data/build_neurostar_state_transition_pairs.py`

### Neuro-JEPA

- `scripts/features/extract_neurojepa_embeddings.py`
- `data/neurostar_adni_t1w_neurojepa_embeddings.csv`
- `data/neurostar_adni_t1w_neurojepa_embeddings_summary.json`
- `docs/02_neurojepa_embeddings_pipeline.md`
- `external/Neuro-JEPA/README.md`
- `external/Neuro-JEPA/notebooks/extract_feature_example.ipynb`

### Probe e task

- A/B: `src/neurostar/neurojepa/probes/downstream.py`
- C: `src/neurostar/neurojepa/survival/probe.py`
- D1/D2: `src/neurostar/neurojepa/probes/cognitive.py`
- E: `src/neurostar/neurojepa/transitions/state.py`
- F: `src/neurostar/neurojepa/transitions/future.py`

### Evidenza storica, solo per audit e contestualizzazione

- `results/tasks_a_to_j_results_report.md`
- `worklog/2026-07-17_original_neurojepa_publication_audit.md`
- `reports/neurojepa_original_features_publication_completion_plan_it.md`
- `reports/task_f_task_j_experimental_report_it.md`

### Output finali autorizzati per Results

```text
results/paper_neurojepa_feature_audit/
  protocol/
  outer_repeat_01_seed_1658731809/
  outer_repeat_02_seed_1872921030/
  outer_repeat_03_seed_621129424/
  paired_report/
```

## 11. Fasi e gate decisionali

### Gate 0: protocol freeze

Passa quando feature, endpoint, split, metriche, bootstrap, correzione multipla e
go/no-go sono versionati prima di guardare i nuovi test.

### Gate 1: integrity

Passa quando leakage test, matched cohorts, provenance, missingness e cohort flow
sono completi. Un solo participant condiviso tra split blocca il paper.

### Gate 2: evidenza core

Un endpoint supporta un claim incrementale solo se effetto, CI paired e q-value
della famiglia globale sono favorevoli, riportando anche la dispersione tra
fold/repeat. Non esiste un criterio binario di successo globale.

titolo/abstract devono cambiare. Non si aggiungono modelli post-hoc per salvarlo.
Se ADAS13 non soddisfa il criterio, il manoscritto diventa un audit negativo o
misto. Non si aggiungono modelli, endpoint o seed post-hoc per salvarlo.
titolo/abstract devono cambiare. Non si aggiungono modelli post-hoc per salvarlo.

### Gate 3: reviewer resistance

Passa quando il manoscritto risponde preventivamente a: novelty, ADNI leakage,
baseline fairness, endpoint multiplicity, calibration, repeated measures,
single-model scope, lack of external validation e data/code availability.

### Gate 4: submission readiness

Passa quando tutti i numeri sono generati automaticamente, testo e supplemento
sono coerenti, figure rispettano le specifiche, dichiarazioni/autori sono chiusi,
il codice e' archiviato con DOI e la cover letter e' pronta.

## 12. Strategia se i risultati cambiano

- Effetti positivi e coerenti: mantenere il titolo interrogativo e concludere con
  valore complementare endpoint-dipendente.
- Same-visit positivo, prognosi nulla: sostituire "prognostic value" con
  "longitudinal utility" oppure dichiarare chiaramente assenza di valore
  prognostico incrementale.
- Tutti gli effetti incrementali nulli: riposizionare come rigorous negative
  benchmark; valutare *Brain Informatics* solo con forte contributo metodologico e
  secondo encoder.
- Eterogeneita tra fold/repeat: riportarla integralmente senza scegliere lo split
  migliore; interpretare il vantaggio come instabile quando direzione o ampiezza
  cambiano sostanzialmente.

## 13. Definizione di completamento

Il progetto e' pronto per submission quando tutti i gate sono passati, il TODO e'
chiuso, nessun numero finale proviene da directory storiche, e ogni claim in
abstract/discussion e' tracciabile a una tabella o figura versionata.
