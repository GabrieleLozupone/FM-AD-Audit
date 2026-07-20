# TODO verso la submission

Stati: `[ ]` da fare, `[~]` in corso, `[x]` completato, `[!]` bloccante.

## Milestone 0 - Protocollo

- [x] Scegliere *Brain Informatics* come target primario.
- [x] Definire titolo, domanda, claim massimo e posizionamento editoriale.
- [x] Separare questo paper dalla bozza Stage 1 in `report/`.
- [~] Protocollo scientifico e machine-readable completo; manca implementazione
  dei gate e firma interna prima del freeze.
- [x] Definire il data dictionary dei feature group per ogni task.
- [x] Definire endpoint primario, secondari, sensitivity e FDR families.
- [x] Definire griglie di alpha/penalizzazione e criteri validation.
- [x] Ritirare i seed storici e registrare tre nuovi seed derivati
  deterministicamente per la nested CV ripetuta.

## Milestone 1 - Implementazione minima

- [x] Estendere i probe con `morphometry_only`.
- [x] Estendere i probe con `clinical_demographic`.
- [x] Rendere uniforme `conventional_full` tra task compatibili.
- [x] Aggiungere `neurojepa_morphometry` e `neurojepa_conventional`.
- [x] Aggiungere controlli automatici anti-target leakage.
- [x] Implementare reporter paired subject-bootstrap riutilizzabile.
- [x] Implementare CI, FDR, conteggi e metriche di calibrazione.
- [x] Aggiungere test unitari per split, pairing, segno delle differenze e bootstrap.
- [x] Documentare comandi, input e output in `README.md` e `RUNBOOK.md`.
- [x] Implementare 3 ripetizioni di nested CV participant-level 4×5.
- [x] Generare manifest di integrita immutabile per gli embedding esistenti:
  hash, chiavi univoche, dimensione 768 e finitezza.
- [x] Rifittare preprocessing e modello selezionato sull'intero outer-development
  per classificazione, cognizione, morfometria futura e survival.
- [x] Applicare pesi participant-equal nel fitting dei task a visite ripetute.
- [x] Imporre coorti paired identiche, arricchire le pair table con la nuova
  morfometria ed escludere le coppie anatomiche a 6 mesi.
- [x] Implementare Uno C, cumulative/dynamic AUC, IPCW Brier, calibrazione
  survival e diagnostica Cox con residui di Schoenfeld.
- [x] Generare curve di calibrazione smooth per regressione e classificazione.

## Milestone 2 - Coorte e nested CV bloccata

- [x] Creare copie paper-specific draft di manifest e pair table; rigenerarle
  obbligatoriamente dopo il freeze.
- [x] Non usare né rigenerare i seed storici `20260623` e `20260718`.
- [x] Generare 4 outer fold × 3 repeat con i nuovi seed bloccati (artefatto
  draft verificato; da rigenerare dopo il freeze per l'analisi confermatoria).
- [x] Verificare zero overlap di participant tra train/validation/test.
- [x] Generare cohort flow gerarchico e riconciliato, esclusioni, eventi,
  follow-up, missingness, horizon error e precisione per task e outer fold.
- [x] Salvare hash, commit, environment e checkpoint provenance negli artefatti
  draft; rigenerare da commit pulito dopo il freeze.

## Milestone 3 - Esperimenti finali

- [ ] Eseguire AD-vs-CN come endpoint core diagnostico-state.
- [ ] Eseguire same-visit ADAS13, ADAS11 e MMSE.
- [ ] Eseguire future cognition a 12 mesi con carry-forward.
- [ ] Eseguire 24/36 mesi come sensitivity bloccata.
- [ ] Eseguire future hippocampus/ICV e ventricles/ICV horizon-matched.
- [ ] Eseguire MCI conversion 36 mesi con calibrazione.
- [ ] Eseguire first-MCI survival con eventi e censura verificati.
- [ ] Eseguire Task E solo se non ritarda il pacchetto principale.
- [ ] Generare il report completo senza selezionare endpoint o risultati favorevoli.

## Milestone 4 - Rafforzamento consigliato

- [ ] Selezionare un secondo encoder MRI foundation/self-supervised comparabile.
- [ ] Eseguirlo sul protocollo congelato senza nuovo tuning sul test.
- [ ] Valutare una coorte esterna e mappare gli endpoint compatibili.
- [ ] Eseguire stress test per ADNI phase/site/scanner.
- [ ] Decidere con gli autori quale estensione completare prima della submission.

## Milestone 5 - Figure, tabelle e supplemento

- [ ] Table 1: cohort characteristics e flow.
- [ ] Table 2: task e protocol definitions.
- [ ] Figure 1: brain-informatics pipeline.
- [ ] Figure 2: paired-effect forest plot con dispersione fold/repeat.
- [ ] Figure 3: performance per orizzonte e persistence baseline.
- [ ] Table 3: conversion, calibration e survival.
- [ ] Additional file: data dictionary e criteri completi.
- [ ] Additional file: metriche, CI e hyperparameters del run confermatorio.
- [ ] Additional file: ROC/PR/calibration e sensitivity analyses.
- [ ] Controllare titoli figura <= 15 parole e legende <= 300 parole.

## Milestone 6 - Manoscritto

- [~] Creare lo scaffold inglese specifico per *Brain Informatics*.
- [~] Completare review della letteratura con fonti primarie aggiornate; mappa
  iniziale aggiunta per JEPA, foundation MRI, ADNI longitudinale, baseline
  convenzionali e costo computazionale. Restano metadata bibliografici finali e
  ricerca sistematica pre-submission.
- [ ] Scrivere Methods dalle sorgenti canoniche, non dai report narrativi.
- [ ] Inserire Results solo dagli output paper-specific.
- [ ] Scrivere Discussion calibrata e limitations complete.
- [ ] Finalizzare abstract di 150--250 parole e 3--10 keyword.
- [ ] Completare dichiarazioni, ethics/ADNI acknowledgement e CRediT roles.
- [ ] Dichiarare in modo conforme l'eventuale uso di LLM nella preparazione.
- [ ] Revisionare inglese scientifico e consistenza terminologica.

## Milestone 7 - Riproducibilita e submission

- [ ] Preparare repository pubblico senza dati/embedding ADNI individuali.
- [ ] Aggiungere esempio sintetico machine-readable.
- [ ] Archiviare release di codice/config su Zenodo e ottenere DOI.
- [ ] Verificare licenza del codice e dei modelli esterni.
- [ ] Convertire il manoscritto al template Springer Nature LaTeX.
- [ ] Attivare doppia spaziatura, line e page numbering.
- [ ] Compilare con pdfLaTeX/TexLive compatibile e controllare riferimenti.
- [ ] Preparare cover letter specifica per Track 3/4.
- [ ] Confermare autori, ordine, corresponding author, funding e conflitti.
- [ ] Verificare copertura o budget APC prima della submission.
- [ ] Eseguire checklist pre-submission e approvazione di tutti gli autori.
