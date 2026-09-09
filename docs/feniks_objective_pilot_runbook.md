# FENIKS : gradient VI complet et pilote d'objectif

Etat du 9 septembre 2026. Ce document distingue les sorties operateur deja
mesurees des nouveaux tests implementes, encore a executer sur Jean-Zay.

## 1. Ce que le replay change dans le diagnostic

Le job **1959175**, `frozen_parent_long_replay_v1`, a termine en 12m59s :
contrat PASS, 16/16 cas, 196687 evaluations forward et 3 gradients de controle.
Il n'a entraine aucun parametre. Les checkpoints finaux 512 sont ceux de
`frozen_parent_long_local_vi_v1`, job 1957394.

A K4096, les 16/16 propositions locales observees et 15/16 simulees ont
`bad_k=1`. Les quatre bons-k locaux de K1024 ne le restent pas a K4096.
L'ELBO et les residus locaux sont generalement bien reproduits. Exemples tires
des chiffres colles par l'operateur, pas de fichiers telecharges :

| Cas | Proposition | ESS K1024 | ESS K4096 | RMS K1024 | RMS K4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| observed_002 | final 512, depart 0 | 43.75 | 4.14 | 1.8158 | 1.8099 |
| simulated_003 | ancre C | 115.47 | 384.00 | 2.9021 | 2.6914 |
| simulated_003 | final 512, depart 0 | 7.49 | 39.70 | 0.8807 | 0.8880 |
| simulated_003 | final 512, depart 1 | 5.15 | 79.51 | 0.8605 | 0.8651 |

Conclusion : prolonger encore la meme optimisation ou seulement augmenter K
n'est pas un plan correctif suffisant. Cela ne prouve ni des modes manquants,
ni une famille impossible, ni un gradient faux. Les moyennes ponderees de
banques a ESS proche de un ne deviennent pas des cibles d'entrainement.

## 2. Hypotheses testees, dans l'ordre

**H1 : erreur dans le gradient complet de l'objectif.** Les precedents audits
du decodeur ne verifient pas toute la chaine parametres de q -> tirage ->
densite/prior/vraisemblance. Il faut verifier cette chaine avant d'attribuer le
probleme au choix de l'objectif.

**H2 : objectif reverse-KL peu favorable aux poids d'importance.** Un gradient
correct peut reduire l'ELBO negative sans ameliorer la couverture de la cible
par q. Apres H1, comparer une adaptation de log-densite ponderee avec un controle
reverse-KL depuis la meme distribution est une experience falsifiable.

**H3 : exploration encore insuffisante.** Une adaptation wake n'invente pas
les regions absentes de ses tirages. Beaucoup de rejets du garde-fou sont un
resultat informatif, pas une raison pour les supprimer ou recommencer jusqu'a
obtenir un lot acceptable.

## 3. Audit de l'objectif entier

Tous les 16 cas et les deux checkpoints finaux fixes sont conserves, sans
classement. Le prior, le decodeur, les observations, masques, incertitudes et
contextes restent ceux du contrat qualifie. Tous les 32 points sont audites
avant de construire un optimiseur.

L'objectif de controle est :

```text
x_phi = sample(q_phi, epsilon)
L(phi; epsilon) = mean(logq_phi(x_phi) - logprior(x_phi) - loglike(x_phi))
```

Le meme bruit `epsilon` (MC8) est utilise aux points central, plus et moins.
AD/FD est examine sur deux directions par bloc : moyenne de base, log-ecart-type
de base, couches du flow. Les termes `logq`, `-logprior`, `-loglike` et le total
sont inspectes separement. L'identite entre densite renvoyee par le tirage et
densite inverse est verifiee en valeur et en derivee. Ce sont des tests
directionnels, pas une preuve exhaustive sur tous les parametres.

L'arithmetique native est le controle principal. Une copie float64 des
parametres sert uniquement a localiser la precision; les casts float32 internes
ne disparaissent pas automatiquement. Son PASS ne remplace jamais un
FAIL/INCONCLUSIVE natif. Toutes les sources restent immuables.

**Aucune adaptation n'est autorisee avant PASS de tous les audits natifs.**
INCONCLUSIVE signifie que les stencils ne fournissent pas une comparaison
resolue selon le contrat, pas que le gradient est demontre incorrect. Le
rapport indique le bloc et le terme qui motivent l'arret.

## 4. Comparaison corrective conditionnelle

Deux bras, deux departs, memes 16 cas. Les deux bras de chaque depart sont
initialises par le meme checkpoint final 512, pas par le meilleur checkpoint.

| Bras | Budget d'adaptation | Objectif |
| --- | --- | --- |
| `reverse_mc32` | 128 mises a jour x 32 tirages | ELBO negative actuelle, gradient reparametre |
| `wake_mixture256` | 16 tentatives x 256 tirages | Log-densite inverse ponderee, tirages/poids detaches |

Chaque bras/depart dispose de **4096 evaluations de tirages par le decodeur**.
Ce n'est pas une egalite de temps, de FLOPs, de pas Adam ou de retropropagation.
Le cout de l'audit et des evaluations est compte separement de ce budget
d'adaptation. Les deux bras sont executes sequentiellement, sur un seul GPU.

Pour la tentative wake t :

```text
r_t(x) = 0.5 q_phi_t(x | y) + 0.5 q_C(x | y)
x_i ~ r_t
logw_i = logprior(x_i) + loglike(x_i) - log r_t(x_i)
wbar = softmax(logw)
L_wake(phi) = -sum_i stop_gradient(wbar_i) log q_phi(stop_gradient(x_i) | y)
```

`q_C` est l'ancre amortie initiale figee, pas le depart local perturbe.
La densite du melange est calculee avec les deux composantes pour chaque
tirage, par `logaddexp - log(2)`. Le gradient ne passe ni par les poids ni par
le decodeur dans ce bras. Les distributions restent jointes et normalisees;
on ne remplace jamais un ensemble pondere par son meilleur tirage, sa moyenne
ou une pseudo-verite.

Une tentative doit avoir des entrees finies, **ESS >= 16** et **poids maximal
<= 0.20**. Sinon les parametres et l'etat Adam sont exactement conserves.
Elle consomme tout de meme ses 256 tirages du budget. Pas de boucle de retries
jusqu'au succes, pas de clipping ou temperage des poids, pas de banque
accumulee. Le rapport conserve le nombre de tentatives et d'updates acceptees.

Cette orientation de l'objectif de proposition s'inscrit dans le principe de
[Bornschein et Bengio, Reweighted Wake-Sleep](https://arxiv.org/abs/1406.2751).
Le gradient wake-phi auto-normalise est explicite dans
[Le et al. (2020), equation 7](https://proceedings.mlr.press/v115/le20a/le20a.pdf).
Le melange et les garde-fous sont des choix locaux du pilote. L'estimateur est
biaise a K fini; rejeter les lots concentres peut aussi biaiser l'adaptation.
Ni ces articles ni les garde-fous ne garantissent le support FENIKS.

## 5. Evaluation et decision

Checkpoints fixes apres 1024 et 4096 tirages du budget d'adaptation :
respectivement 32/128 updates du controle et 4/16 tentatives wake. Une tentative
rejetee compte dans l'avancement du budget, pas comme une mise a jour.
Evaluation independante de l'optimisation : K1024 au total a l'intermediaire
(2 x 512), K4096 au total au final (2 x 2048), avec deux repetitions independantes.
Les graines d'evaluation
ne choisissent ni les updates ni les checkpoints.

Comparer tous les cas et les deux departs avec ESS, Pareto-k, poids maximal,
ecart d'evidence entre repetitions, ELBO et prediction photometrique. La loss
wake et l'ELBO ne sont pas directement comparables entre elles; utiliser les
metriques d'evaluation communes. Des residus meilleurs seuls ne suffisent pas.

Rejets wake massifs : exploration actuelle insuffisante pour ce protocole.
Updates acceptees mais support non ameliore : correction non demontree.
Amelioration coherente sur les deux departs : candidat a confirmer sur une
cohorte independante. Aucun chemin ne selectionne un gagnant automatiquement,
ne produit d'enseignant NPE ou n'autorise l'apprentissage de population.

## 6. Lancement Jean-Zay

Prerequis : checkout a jour contenant le nouveau lanceur et environnement
`shine`. Une seule allocation **1 H100, 1 noeud, 1 GPU/noeud, 3 heures**.
Plafond logiciel : 1 200 000 evaluations comptabilisees, budget temps 9900s.
Forward et gradients ne sont pas des couts equivalents. Le preflight conserve
son facteur de securite et peut interrompre avant de depasser le budget.

```bash
cd "$WORK/dsps-popcosmos"
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
git pull --ff-only origin feature/feniks-exact-posterior-benchmark

export REPO_DIR="$PWD"
export CACHE_ROOT="$SCRATCH/feniks_sc_drws_runtime"
BASE="/lustre/fsn1/projects/rech/jrx/urx63nr/feniks_sc_drws_r29_hardmerge_20260828_002111"

bash scripts/submit_feniks_sc_drws_objective_pilot.sh \
  outputs/logs/feniks_sc_drws_balanced_npe_latest.env \
  "$BASE/frozen_parent_precision_night_v1" \
  "$BASE/frozen_parent_long_local_vi_v1" \
  "$BASE/frozen_parent_objective_pilot_v1"
```

Verifier le **nouveau JobID** et la destination `frozen_parent_objective_pilot_v1`.
Un echec avant `diagnostic_job=...` n'est pas une soumission. La commande
refuse une destination deja utilisee : conserver les artefacts et nommer un
nouvel essai, sans ecraser l'ancien.

```bash
bash scripts/monitor_feniks_sc_drws_local_vi_diagnostic.sh
```

Le crash du terminal ne stoppe pas un job deja soumis. Apres reconnexion,
revenir dans le repo, activer `shine`, puis relancer seulement le moniteur.
Pour les logs :

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
tail -F \
  "$DIAGNOSTIC_LOG_ROOT/diagnostic-${DIAGNOSTIC_JOB}.out" \
  "$DIAGNOSTIC_LOG_ROOT/diagnostic-${DIAGNOSTIC_JOB}.err"
```

Apres la fin (ou un arret motive par l'audit) :

```bash
source outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env
python scripts/summarize_feniks_sc_drws_objective_pilot.py "$DIAGNOSTIC_ROOT"
```

Ne pas relancer automatiquement si l'audit ne passe pas. Inspecter son terme,
son bloc et son statut. La completion Slurm ou un fichier FINAL ne certifient
pas la qualite scientifique.

## 7. Artefacts et traque du blocage

| Chemin relatif a la nouvelle racine | Contenu |
| --- | --- |
| `OBJECTIVE_AUDIT.json` | Agregat PASS/NOT_PASSED des 32 points et empreintes des checkpoints sources |
| `cases/<case>/audit_start_<s>/AUDIT.json` | Detail natif/parameter64 et termes/blocs controles |
| `cases/<case>/audit_start_<s>/stencils.csv` | Valeurs AD/FD par direction, composante et pas |
| `cases/<case>/amortized/SUMMARY.json` | Evaluation de l'ancre C figee |
| `cases/<case>/source_<s>/SUMMARY.json` | Evaluation du checkpoint final 512 avant adaptation |
| `cases/<case>/reverse_<s>/optimization.csv` | Trajectoire du controle |
| `cases/<case>/wake_<s>/optimization.csv` | Tentatives, gardes et mises a jour appliquees |
| `cases/<case>/{reverse,wake}_<s>/draws_01024/` | Checkpoint intermediaire au budget fixe, SUMMARY et tirages directs |
| `cases/<case>/{reverse,wake}_<s>/draws_04096/` | Checkpoint final au budget fixe, SUMMARY et tirages directs |
| `cases/<case>/COMPLETE.json` | Resultats et comptes d'updates/tentatives du cas |
| `FINAL.json` | Execution complete ou arret avant adaptation, sans promotion |

`FINAL.status=OBJECTIVE_AUDIT_NOT_PASSED` implique `optimization_started=false`.
Un controle natif non PASS bloque la creation des optimiseurs. Apres execution
du pilote, `FINAL.status=OBJECTIVE_PILOT_COMPLETE` atteste sa fin technique,
pas un support PASS. Les anciens recus et banques ne sont jamais reecrits.
L'agregat conserve les SHA256 des audits detailles et stencils. Ils sont
verifies avant adaptation, a chaque fin de cas et lors du resume CPU.

## 8. Documentation et limites de validation

Page Sphinx : `docs/source/feniks_current_status.rst`.
Chronologie : `docs/feniks_decoder_debug_log.md`.
Figures reproductibles : `scripts/plot_feniks_objective_pilot.py`.

Validation locale : **68 tests cibles passent**, avec le vrai flow conditionnel
et des cibles analytiques/mock. Les tests couvrent aussi les faux gradients,
les differences finies non resolues, le gel d'Adam sur rejet, le verrou avant
optimisation et l'alteration des preuves d'audit. Ruff, compileall, syntaxe Bash,
aides CLI et Sphinx `-W` passent. HTML inspecte en bureau et mobile.

```bash
python -m pytest -q \
  tests/test_local_vi_objective_audit.py tests/test_local_wake_diagnostic.py \
  tests/test_local_vi_diagnostic.py tests/test_qualified_local_vi.py \
  tests/test_long_local_vi.py tests/test_objective_pilot.py
```

```bash
python scripts/plot_feniks_objective_pilot.py
sphinx-build -W -b html docs/source outputs/docs_support_probe
```

HTML : `outputs/docs_support_probe/feniks_current_status.html`.
La figure de protocole est un schema, pas un resultat du pilote. Les exemples
du replay sont explicitement transcrits des sorties operateur. Les tests CPU
avec cibles synthetiques verifient les contrats logiciels; ils ne remplacent
ni le nouvel audit H100 ni la validation scientifique. Les anciens smokes
`fit/posterior` cites dans AGENTS ne sont pas revendiques si leurs configs
sont absentes du checkout.
