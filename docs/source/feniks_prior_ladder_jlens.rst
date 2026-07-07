FENIKS Prior Ladder and Jacobian Lens
=====================================

This runbook is for controlled FENIKS experiments on the 18D synthetic DSPS
closure dataset. HLTDS and FS2 remain useful debug/reference paths, but the
experiments here assume the FENIKS train/validation/test split.

Scientific Contract
-------------------

The ladder keeps three regimes separate:

* deterministic reconstruction, with ``kl_weight_max=0`` and no learned-prior
  term in the loss;
* NN+DSPS with a latent RealNVP prior, trained jointly under the amortized
  ELBO term ``E_q[logq - logp_beta]``;
* supervised RealNVP prior learning on truth parameters transformed to latent
  ``x`` space, followed by frozen-prior or fine-tuned amortized runs.

All experiment configs keep per-band zero-point calibration disabled.

Experiment Configs
------------------

The overlays live under ``configs/experiments/``:

* ``feniks_autoencoder_reconstruction_18d_h100.yaml``
* ``feniks_joint_realnvp_kl_fixed_18d_h100.yaml``
* ``feniks_joint_realnvp_kl_annealed_18d_h100.yaml``
* ``feniks_supervised_truth_realnvp_18d_h100.yaml``
* ``feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100.yaml``
* ``feniks_amortized_supervised_prior_finetune_18d_h100.yaml``

Use ``LIMIT=20000`` for subset training while comparing regimes. Set
``LIMIT=`` explicitly for a full training run.

Multi-GPU Training
------------------

Amortized training supports optional local-device data parallelism:

* ``DATA_PARALLEL=single`` keeps the one-device path.
* ``DATA_PARALLEL=auto`` uses pmap only when several local JAX devices are
  visible.
* ``DATA_PARALLEL=pmap`` requires several local JAX devices and averages
  gradients with ``jax.lax.pmean``.

``BATCH_SIZE`` and ``JAX_BATCH_SIZE`` are global batch sizes. In pmap mode,
``JAX_BATCH_SIZE`` must be divisible by the number of local GPUs. The Slurm
script requests one GPU by default; pass ``--gres=gpu:4`` at submission time
for a four-GPU H100 training job.

Autoencoder Baseline
--------------------

Train the deterministic reconstruction baseline:

.. code-block:: bash

   sbatch --gres=gpu:4 --export=ALL,STAGE=ae_train,DATA_PARALLEL=pmap,LIMIT=20000,EPOCHS=50 \
     scripts/feniks_prior_ladder_h100.slurm

Run held-out inference:

.. code-block:: bash

   sbatch --export=ALL,STAGE=ae_infer,INFER_LIMIT=5000 \
     scripts/feniks_prior_ladder_h100.slurm

Run a sharded Jacobian Lens diagnostic:

.. code-block:: bash

   sbatch --array=0-3 --export=ALL,STAGE=ae_jlens,JLENS_NUM_SHARDS=4,JLENS_LIMIT=1024 \
     scripts/feniks_prior_ladder_h100.slurm

Joint RealNVP Prior
-------------------

Fixed small KL:

.. code-block:: bash

   sbatch --gres=gpu:4 --export=ALL,STAGE=joint_fixed_train,DATA_PARALLEL=pmap,LIMIT=20000,EPOCHS=80 \
     scripts/feniks_prior_ladder_h100.slurm

Annealed KL:

.. code-block:: bash

   sbatch --gres=gpu:4 --export=ALL,STAGE=joint_annealed_train,DATA_PARALLEL=pmap,LIMIT=20000,EPOCHS=100 \
     scripts/feniks_prior_ladder_h100.slurm

For either run, use ``joint_fixed_infer`` / ``joint_fixed_jlens`` or
``joint_annealed_infer`` / ``joint_annealed_jlens`` afterwards.

Supervised Truth Prior
----------------------

Train a RealNVP prior directly on truth ``x`` samples:

.. code-block:: bash

   sbatch --gres=gpu:4 --export=ALL,STAGE=supervised_prior_train,DATA_PARALLEL=pmap,LIMIT=20000,EPOCHS=80 \
     scripts/feniks_prior_ladder_h100.slurm

Regenerate the final truth-vs-prior report:

.. code-block:: bash

   sbatch --export=ALL,STAGE=supervised_prior_report \
     scripts/feniks_prior_ladder_h100.slurm

Frozen Supervised Prior NN+DSPS
-------------------------------

Train the amortized model with the supervised RealNVP checkpoint frozen:

.. code-block:: bash

   sbatch --gres=gpu:4 --export=ALL,STAGE=supervised_frozen_train,DATA_PARALLEL=pmap,LIMIT=20000,EPOCHS=100 \
     scripts/feniks_prior_ladder_h100.slurm

Then run inference and J-lens:

.. code-block:: bash

   sbatch --export=ALL,STAGE=supervised_frozen_infer,INFER_LIMIT=5000 \
     scripts/feniks_prior_ladder_h100.slurm

   sbatch --array=0-3 --export=ALL,STAGE=supervised_frozen_jlens,JLENS_NUM_SHARDS=4,JLENS_LIMIT=1024 \
     scripts/feniks_prior_ladder_h100.slurm

Fine-Tune Variant
-----------------

The fine-tune config starts from the supervised checkpoint, freezes it for ten
epochs, then alternates encoder and prior updates. Treat it as experimental:
it can improve photometric ELBO while degrading truth-prior overlap.

.. code-block:: bash

   sbatch --gres=gpu:4 --export=ALL,STAGE=supervised_finetune_train,DATA_PARALLEL=pmap,LIMIT=20000,EPOCHS=100 \
     scripts/feniks_prior_ladder_h100.slurm

Jacobian Lens Outputs
---------------------

The dedicated CLI is the primary path:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/experiments/feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100.yaml \
     amortized-jacobian-lens-diffsky \
     --dataset Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet \
     --checkpoint outputs/runs/feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100/checkpoints/best.eqx \
     --feature-stats outputs/runs/feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100/feature_stats.json \
     --out outputs/runs/feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100_jlens \
     --limit 1024 \
     --batch-size 8 \
     --mode both

The command writes sharded parquet/csv/json artifacts under
``jacobian_lens_shards/part_XXXXXX/``. Combine shards with:

.. code-block:: bash

   python -m euclid_dsps.cli \
     --config configs/experiments/feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100.yaml \
     amortized-finalize-jacobian-lens \
     --out outputs/runs/feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100_jlens

Training remains single-GPU by default unless ``DATA_PARALLEL=auto`` sees
several local devices or ``DATA_PARALLEL=pmap`` is requested explicitly. J-lens
diagnostics remain sharded through Slurm arrays.
