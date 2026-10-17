# Semantic utility comparison

Local validation outputs are preliminary. Formal server runs require both complete50.
No gate/coverage/view augmentation is used. CE(task)+CE(semantic) jointly trains the backbone.
B changes scoring reference only; fixed directions do not supervise or align embeddings.
Utility is leave-one-occurrence-out marginal CE, not intrinsic client quality.

## Completion and dispatch

```json
{
  "compared_rounds": 2,
  "profile_dispatch_mismatch": 0,
  "client_dispatch_mismatch": 0,
  "same_completed_length": true,
  "both_complete50": false,
  "round0_backbones_equal": true,
  "final_global_equal": true,
  "valid_paired_dispatch": true
}
```

| arm | complete50 | utility n | Spearman | AUROC | classification |
|---|---|---:|---|---|---|
| original | False | 10 | -0.24848484848484848 | 0.5 | INSUFFICIENT_EVIDENCE |
| fixed_class | False | 10 | 0.44242424242424244 | 0.5625 | INSUFFICIENT_EVIDENCE |

## Actual protocol and runtime

| arm | completed | GPU mapping | train samples | validation n | source commits | gate | wall seconds |
|---|---:|---|---:|---:|---|---|---:|
| original | 2 | physical visible=0, logical=cuda:0 | 50000 | 500 | baseline619c02e / semantic31c8e8c | false | 26.074 |
| fixed_class | 2 | physical visible=0, logical=cuda:0 | 50000 | 500 | baseline619c02e / semantic31c8e8c | false | 25.757 |

Train recipe: CIFAR100 cached Dirichlet.3/seed1;100clients/10slots;7profiles;ep5/bs50/dropLast;SGD .01*.998^round,momentum.5,wd1e-4;no augmentation.
Aggregation is paper unnormalized delta with gamma1. Reserved subset is never evaluated.

## Simple signals and original/fixed scores on the same observations

| arm | signal | n | positive/negative | Spearman | Pearson | AUROC | round-bootstrap Spearman95CI |
|---|---|---:|---|---|---|---|---|
| original | semantic_score | 10 | 8/2 | -0.24848484848484848 | -0.3472927135321006 | 0.5 | None |
| original | original_score | 10 | 8/2 | -0.24848484848484848 | -0.3472927135321006 | 0.5 | None |
| original | fixed_score | 10 | 8/2 | 0.44242424242424244 | 0.3244085771805411 | 0.5625 | None |
| original | prototype_concentration | 10 | 8/2 | -0.01818181818181818 | 0.1887832619710439 | 0.75 | None |
| original | update_norm | 10 | 8/2 | 0.4909090909090909 | 0.4451263581554192 | 0.5625 | None |
| original | sample_count | 10 | 8/2 | 0.47416632407045034 | 0.3163024320222351 | 0.59375 | None |
| original | profile | 10 | 8/2 | 0.3963779236805213 | 0.40838362203470197 | 0.8125 | None |
| original | width | 10 | 8/2 | 0.3158168135806891 | 0.393060629530054 | 0.6875 | None |
| original | depth | 10 | 8/2 | -0.051899929610768204 | -0.12673932961011214 | 0.625 | None |
| original | loo_update_cosine | 10 | 8/2 | 0.23636363636363636 | -0.12433195571912974 | 0.3125 | None |
| original | reliability | 0 | 0/0 | None | None | None | None |
| fixed_class | semantic_score | 10 | 8/2 | 0.44242424242424244 | 0.3244085771805411 | 0.5625 | None |
| fixed_class | original_score | 10 | 8/2 | -0.24848484848484848 | -0.3472927135321006 | 0.5 | None |
| fixed_class | fixed_score | 10 | 8/2 | 0.44242424242424244 | 0.3244085771805411 | 0.5625 | None |
| fixed_class | prototype_concentration | 10 | 8/2 | -0.01818181818181818 | 0.1887832619710439 | 0.75 | None |
| fixed_class | update_norm | 10 | 8/2 | 0.4909090909090909 | 0.4451263581554192 | 0.5625 | None |
| fixed_class | sample_count | 10 | 8/2 | 0.47416632407045034 | 0.3163024320222351 | 0.59375 | None |
| fixed_class | profile | 10 | 8/2 | 0.3963779236805213 | 0.40838362203470197 | 0.8125 | None |
| fixed_class | width | 10 | 8/2 | 0.3158168135806891 | 0.393060629530054 | 0.6875 | None |
| fixed_class | depth | 10 | 8/2 | -0.051899929610768204 | -0.12673932961011214 | 0.625 | None |
| fixed_class | loo_update_cosine | 10 | 8/2 | 0.23636363636363636 | -0.12433195571912974 | 0.3125 | None |
| fixed_class | reliability | 0 | 0/0 | None | None | None | None |

## Incremental utility prediction

### original

```json
{
  "status": "INSUFFICIENT_SAMPLES",
  "n": 10,
  "rounds": 1
}
```

### fixed_class

```json
{
  "status": "INSUFFICIENT_SAMPLES",
  "n": 10,
  "rounds": 1
}
```

## Capacity, stage and class-coverage groups

| arm | group | n | diagnostic rounds | Spearman | AUROC | status |
|---|---|---:|---:|---|---|---|
| original | Small | 3 | 1 | 0.5 | 1.0 | INSUFFICIENT_SAMPLES |
| original | Medium | 5 | 1 | -0.2 | 0.5 | INSUFFICIENT_SAMPLES |
| original | Full | 2 | 1 | None | None | INSUFFICIENT_SAMPLES |
| original | Early | 10 | 1 | -0.24848484848484848 | 0.5 | INSUFFICIENT_SAMPLES |
| original | Middle | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |
| original | Late | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |
| original | valid_classes_lt25 | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |
| original | valid_classes_25to49 | 10 | 1 | -0.24848484848484848 | 0.5 | INSUFFICIENT_SAMPLES |
| original | valid_classes_ge50 | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |
| fixed_class | Small | 3 | 1 | 0.5 | 0.5 | INSUFFICIENT_SAMPLES |
| fixed_class | Medium | 5 | 1 | 0.4 | 1.0 | INSUFFICIENT_SAMPLES |
| fixed_class | Full | 2 | 1 | None | None | INSUFFICIENT_SAMPLES |
| fixed_class | Early | 10 | 1 | 0.44242424242424244 | 0.5625 | INSUFFICIENT_SAMPLES |
| fixed_class | Middle | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |
| fixed_class | Late | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |
| fixed_class | valid_classes_lt25 | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |
| fixed_class | valid_classes_25to49 | 10 | 1 | 0.44242424242424244 | 0.5625 | INSUFFICIENT_SAMPLES |
| fixed_class | valid_classes_ge50 | 0 | 0 | None | None | INSUFFICIENT_SAMPLES |

## Geometry and class availability

| arm | group | occurrences | valid classes mean | comparable classes mean | anchor cold classes mean |
|---|---|---:|---:|---:|---:|
| original | Small | 8 | 29.125 | 10.75 | 18.375 |
| original | Medium | 10 | 28.9 | 14.7 | 14.2 |
| original | Full | 2 | 27.0 | 25.5 | 1.5 |
| fixed_class | Small | 8 | 29.125 | 10.75 | 18.375 |
| fixed_class | Medium | 10 | 28.9 | 14.7 | 14.2 |
| fixed_class | Full | 2 | 27.0 | 25.5 | 1.5 |

original: cross-client same/different-class geometry (descriptive, no semantic-collapse success criterion):
```json
{
  "distinct_client_pairs": 36,
  "same_class": {
    "n": 36,
    "mean": 0.7452284726831648,
    "median": 0.7416359484195709
  },
  "different_class": {
    "n": 36,
    "mean": 0.6955244822634591,
    "median": 0.6954080164432526
  },
  "interpretation": "Descriptive prototype geometry, not channel alignment or update usefulness"
}
```

fixed_class: cross-client same/different-class geometry (descriptive, no semantic-collapse success criterion):
```json
{
  "distinct_client_pairs": 36,
  "same_class": {
    "n": 36,
    "mean": 0.7452284726831648,
    "median": 0.7416359484195709
  },
  "different_class": {
    "n": 36,
    "mean": 0.6955244822634591,
    "median": 0.6954080164432526
  },
  "interpretation": "Descriptive prototype geometry, not channel alignment or update usefulness"
}
```

## Seven profiles (final diagnostic-validation checkpoint)

| arm | profile | final validation CE | final validation accuracy |
|---|---:|---:|---:|
| original | 0 | 4.610318237304687 | 2.4 |
| original | 1 | 4.588893188476563 | 2.0 |
| original | 2 | 4.562432861328125 | 1.6 |
| original | 3 | 4.591475219726562 | 2.2 |
| original | 4 | 4.550337524414062 | 2.8 |
| original | 5 | 4.535115844726563 | 4.2 |
| original | 6 | 4.541770874023437 | 4.0 |
| fixed_class | 0 | 4.610318237304687 | 2.4 |
| fixed_class | 1 | 4.588893188476563 | 2.0 |
| fixed_class | 2 | 4.562432861328125 | 1.6 |
| fixed_class | 3 | 4.591475219726562 | 2.2 |
| fixed_class | 4 | 4.550337524414062 | 2.8 |
| fixed_class | 5 | 4.535115844726563 | 4.2 |
| fixed_class | 6 | 4.541770874023437 | 4.0 |

## Interpretation boundaries

Intervals resample whole rounds (1000 replicates); six planned clusters are few and intervals are exploratory.
Incremental prediction holds out entire rounds, uses fixed ridge penalty1 and fits scaling on training rounds only.
Missing anchor scores stay missing. Shared-observation comparison excludes cold-start missing references in BOTH scores.
Anchor-free concentration is descriptive. A zero neutral reference has undefined cosine and is not assigned a fake score.
Capability/stage/class-coverage groups with too few observations are INSUFFICIENT_SAMPLES.
Reserved official-test subset is not evaluated. It is historically exposed, not a pristine independent test.
See predictive_metrics.json for round-cluster CIs, class counts, subgroup results and incremental-vs-simple-baseline evidence.
