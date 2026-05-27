# Day 1 Notes — Introduction to Machine Learning
### CPSC 483 · Mon 5/26 · Dr. Panangadan · 98 slides

> Lecture covers: what learning means, what ML is, why it exists, the full taxonomy of ML approaches, and a worked research example (detecting public-transit use from phone sensors).

---

## 1. What "learning" means

Before defining ML, the lecture grounds the term *learning* itself. Learning is the process of acquiring **knowledge**, and knowledge takes several forms:

- **Facts** — discrete true statements (a red car; π = 3.14159…)
- **Patterns** — repeated regularities in facts
- **Concepts** — classes formed by generalizing across patterns (0, 2, 4, 6 → even numbers; dog/cat/bird → animal)
- **Rules** — implications (if P then Q; if it rains, the ground is wet)
- **Models** — abstract representations, often mathematical (v = d/t, Fibonacci recurrence, simulations)
- **Skills** — the ability to perform complex activities (playing soccer, writing programs)

ML systems acquire knowledge in most of these forms — patterns, concepts, rules, and models — from data rather than from explicit human encoding.

## 2. ML definitions worth knowing

Three canonical definitions, each emphasizing something different:

- **Samuel (1959):** computers learning *without being explicitly programmed*. The earliest framing — ML as the alternative to rule-writing.
- **Simon (1983):** any change in a system that makes it perform better on repetition of a task, or on a related task from the same population. Emphasizes *improvement through experience*.
- **Mitchell (1997):** a program learns from experience E with respect to a task T and performance measure P if its performance at T (measured by P) improves with E. The most operational definition — it forces you to name three things (T, P, E) for any ML problem.

Mitchell's framing is the one to memorize. On any new problem the first questions are: *what's the task? what's the metric? what's the experience (data)?*

## 3. Traditional programming vs. the ML approach

A worked example: write a function that guesses whether a name is masculine or feminine.

**Traditional approach** — write rules from intuition:
```python
def m_or_f(name):
    if name[-1] == 'a':
        return "F"
    else:
        return "M"
```
Works on `Laura → F`, `Carlos → M`, etc., but breaks on `Jose`, `Belen`, and many counterexamples. Adding more rules is brittle.

**ML approach** — collect labeled examples and let the algorithm generalize:
```python
def m_or_f(name):
    mydata = [("Laura", "F"), ("Carlos", "M"), ("Jose", "M"),
              ("Maria", "F"), ("Belen", "F")]
    # compare 'name' against mydata.Input, find the closest match
    # (nearest neighbor), and return its corresponding output
```

The structural shift is captured in two diagrams:

| Paradigm | Input to computer | Output |
|---|---|---|
| Traditional programming | data + a program | output |
| Machine learning | data + outputs | a program (model) |

ML produces the *function* itself from examples. That function then runs in a separate **testing** phase on new data.

## 4. Why ML?

ML is the right tool when:
- existing solutions need huge rule lists or constant fine-tuning,
- the environment changes — an ML system can be retrained on fresh data,
- the problem is so complex that no traditional algorithm gives a good solution,
- or you want *insights* from large amounts of data (data mining).

## 5. ML's place in AI

ML is one branch of AI, alongside logic, search algorithms, statistics, and neural networks (which themselves now overlap heavily with ML).

## 6. Discussion questions the instructor flagged
Worth thinking through — these are the framing questions for the class activity, and the kind of conceptual prompts that resurface in exam essays.

1. One task in your life that could be automated.
2. A situation where a pet surprised you with what it learned.
3. Who's responsible when an ML-based system causes harm — designer, user, deployer?
4. Is it sufficient to treat ML systems as black boxes, or must we understand their internals to mitigate bias?
5. The energy and water costs of training large language models — what mitigation steps exist?
6. If ML models can predict student success, should professors use them?
7. Is the current ML wave here to stay, or a fad?

## 7. Taxonomy of ML — the big organizing structure

This is the spine of the lecture. Six categories by supervision style, plus a handful of orthogonal "other" categories.

### Six main categories

| Category | What it learns from | Outputs / problems |
|---|---|---|
| **Rote learning** | direct memorization | pattern matching, lookup |
| **Supervised** | labeled examples (X, y) | classification, regression |
| **Unsupervised** | unlabeled data | clusters, associations, features |
| **Semi-supervised** | small labeled set + large unlabeled set | mixed approach |
| **Self-supervised** | unlabeled data with labels *generated from the data itself* | representation learning, modern LLMs |
| **Reinforcement** | reward signal from acting in an environment | learned policies / action rules |

### Other categories (orthogonal — describe *how* rather than *what supervision*)
- **Deep learning** — many-layered neural nets.
- **Online / adaptive learning** — incremental updates as new data arrives.
- **Transfer learning** — adapt a model trained on one task to a related task.
- **Ensemble learning** — combine multiple models for better predictions.

---

## 8. Supervised learning in detail

### Data structure
Each row is an **instance** with two parts:
- **Features / feature vector** — input attributes (X₁, …, Xₖ).
- **Label / class / target** — the output Y.

Two worked examples used:
- The "names → M/F" toy dataset, with derived features: `nchars`, `lastchar`, `nvowels`.
- The classic **iris** dataset: 4 numeric features (sepal length/width, petal length/width) → species (setosa, versicolor, virginica).

### Training and testing
Data is split into:
- **Training data** — features + labels used to learn `f: X → Y`.
- **Test data** — held back. Features only go into the model; labels are used to *evaluate* predictions.

Process: train on train features+labels → predict on test features → compare predictions against test labels.

### Formal statement
Given training data {(X₁, …, Xₖ, Yⱼ) | j = 1…N}, learn a function `f: X → Y`. The process of learning `f` is called **training**. Then use `f` to predict outcomes ŷ for new X:
- If ŷ is continuous → **regression**.
- If ŷ is discrete → **classification**.

Mental model: **classification finds a decision boundary** that separates groups; **regression finds a function that fits** the points.

### Examples
- **Digit / character recognition** — classify images of handwritten 0–9. Used in license-plate readers, multimedia info extraction.
- **Loan-application risk** — classify applications as high vs low risk using income, credit history, debt.

### Supervised methods (names to know, even if not yet covered)
- For **regression**: least squares, gradient descent, KNN, neural nets, naïve Bayes, decision trees, random forests, SVM, logistic regression.
- For **classification**: logistic regression, KNN, neural nets, naïve Bayes, decision trees, random forests, SVM.

Note that most supervised methods can be tuned to do *either* classification or regression with small changes — they aren't strictly one or the other.

---

## 9. Unsupervised learning

No labels. You're given data and you look for structure. Three main flavors:

- **Clustering** — group similar instances. Outliers are the instances that don't fit any cluster (this is **outlier / anomaly detection**).
- **Association rule mining** — find items that co-occur. Rule form X → Y, evaluated by:
  - *Support* = fraction of transactions containing X ∪ Y (e.g., {bread, milk, diaper} in 2 out of 5 transactions → s = 0.4)
  - *Confidence* = fraction of X-containing transactions that also contain Y (e.g., 2 of 3 → c = 0.67)
  - Classic use: market-basket analysis.
- **Feature selection** — find which features actually matter; reduce dimensionality.

### Applications
Fake-news detection, object detection, recommender systems, fraud detection, visualization.

### Methods (names to know)
- Clustering: partitional, hierarchical, density-based, model-based.
- Association rules: Apriori, FP-growth.
- Feature selection: correlation/covariance matrix, information gain, Fisher's LDA, PCA.
- Language modeling: learning from text data.

### Three problems contrasted visually
- **Classification** — colored regions separated by boundaries.
- **Regression** — a fitted curve through points.
- **Clustering** — groups discovered without labels.

---

## 10. Semi-supervised learning

The motivation is practical: labeled data is expensive (human annotation by experts), but unlabeled data is cheap and plentiful. SSL uses a small labeled set + a large unlabeled set.

**Self-training** is the canonical algorithm:
1. Train a classifier on the labeled data alone.
2. Apply the classifier to unlabeled data to generate predicted labels.
3. Treat the high-confidence predictions as new training data; retrain.

Only the most-confident pseudo-labels are added at each iteration to avoid cascading errors.

---

## 11. Self-supervised learning

Distinct from semi-supervised: here labels come from the *data itself*, not from a small labeled seed.

Key idea: design a **pretext task** that creates supervision automatically. Examples:
- **Masked language modeling:** "The cat ___ on the mat" — predict the masked word. (This is how BERT, GPT-family pretraining works.)
- **Computer vision pretext tasks:** predict relative position of image patches, colorize grayscale images, etc.
- **Autoencoders:** compress input then reconstruct it — the bottleneck representation is what you learn.

### Capabilities
- Predict any part of the input from any other part.
- Predict future from past (or vice versa).
- Predict the obscured from the visible.

### Benefits and limits
- **Benefits:** scales without labeling cost; conceptually closer to how humans learn.
- **Limits:** requires lots of compute; standalone accuracy on downstream tasks may be lower than supervised.

---

## 12. Reinforcement learning

Goal-directed learning by acting in an environment and receiving rewards.

```
Agent ──action──▶ Environment
  ▲                    │
  └──reward/state──────┘
```

The agent **explores** different actions and progressively **exploits** the ones that yield the most reward, aiming for maximum *long-term* reward (not just immediate).

- Training data can seed an initial model, but isn't required.
- Studied in many fields: game theory, control theory, decision theory, operations research, simulation optimization.

### Famous applications
- **AlphaGo** beat human champion Lee Sedol; **AlphaZero** learned the rules itself and surpassed AlphaGo.
- Robot locomotion, autonomous vehicles, control problems, business decision making.

### Methods (named for vocab)
- **Model-based RL:** dynamic programming on Bellman optimality — policy iteration, value iteration.
- **Model-free RL:** Monte-Carlo methods, temporal-difference learning, SARSA, Q-learning, Deep Q-Networks (DQN). On-policy vs off-policy distinction.

### Is RL supervised or unsupervised?
**Both, and neither.** It's *supervised* in that policies improve via reward feedback. It's *unsupervised* in that explicit goals aren't given — the agent discovers them via trial and error.

---

## 13. The "other categories"

### Deep learning
- Many-layered networks that learn hierarchical features automatically.
- Needs high-performance compute (GPUs/TPUs) and large datasets.
- Applications: image recognition, autonomous driving, text generation, medical research.

### Online / adaptive learning
- Updates the model incrementally as data streams in — no full retraining pass.
- Critical for big-data analytics and any streaming setting.

### Transfer learning
- Train on a source task, *transfer* the learned representations to a related target task.
- Examples: ImageNet-pretrained CNNs (VGG, ResNet) for new vision tasks; pretrained language models (BERT, GPT) for NLP downstream.
- Big practical advantage: less data needed for the target task.

### Ensemble learning
- Combine multiple learners for better predictions than any single one.
- Examples: **random forests** (bootstrap + many decision trees), Bayes-optimal classifiers.
- Applications: malware/intrusion detection, fraud detection, emotion recognition.
- Challenges: mixing models with different knowledge representations; compute cost.

---

## 14. Choosing the right ML method

A decision tree of practical considerations:

- **Do you have labels?** → supervised, otherwise unsupervised or semi-supervised.
- **What's the output type?** → regression / classification / clustering / association rules.
- **Are you learning a policy?** → reinforcement learning.
- **Do you have GPUs and lots of data?** → deep learning is viable.
- **Is the data streaming?** → online learning; otherwise batch is fine.
- **Among algorithms, weigh:** accuracy, training/prediction speed, noise tolerance, model complexity, interpretability.

---

## 15. Disciplines that feed into ML
Worth knowing so the math prerequisites in later chapters don't feel arbitrary:

- **Computer science:** data structures, algorithms, databases, HPC, complexity theory, AI.
- **Mathematics:**
  - *Statistics & probability* — descriptive stats, sampling, hypothesis testing, regression, distributions, Bayes' theorem.
  - *Linear algebra* — vectors, matrices, eigenvalues, factorization, orthogonality, PCA.
  - *Calculus* — derivatives, partial derivatives, optimization, sigmoid/logit.
  - *Numerical analysis* — computational optimization.
- **Information theory:** information, entropy.
- **Decision science / control theory** — RL, game theory.
- **Psychology, neurobiology** — cognitive learning.
- **Philosophy** — logic and reasoning.

---

## 16. Open questions in ML

The slide listing "current issues" is essentially the exam-essay-question pool:

- How do we frame applications as ML problems?
- Which algorithms apply, and how do training-data size and noise affect accuracy?
- How do we measure accuracy on unseen data (i.e., **generalization**)?
- How does model complexity matter? — **Occam's razor:** prefer the simplest answer that fits.
- How can prior knowledge accelerate learning?
- What are the theoretical limits of learnability?
- How can systems alter their own representations? (the deep-learning question)
- What can biological learning teach us?

---

## 17. Case study — public-transit detection from phone sensors

The last third of the deck is a worked research example. It's worth knowing because it shows the *full* ML pipeline applied end-to-end, and it'll likely come up as a template for project work.

### Motivation
GPS is good at *where* you are, not *how* you're traveling. Knowing whether someone is on a bus would enable:
- Transit-aware location recommendations (vs. Yelp/Google Maps, which only consider current location).
- Incentivizing public-transit use via rewards on detected bus trips.

### First approach: GPS + GTFS matching (no ML yet)
**Input:** a GPS track from the phone + a city's transit schedule in **GTFS** (General Transit Feed Specification — open standard adopted by OCTA, LACMTA, CTA, etc.).
**Output:** time segments classified as "on a bus" or "not."

**Algorithm:**
- For each GPS point, check whether there's a bus route nearby AND a scheduled trip on it at that time of day.
- Multiple thresholds had to be optimized:
  - Max distance from a route: tested 30/60/90m → chose 60m.
  - Max distance from a bus stop: tested 300/600/900m → chose 900m.
  - Max time difference between the GPS time and the scheduled bus time.
- Aggregate classifications over a sliding window of 5 points; label "on bus" if ratio ≥ 0.75.

**Result:** ~77.8% average accuracy across 14 trips (>20,000 GPS points), 27 hyperparameter combinations tested.

**Problems noted:**
- GPS is power-hungry and inaccurate in dense urban areas (skyscrapers cause multipath).
- GTFS schedules are static — they don't reflect delays or detours.

### Second approach: ML on inertial sensors
**Hypothesis:** the *vibration profile* of a bus ride differs subtly from a car ride.

**Sensors used** (modern Android phones have all of these):
- **Accelerometer** — 3-axis acceleration (physical + gravity).
- **Gyroscope** — 3-axis rotation rate.
- **Gravity sensor** — composite estimate from accel + gyro.
- **Android Activity Recognition API** — already classifies into walking/running/in-vehicle/etc.

**Pipeline:**
1. **Segmentation:** windows ≥ T ms; if fewer than 4 data points, expand to 2T.
2. **Feature extraction** (orientation-invariant):
   - Mean squared magnitude of acceleration.
   - Mean squared magnitude of rotation rate.
   - Variance of acceleration magnitude.
   - Variance of rotation rate magnitude.
   - Relative angle between sensor axes.
   - Variance of the relative angle.
3. **Classification:** Random Forest (scikit-learn).
4. **Evaluation:** ~151,000 data points, ~15 hours of activity; checked effect of segment length; ranked feature importance by depth of nodes in each tree.

**Why this matters as a teaching example:**
- It shows the **whole arc**: framing the problem → choosing inputs → handcrafting features → training a classifier → evaluating → iterating.
- It illustrates **feature engineering** (the hand-designed invariant features) — a key sub-skill that pure deep learning sometimes hides but is still tested in this course.
- It introduces **random forests** before Chapter 6 — keep the name in mind.

---

## 18. Summary cheat sheet

| Concept | One-liner |
|---|---|
| Mitchell's definition | improvement at task T (measured by P) with experience E |
| Supervised | learns f: X→Y from labeled (X,y) |
| Classification vs regression | discrete output vs continuous output |
| Unsupervised | finds structure without labels (clusters, associations) |
| Semi-supervised | small labeled set bootstrapped by lots of unlabeled |
| Self-supervised | labels generated from the data itself (LLM pretraining) |
| Reinforcement | learns a policy from action → reward feedback |
| Transfer learning | adapt a model trained on task A to related task B |
| Ensemble learning | combine multiple weaker models (e.g. random forest) |
| Occam's razor | prefer the simplest model that fits |
| No-free-lunch | no algorithm is best for all problems → try several |

---
*Notes built 2026-05-27 from Day 1 slide deck (98 slides).*
