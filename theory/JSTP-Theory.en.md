# Joint State Transition Prediction: A Theory of Intelligence

## From things happening together in the world to language that can carry a thought forward

**Theory proposed by Ke Yue**  
**Document:** Full theoretical manuscript and draft formalization  
**First draft:** 16 November 2026 · **Chinese text revised:** 19 November 2026

[Chinese original](JSTP-详细理论稿-2026-11-19.md) · [English PDF](JSTP-Theory-Ke-Yue.en.pdf) · [Companion empirical paper](https://doi.org/10.5281/zenodo.24366742)

> **Central thesis.** The basic function of intelligence is prediction. The system is itself part of the world. From a situation constituted jointly by itself and its surroundings, it predicts what will change next. The prediction may be no more than “my hand will reach out,” or it may extend to the details of how the hand and an object will move together. “Joint” means that the agent and its surroundings already participate in the same process of change. It does not mean that every prediction must cover the futures of both. The theory further proposes that segmentation can be learned within this same network, and that higher intelligence exists only at the level of language: it arises when the same predictive process continues with language as its content.

---

## Abstract

This manuscript proposes a theory of intelligence built around joint state transition prediction (JSTP). It asks whether a single account can connect experience in the world, the nervous system, learning, language, action, logic, causation, and science.

The author's starting point is that the nature of things consists in their undergoing transitions together with other things. The study of intelligence requires a weaker premise: an observer can distinguish some local parts and experience them changing together or separately. The system itself is one of the participants. Many units that can be active separately yet remain connected give it the means to distinguish local information. Which parts it comes to track as one object depends on their connections, ongoing activity, and past experience. There is no need to supply the network with an inventory of objects already segmented.

Observations can be incomplete, and so can the prediction itself. On seeing a cup slip, the system can directly predict that it will reach out, or expect only that the cup will move roughly downward. It need not compute a video in the background and leave it undisplayed. Conversely, if catching the cup depends on timing and position, the theory allows those details to be predicted. Learning changes not only what the system retains about an event but also how it will distinguish, track, and predict things in the future. Compression need not be an additional basic mechanism of intelligence.

Language is not a code deliberately designed by a subject that already understands the world. Sounds, actions, facial expressions, and written marks begin as elements of actual situations. In shared activity they recur alongside other elements, making it possible to predict the rest of a situation, or what happens next, from one part of it. A word neither summarizes an entire event nor stands in for the whole of reality. It is an element from which further predictions can proceed. As sounds and symbols come to predict other sounds and symbols, language develops a space of states in which activity can continue for a time without the corresponding external scene being present.

The theory makes a strong claim: nonlinguistic prediction can be complex and can get things right, but it does not itself judge whether a reason holds or whether something is right or wrong. Higher intelligence of that kind arises only at the level of language. When a judgment is made, the same network is still predicting; no additional judge appears. Action need not pass through judgment. The body's motor pathways or an actuator interface can directly realize a prediction of the system's own next action. Anticipating detailed consequences and comparing alternatives before acting is another route the framework allows, not the definition of action.

This leads to a proposal for a unified artificial intelligence. World events, language, memory, the body, and action can inform predictions of one another in a common system of latent states. Different forms of data provide different things to learn; they need not correspond to fundamentally different kinds of intelligence.

**Keywords:** joint state transition; prediction; implicit structure; emergence of language; action; generalization; local worlds; unified multimodal intelligence.

---

## How to read this manuscript

This is a hypothesis under development, not an established result in neuroscience. When the manuscript says “the theory holds,” it states the theory's own position. The notation makes its claims more precise; it does not supply a complete neural implementation, training algorithm, or proof.

The author's later clarifications take precedence. Three distinctions are especially important: **the situation from which a prediction is made versus the particular thing predicted; a structure that can distinguish local information versus experience that has already grouped parts into an object; and performing an action correctly versus judging it to be correct.** Words are elements of shared situations, not substitutes for whole events. The sequence “first compress, then symbolize” in an early English draft is not a basic mechanism of this account.

A theoretical claim, an explanation proposed on its basis, and an engineering implementation are different things. Building a model that works in a certain way does not establish that people work that way. Section 15 separates completed experiments from questions still open.

## Contents

1. [What is the theory trying to explain?](#1-what-is-the-theory-trying-to-explain)
2. [Minimal assumptions about reality and local worlds](#2-minimal-assumptions-about-reality-and-local-worlds)
3. [What is a joint state?](#3-what-is-a-joint-state)
4. [Prediction with incomplete information](#4-prediction-with-incomplete-information)
5. [What implicit structure is, and what it is not](#5-what-implicit-structure-is-and-what-it-is-not)
6. [Learning, generalization, and knowledge](#6-learning-generalization-and-knowledge)
7. [How language can arise in practice](#7-how-language-can-arise-in-practice)
8. [How language can sustain a predictive process of its own](#8-how-language-can-sustain-a-predictive-process-of-its-own)
9. [Where do causation, logic, and reasoning belong?](#9-where-do-causation-logic-and-reasoning-belong)
10. [Action, goals, and exploration](#10-action-goals-and-exploration)
11. [Thought, self, and consciousness](#11-thought-self-and-consciousness)
12. [Science and reality](#12-science-and-reality)
13. [Architectural implications for unified AI](#13-architectural-implications-for-unified-ai)
14. [Two worked examples](#14-two-worked-examples)
15. [Testable claims and experiments that distinguish them](#15-testable-claims-and-experiments-that-distinguish-them)
16. [Present limits and open questions](#16-present-limits-and-open-questions)
17. [A minimal formalization and a register of claims](#17-a-minimal-formalization-and-a-register-of-claims)
18. [The theory in brief](#18-the-theory-in-brief)

[Appendix A: Glossary](#appendix-a-glossary) · [Appendix B: Relation to the early English draft](#appendix-b-relation-to-the-early-english-draft)

---

## 1. What is the theory trying to explain?

### 1.1 More than renaming familiar abilities

“Intelligence makes predictions” is a weak claim. A system might predict while relying on explicit rules, a causal graph, or a planner. The claim here is stronger:

> Perception, language, reasoning, and action need not arise from separate basic mechanisms of intelligence. They can be manifestations of one joint state transition prediction process operating on different kinds of content.

The point is not to rephrase everything as a change of state. Any computation can be described in those terms; that alone explains nothing about intelligence. The question is: **Can predictive learning alone give a system an internal organization of transitions that generalizes, and thereby produce abilities usually assigned to different cognitive faculties?**

### 1.2 Distinguishing experience, internal operation, and language

Three levels need to be distinguished if the language of explanation is not to be projected straight into the model.

| Level | What the theory considers | What not to infer |
| --- | --- | --- |
| Experience in the world | Joint situations the observer encounters, and how they change | The ear can hear the word “because.” Hearing it does not mean that the senses have established a cause. |
| Implicit operation | How internal states take shape through experience and support subsequent prediction | A logically valid output is not by itself evidence of an internal library of logical rules. |
| Language and theory | How people state what objects, relations, causes, logical connections, and scientific propositions are | A concept in an explanation need not be a basic unit of the underlying computation. |

These are not three isolated worlds. Utterances, displayed text, the acts of reading and writing, and internal linguistic activity are themselves part of what goes on in the world and in the system.

The distinction prevents circular explanations: one cannot explain a behavior by saying “the person understood the causal relation,” then cite the same behavior as proof of an internal causal mechanism.

### 1.3 “Pure prediction” does not mean starting with nothing

“Pure” excludes additional, prespecified higher cognitive primitives. It does not exclude the conditions a system needs in order to work.

The system still needs a physical structure that can carry states, a way to retain the influence of history, learning, sensory input, and an output interface through which it can act. It can have multiple timescales, local circuits, and distinct information pathways.

Pure prediction is not an isolated function without memory, training, a body, or feedback. The more precise claim is that **these conditions can support ongoing prediction without requiring a separate agent to do the causing, reasoning, or deciding.**

---

## 2. Minimal assumptions about reality and local worlds

### 2.1 A minimal assumption about reality

The author's view is that **things are, in their nature, things undergoing transitions together.** This is not merely a convenient way to record data. It is the view that things exist in a process of change with other things. One need not first prove this claim about the whole world to study how a system learns to predict. The minimum experimental premise is that an observer can distinguish some parts and observe them changing together or separately.

In the simplest notation:

```text
S_t → S_(t+1)
```

The arrow denotes a transition or succession, not an already established causal influence. The index `t` marks an order or sampling scheme for the process under discussion. It does not require all local worlds to share an absolute clock.

“State,” too, is a unit of description. Describing reality in states does not commit us to the view that it ultimately consists of a sequence of static frames.

### 2.2 Separability and jointness are not two additional entities

An observation may encompass two local parts, `A` and `B`:

```text
(A_t, B_t) → (A_(t+1), B_(t+1))
```

This says only that both take part in a process of change. It does not posit two fully independent states and a further mechanism that assembles them.

Nor do independent descriptions of the parts necessarily exhaust the whole. The configuration in which the parts occur together may itself matter to the description.

Here, “separable” means **locally distinguishable**, not “necessarily independent.” “Joint” means **constituting a state together**, not “already judged by a cognitive system to stand in a particular relation.”

### 2.3 What a theory of local worlds contributes

A local-world account allows different domains to have different state contents, forms of organization, and modes of transition. It does not assume a single global vocabulary adequate to describe everything.

A local description can be written as:

```text
W_i = (S_i, T_i)
```

Here `S_i` denotes the range of states that can be discussed locally, and `T_i` a description of their transitions. There is no guarantee that `T_i` can determine what follows independently of the environment.

Where local worlds interact, one may discuss a larger state that includes them. It does not follow that every local world can be combined without conditions into an omniscient global model.

Local worlds provide the setting for this theory of intelligence: **a system encounters the local world it can reach and learns something of changes elsewhere through the limited connections available to it.** Whether heterogeneous local worlds are the ultimate constituents of reality is a stronger philosophical and physical claim. A capacity for prediction does not prove it.

### 2.4 Intelligence is itself in the world

An intelligent system is not something that stands outside reality to observe it. Its perceptions, utterances, actions, and internal changes participate in real transitions.

A joint state in experience can therefore include not only “what is happening out there,” but “how I am now,” “what I just did,” and “how someone responded to me.”

This is the starting point for treating action as part of prediction: **the agent itself is not an exception outside its predictive model.**

---

## 3. What is a joint state?

### 3.1 A state can carry many kinds of information at once

A joint state is not an indivisible label. It may include sight, sound, bodily sensations, the influence of memory, linguistic activity, and an action already under way.

As an aid to explanation one might write:

```text
Z_t = (visual information, sound, bodily information,
       effects of memory, linguistic activity, action-related information, ...)
```

These are an observer's functional categories, not six neatly separated drawers inside the system.

A simpler implementation-neutral notation is:

```text
Z_t = (z_1, z_2, ..., z_n)
```

Many units that can change together constitute the present state. Several sources of local information can participate in prediction at once. They need not first be named as objects, nor must the system first produce the statement “A is related to B.”

### 3.2 Physiological differentiation is not a one-to-one mapping to concepts

The author's original intuition was that a state is not represented by one indivisible unit: many neurons are active together. A joint state therefore has a possible physiological basis.

The theoretical implication is that **many simultaneously varying degrees of freedom can carry a complex joint state.** It is not the claim that:

```text
Neuron one is the cup.
Neuron two is the hand.
Neuron three is “push.”
```

The theory requires no such correspondence. Overlapping patterns of activity can contribute information to prediction.

To say that “segmentation has its basis in the structure of the neural network itself” is first to say that local activities can differ and influence one another. Having a great many neurons does not mean the cup has already been segmented from the table.

### 3.3 From local differences to an object that can be acted on separately

A pattern on a cup has an edge. So does a hand where it covers the cup. Detecting both edges is not the same as knowing that the first lies on the cup while the second belongs to an occluding hand. Segmentation works in two directions: distinguishing different things and bringing together parts whose appearance differs or that are temporarily separated from view.

When the cup is picked up, hand, handle, and body all move together. When it is released, the hand can leave on its own; the handle and body still rotate together. One instance of common motion does not distinguish those two relations. Experience of both grasp and release gives a network an opportunity to do so. The proposal is that, in predicting these different continuations, a single network can learn which activities should be held together and which updated separately. This can become a stable, reusable way of grouping without an object name or an output segmentation map.

To test whether a system treats something as an individual, arrange two histories with identical final images after occlusion. In one, the two exposed ends belong to one rigid object. In the other, they belong to two objects that had been moving together. Push only the left end. Record in advance whether the network expects the right end to move, then compare its prediction with the outcome. **Outlining a currently visible region is one kind of evidence. Using history to predict what else the same action will move is another. Preserving identity through occlusion is a stronger result still.** A score on the first cannot stand in for the other two.

### 3.4 Jointness is not an additional cognitive step

The process need not be:

```text
identify A → identify B → establish an A–B relation
           → assemble a joint state → begin predicting
```

The theory allows a more direct account:

```text
Several kinds of information enter and change the current internal state together.
That state immediately participates in further prediction.
```

“Cup,” “hand,” “contact,” and “push” may be words used later to describe the process. They are not operations that must be completed before prediction can begin.

### 3.5 Notation

The notation below is descriptive. It does not prescribe modules.

| Symbol | Meaning |
| --- | --- |
| `S_t` | The state of reality under discussion; the system need not have complete access to it |
| `O_t` | The partial observation actually available now |
| `Z_t` | The system's current internal joint state |
| `H_t` | The interaction history currently available, including its retained influence |
| `L_t` | Language-related state, one aspect of the whole |
| `A_t` | Action or control output, one aspect of the agent's participation in change |
| `θ` | Relatively lasting parameter structure shaped by training or experience |
| `Δ` | The number of steps, event interval, or duration covered by a prediction |

`O_t`, `L_t`, and `A_t` can overlap as descriptions. Hearing one's own words, for example, is both an auditory observation and part of linguistic activity. They need not be disjoint physical sets.

---

## 4. Prediction with incomplete information

### 4.1 There is no need to receive a complete state first

The theory does not hold that a person must first gather all the information and then compute on a complete world state.

A small amount of input can already give rise to an expectation. As more input arrives, that expectation can continue to change.

Seeing only the corner of an object may lead one to expect something about its unseen parts. Moving one's gaze brings new evidence that may support, alter, or overturn that expectation.

The sequence is not:

```text
complete observation ends → prediction begins → action begins
```

The processes overlap:

```text
partial input ↔ expectation of what follows ↔ new input ↔ revised expectation
                          ↕
                 own action and new feedback
```

This is a sketch in words, not a requirement that a real system visit these boxes in a fixed order.

### 4.2 Predicting something is not completing the future

When a cup has just slipped off the edge of a table, predictions can vary considerably in what they specify:

| What is predicted this time | What need not also be predicted |
| --- | --- |
| My hand is about to reach out. | There need be no rehearsal of how the cup falls or whether I catch it. |
| The cup will go roughly downward; I will soon feel its weight on my hand. | There need be neither a clear image nor an exact time and place of contact. |
| Where the rim will pass, when the fingers will touch it, or even a short sequence of images. | A detailed prediction of this event need not assign a future value to everything in the background. |

These are not three steps every prediction must pass through. If the system predicts only that it will reach, there is no reason to assume it secretly generated an entire future and simply did not decode the image. **“Only the reach” concerns scope; “roughly downward” concerns precision; “not an image” concerns representational form.** These can vary independently. A detailed visual prediction and a direct action prediction are both permitted cases.

Prediction may also concern information not yet obtained about the present: what will be revealed behind an occluder, or what one will hear when someone finishes a sentence. The question is whether subsequent observation or action can check what was actually expected, not whether the system can reconstruct every detail of the world.

### 4.3 A real change and a prediction of change are not the same

Distinguish:

```text
What actually happened: S_t → S_(t+1).

What the system predicted: on the information already available,
it formed an expectation about a possible continuation.
```

Writing `Z_t → Z_(t+1)` only says that the system changed internally. It does not yet show that the change was a prediction.

For a particular implementation one must say how internal activity relates to an observation not yet available or to later behavior by the agent, how feedback bears on it, and how the relation supports performance in an unfamiliar situation.

Otherwise the account collapses into “all systems undergo change” and ceases to distinguish intelligence.

### 4.4 Current operation and longer-term learning

Changing the current state is not the same as changing the organization learned over time.

Hearing a new sentence changes what can be used to predict now. Repeated experience of a kind of interaction may change how the system predicts the next time it receives similar input.

```text
Current operation: new input changes Z_t and influences what the system does next.

Learning from experience: feedback changes θ, altering future predictions
under similar conditions.
```

Taking new information into account now is not evidence that a model has updated its parameters. The theory requires a continuing influence of experience and a means of forming predictions, not parameter training on every state change.

---

## 5. What implicit structure is, and what it is not

### 5.1 Implicit structure is reusable predictive organization

“Implicit structure” refers to an internal organization formed through experience: situations that differ but have something in common can give rise to corresponding predictions.

A system that has repeatedly experienced objects disappearing behind occluders and returning may form an appropriate expectation about an unseen part from a fragment of a view.

One need not begin by supposing that the system has stored a rule reading “objects persist.” One can first investigate how its state supports that performance.

“Implicit” does not mean structureless. The organization must be reusable, sensitive to context, and able to work in new combinations.

### 5.2 Do not put causation and logic back into the implicit level by assumption

The strong position here is:

> What is implicit in the first instance is a joint state and its predictive transitions, not a hidden collection of causal relations or logical propositions.

“The model has simply hidden a causal graph in its vectors” is not a default explanation one can accept without testing it. Nor does a correct answer to a reasoning problem make an independent logical structure an observed fact.

A network can learn computations that specialize. If it reliably tracks objects, compares two remarks, or detects contradictions, these are mechanisms shaped by experience. “Not preinstalled” does not mean “not a mechanism.” A researcher can also describe a nonlinguistic response in logical terms without supposing that the responding system has itself judged a proposition.

The theory's specific wager is that treating a statement as a reason and judging whether a conclusion follows from it take place at the level of language. The same predictive network carries out that activity; no additional logic engine need be installed. This needs evidence, not a new name for an internal computation.

### 5.3 “Relation” as a research term and as a basic mechanism

Researchers can of course explain internal operations using words such as “relation,” “variable,” and “circuit.” The theory itself cannot dispense with such terms.

A word used to describe a process need not also be a cognitive primitive that the system must possess in advance.

“Some of this information changed the later output” is an external explanation. It does not require the system first to state to itself, “There is a certain relation between this piece of information and that outcome.”

The objection is to moving concepts from the level of explanation to the level of mechanism without distinguishing the two, not to the use of descriptive language by researchers.

### 5.4 Implicit does not mean forever unintelligible

On this account, people come to understand a model by observing its states and behavior, forming new predictions, and developing language suited to what they observe.

A lack of experience with an implicit system may leave existing language ill-equipped to describe it. Calling the system “high-dimensional,” “distributed,” or “implicit” is no grounds for declaring it unknowable in principle.

There are three distinct objectives: relating internal changes to behavior, capturing the relation in a concise concept, and reconstructing every computation. The first two are useful without the third being complete.

What matters most is not whether every internal direction has a name. **Does the proposed explanation help us predict and change what the system does, or does it merely redescribe its output?**

---

## 6. Learning, generalization, and knowledge

### 6.1 Learning is not registering an answer for every situation

A system might remember many complete situations and still fail as soon as one is slightly changed.

What matters here is not the number of situations acquired but whether different situations jointly shape a reusable organization for prediction.

```text
many actual experiences
    → the same system repeatedly predicts and receives feedback
    → its internal organization of transitions changes
    → useful expectations remain possible in new situations
```

Generalization is an outcome and an aim of learning, not a separately installed “generalizer.”

### 6.2 Why variation in the data matters

A model that has seen only red balls moving downward may bind color to the motion that follows.

Varying color, direction, background, occlusion, and operation does more than increase the number of complete scenes. It changes which ways of predicting can continue to work.

The aim is not to randomize every variable. Experience needs to expose differences that matter: what varies only in the background, what accompanies a genuine difference in what follows, and which previously successful predictions do not transfer.

Diversity does not guarantee the structure we want. Gaps in the data, the training objective, and how capacity is allocated may still lead a system to rely on unstable cues. These are problems for an implementation to address.

### 6.3 Compression may describe a result; it need not be an extra premise

An early draft spoke of “compressing reality and then forming concepts.” The author's later revision does not require this sequence.

It is not necessary to suppose a compression faculty first summarizes whole events into concepts and passes them to a predictor.

> In the course of prediction, the system develops internal organization that can be used again. An observer may call some of the results compression, but prediction need not first undergo a separate operation of compression.

Likewise, “seeing a little and anticipating more” can be partial information giving rise to further predictions. It need not mean that a whole event was once stored and is now retrieved from a compressed archive.

### 6.4 Where is knowledge?

Functionally, knowing something can mean being able, in the relevant circumstances, to form an appropriate expectation, produce a linguistic continuation, or act appropriately.

Not every item of knowledge must therefore be stored as a proposition in a searchable list.

Putting knowledge into language nevertheless matters. Language gives a system access to events it has not experienced, allows it to compare accounts, and lets one linguistic state give rise to further predictions.

Knowledge can be expressed in a skilled action one need not put into words, or in symbols that can be shared and fed back as input. The two need not be products of fundamentally different kinds of intelligence.

### 6.5 What emergence means here

When learned predictive structures can work together in a new situation, the system may become capable of a task it could not previously perform reliably.

Experience leaves behind not just “last time's answer,” but potentially a way of processing similar information in the future. A pattern of activity may preserve a cup's identity during one occlusion; only experience across different occlusions might turn that behavior into a transferable ability. The manuscript calls a division of function *emergent* when it is learned rather than specified function by function. This is about an individual's experience changing a network, not learned connections being directly inherited. The connection biases that evolution and development supply are a separate level of explanation.

An origin in predictive training still requires specific evidence. The appearance of an ability does not by itself show that it came entirely from one training factor. Nor should every change in performance be attributed to a mysterious threshold of scale.

---

## 7. How language can arise in practice

### 7.1 A sound is first of all part of what happens

The proposed starting point for language is not a subject that first understands the world and then decides to encode it in sounds.

In ordinary life a person's vocalizations, another person's actions, changes in an object, bodily sensations, and social responses already occur together. To a predictive system these are first of all parts of experience.

An observer might describe an episode as:

```text
seeing something + hearing a sound + seeing someone reach
                 + feeling something in one's own body
```

The system does not first have to know which element is “language.” It continues to predict in a situation that includes them all.

### 7.2 Co-occurrence can establish predictive links in both directions

In repeated shared activity a situation may make a sound expected; hearing that sound may in turn make a later sight, action, or bodily sensation expected.

Take a word that already exists, “water”:

```text
a cup, pouring, the sound “water,” thirst, what drinking feels like
```

Different combinations can occur across many episodes. From some of the elements the system predicts others, or what follows them.

This is not first forming a complete concept of a “water event” and then attaching a word. The sound was in the event all along.

The example illustrates how a predictive link may form. It does not assert that the first human language began with a ready-made vocabulary. The stronger proposal about origins is that vocalizations with no initially fixed meaning can, through repeated interaction, acquire a stable role in particular predictions.

### 7.3 A sound can acquire a symbolic function without standing in for all of reality

A sound comes to function as a symbol insofar as, in context, it gives rise to expectations concerning other states.

Even without water in sight, hearing a sound associated with water may be followed by looking for a cup, waiting for someone to pour, or describing a scene.

An observer might say “the sound stands for water.” The more basic explanation in this theory is that **the sound can now enter a predictive process shaped by earlier experience.**

The claim is not:

```text
one word = a summary of a whole event = a substitute for the whole of reality
```

Rather:

```text
A word is one element in a joint state.
In the present context, that element can give rise to many further predictions.
```

It can be described as a cue, an association, or a partial proxy without containing or replacing the whole event.

### 7.4 Why the same word can lead to different things

If meaning arises from what an element does in a joint state, a word's effects should not be fixed independently of context.

Hearing “water” when thirsty may lead to drinking. During a leak it may lead to shutting a valve. In chemistry class it may be followed by a molecular formula; in casual conversation it may enter a metaphor.

The theory does not require one internal entity that exhausts every use of the word. The same sound or written element enters different joint states and therefore leads to different expectations.

This is also why memorizing a word in isolation is not the same as mastering its uses in practice.

### 7.5 What happens between people

Language is not the unilateral achievement of one isolated individual. One person's utterance enters another's situation; the second person's response becomes part of what happens next for the first.

```text
One person vocalizes in a particular situation.
Another acts, responds, or makes a different sound.
The first person receives new feedback.
Similar exchanges recur in other settings.
```

On this account, sounds and patterns of behavior that participants can repeatedly take up become more stable. The participants do not first finish designing a linguistic structure together and only then put it to use. The structure takes shape in use and prediction.

Stable does not mean permanent. Expressions can change as groups, circumstances, and later experiences change.

### 7.6 Not only sound

Speech is the most immediate example, but vocal cords do not constitute the essence of language in this theory.

Gestures, facial expressions, visible marks, writing, and other signals that can recur in interaction can take part in comparable processes.

The critical property is not that the material be sound. It is that an element can recur, give rise to predictions, and come to be followed by other elements of its kind.

### 7.7 In what order does language form?

The theory rejects the following goal-directed story of origins:

```text
understand reality in full → decide to summarize it → design symbols
                          → express thoughts with those symbols
```

Its alternative is:

```text
Sounds, actions, and perceptions are already present in what happens.
They enter prediction through shared experience.
Certain sounds and marks begin to elicit stable expectations.
The ways they can be followed by other sounds and marks grow richer.
Language takes shape through this activity.
```

Symbolic function is a result of the process, not a design that must precede it.

---

## 8. How language can sustain a predictive process of its own

### 8.1 From sound and world to sound and sound

Initially a sound may occur mainly in a concrete situation and lead to a corresponding activity. As communication continues, the sequences of the sounds themselves acquire continuations that can be learned.

The system can then expect a sound from a sight or another sound from the first. Written words can evoke a scene or lead to more writing.

```text
partial information about the world ↔ sound or symbol
sound or symbol ↔ another sound or symbol
```

“Predict one another” does not make two words agents doing the predicting. One system, in different contexts, can use either element to form an expectation about the other or about what follows it.

### 8.2 Language the system produces becomes new input

When the system utters a sentence, or continues a linguistic activity internally, what it has just produced changes its current state.

The next prediction depends not only on the original question but also on the words already produced.

```text
present state → produce a fragment of language
the fragment enters the new state → produce a further fragment
that fragment in turn enters the following state → the process continues
```

Viewed from outside, this can appear as explanation, association, reasoning, writing, or asking and answering one's own questions. The theory does not require a new basic operation for each phenomenon.

The important change is that **the result of one prediction can become a condition of the next.**

### 8.3 Running on its own does not mean leaving reality behind

Language can continue for a while without the corresponding scene unfolding at that moment. That does not mean it has never been shaped by the world.

When one says “inspect the equipment at the warehouse tomorrow,” neither the warehouse nor the inspection need be in view. How the words fit together nevertheless has to do with earlier experience, conversations, operations, and feedback.

Nor is language an entity outside the physical world. Speaking, writing, reading, and internal activity remain changes of state in an actual system.

A “world of language,” then, is a range of symbolic states in which prediction can continue. It is not an ontologically separate universe.

### 8.4 Why situate higher intelligence at the level of language?

The theory holds that the essence of higher intelligence is the capacity of a language system to continue operating on its own, not the physical brain or computer as such.

Language allows prediction to return repeatedly to symbolic states that can be preserved, shared, and taken up again. The external event need not be experienced anew each time.

A question can be stated and its answer become the condition for a new question. A hypothesis can be kept in play while several steps unfold around it. Someone else's experience can enter the present system through language.

The author's claim is not that language expresses judgments that already existed more clearly. It is that **higher intelligence exists only at the level of language. The basic function remains prediction. It is when predictions concern statements, reasons, objections, and the responses that follow that causal explanation, logic, and judgments of right and wrong become things the system does.** Language need not be voiced. Signing, writing, or mathematical notation can serve this function when they carry these contents and allow them to be taken up. That does not make every internal code a language.

Consider causal relations. The author's position is that there are no causal relations as such in the world; causal relations exist in linguistic constructions such as “because ... therefore ...” that express them. Other logical inferences, and mathematics too, belong on this view to the linguistic level rather than to the world as encountered. Yet because the language level arises from the world, higher intelligence operating in language can describe that world accurately.

Seeing a cup tilt, reaching to catch it, even catching it with the other hand, can all be flexible nonlinguistic predictions. A different activity begins with “Did I knock it over, or did the table shake? If I never touched the table, does my earlier reason still hold?” An explanation has itself become something that can be retained, challenged, and withdrawn. The distinction is not the difficulty of the movement or whether the answer is spoken or indicated by hand. Whether this second ability requires language is an empirical question. Defining “higher” as “spoken aloud” would not answer it. Section 15 describes a test.

### 8.5 Greater complexity does not mean greater correctness

Language can generate new combinations of language. Continued generation does not guarantee either more knowledge or more accurate conclusions.

The same recursive process can repeat itself, wander away from the question, contradict itself, or elaborate a false premise indefinitely.

“The more prediction, the greater the complexity” is better understood as the possibility of forming more complex states through linguistic activity. Which complex structures remain useful depends also on feedback from practice, correction by others, formal constraints, and training experience.

Generating more content and producing more reliable knowledge are different achievements.

### 8.6 Two sources of limits on intelligence

The author's later account distinguishes two kinds of limitation.

One lies in the predictive organization itself: can the system retain the influence of a sufficiently long history, handle combinations of conditions, and form reliable expectations in unfamiliar situations?

The other lies in the linguistic state space: can the available expressions distinguish the problem, preserve intermediate results, and support further operations on them? A new symbol or notation can make it easier to continue working on what was previously hard to keep in play.

Improving intelligence may therefore involve more than increasing internal capacity. It might also involve changing the language, symbols, or forms of interaction available to the system. This is a research direction proposed by the theory, not a performance guarantee for any particular model.

---

## 9. Where do causation, logic, and reasoning belong?

### 9.1 The central position

One of the theory's most distinctive claims is:

> Anticipating what will follow an action is not the same as judging “it is the cause.” The theory places articulable causation, judgments about what follows from what, and judgments of right and wrong at the level of language. Implicit joint state transition prediction still supports these activities.

This differs both from “there is already an internal causal model which language translates” and from “the system first completes an inference by logical rules and then puts the answer into a sentence.”

The aim is to explain these higher abilities, too, as prediction continuing in linguistic states.

### 9.2 How causal expressions enter prediction

Through repeated experience and communication, people encounter particular actions, changes, and expressions together.

A switch changing position, a light changing, someone's explanation, and the next operation can all enter the same learning process. Expressions such as “switch,” “it lit up,” “because,” and “therefore” take part in prediction in the course of practice.

Later, without a lamp in front of them, people can explain these situations and anticipate what would happen under different operations.

The point is that language works together with practice. The system does not have to locate an internal object called “causation” before the corresponding expressions can form.

### 9.3 Temporal regularities and interventions remain different questions

Explaining causal language as a product of prediction does not turn every sequence of events into a correct causal relation.

Suppose a system always hears an alarm after a red light comes on. It can learn to predict the alarm from the light. “If only the light were changed from red to green, would the alarm stop?” is a different question requiring experience or assumptions.

Two devices could produce much the same record. In one the red-light circuit triggers the alarm. In the other a common controller drives the light and the alarm separately. Changing the light can therefore have different consequences in the two devices.

Saying that both tasks involve prediction cannot erase their difference. The theory has to explain **what kinds of interaction allow the same predictive system to form appropriate expectations in these different cases.**

It may do so without a prespecified causal graph. Merely calling every observation a state transition does not solve the problem.

### 9.4 How a logical continuation can work

Consider the symbolic case:

```text
Every A is a B.
x is an A.
Therefore x is a B.
```

On this account a system can acquire this stable continuation through linguistic experience, symbolic operations, and correction. It need not call a general-purpose logic engine each time.

That is not a claim about memorizing a fixed sentence. What needs explaining is whether the system respects the relevant rule when names are replaced, wording changes, irrelevant material is inserted, or conditions are nested.

Completing familiar patterns but failing on new combinations would demonstrate local linguistic prediction, not adequate logical generalization.

### 9.5 Frequent output is not valid inference

Logical validity cannot be defined as “this sentence often follows that one.”

```text
If A, then B.
B is true.
Therefore A is true.
```

The passage may sound fluent, yet its conclusion is not guaranteed by the premises. A case in which B is true and A is false suffices to show why.

The theory must explain how a predictive system learns linguistic behavior that respects such distinctions. It does not abolish the distinctions.

**A predictive explanation of how reasoning develops does not redefine the standard of correct reasoning as the probability of an output.**

### 9.6 What must an account of the emergence of logic explain?

The proposal is that logic can arise through linguistic practice and prediction rather than being prespecified as an independent mechanism. A scientific account still has to explain:

- why the relevant constraints continue to hold with new symbols and combinations;
- why a change of premises should produce a change of conclusion;
- how particular continuations give rise to more general forms that can be used reliably.

“The linguistic elements started to predict one another” cannot substitute for answers to these questions.

### 9.7 There is no need to claim that reality violates logic

“Logic need not be prespecified as an underlying mechanism of intelligence” does not mean that reality is unconstrained, or that all logical relations are invalid.

The theory can regard linguistic relations as structures formed in use while acknowledging that different expressions face different constraints. Empirical claims answer to observations; formal deductions must conform to the relevant definitions and rules.

Explaining how the constraints arise is not denying that they exist.

---

## 10. Action, goals, and exploration

### 10.1 One's own action can be the entire content of a prediction

The author's central formulation about action is:

> Action is part of the model too: it consists in predicting what one is about to do.

Intelligence is therefore not split into a predictor that only watches the world and a decision maker standing outside it.

Posture, changes in front of the agent, the effects of past behavior, and sometimes a sentence may all influence the present prediction. What is predicted next can be no more than “I will raise my hand.” The person need not see a mental picture of the hand rising, judge in advance that raising it is right, or be aware of having predicted it. This is the author's hypothesis about human and animal action, not a fact one can read off the shape of a movement.

### 10.2 How a prediction participates in an actual movement

The shortest closed loop might be:

```text
present view, body, and experience → predict my hand will rise → motor pathway raises it
                ↑                                                      │
                └────────────── fresh visual and bodily feedback ────────┘
```

Predicting one's own movement through a motor pathway is not the same as imagining another person's hand rising or supposing “what if I raised my hand?” The latter can remain internal. Context, network activity, and output pathways must implement the distinction; there need be no inner observer to approve every movement.

The same network might anticipate that pulling A will move a battery compartment and allow that expectation to influence its next action. It might even unfold and compare several relatively detailed futures. **Checking whether a consequence fits a goal is one route to action that can be studied in its own right. It is not a gate through which all actions must pass.** Conversely, an output of “pull A” alone establishes neither that the model rehearsed the outcome nor that it used no expectation at all.

### 10.3 A goal is not a commander outside the system

A goal can enter the joint state as language, a bodily condition, a memory, or in another form.

“Bring the cup over here,” the cup's position, experience of previous operations, and the current state of the body can all participate in predicting what happens next.

The theory requires no independent “I” standing outside the whole system, comparing options, and issuing a decision. Acting “for the sake of a goal” can be an external description of a predictive process that includes goal-related information.

Putting a goal into the input does not guarantee that the system will achieve it. **How the presence of a goal changes the prediction of one's next action depends on experience and on the way feedback is organized.**

### 10.4 The most probable action is not necessarily the best one

This is a central difficulty for the account of action.

If failure is common in a given training situation, reproducing the most frequent subsequent behavior may reproduce the failure. Predicting “what I usually do” is not automatically the same as predicting “what would succeed this time.”

The theory must explain how these can be connected within one framework. Successful episodes, corrective feedback, goal conditioning, and selection pressures in training might change which continuations the system comes to produce reliably.

Such changes may still be implemented by altering predictive organization. But one cannot jump from “an action is a state too” to “the action will be correct.”

This is an open mechanistic question, not proof that an independent subjective decision maker must be installed.

### 10.5 Danger and exploration

Fleeing danger need not be explained by an extra subject first forming the proposition “I want to survive” and then ordering flight.

Information about danger, changes in the body, past experience, and subsequent movement can take part in one ongoing predictive process. An observer sums it up as “acting to avoid danger.”

Exploration can also be considered in this framework. An unfamiliar situation, earlier experiences of exploring, bodily state, and possible feedback can together influence the prediction of “what I will do next.”

The account must still explain which experiences produce sustained exploration rather than avoidance or inactivity. Calling the behavior a prediction does not explain why it occurs.

### 10.6 Planning need not be an independent basic mechanism

A sequence of predictions involving future language, one's own actions, and environmental changes may appear as planning at the level of behavior:

```text
present situation → prepare tools → reach the location → do the work → get feedback
```

An engineer can assign this to a module called a planner. The theoretical question is whether it must be a basic mechanism of intelligence distinct from prediction.

The theory's answer tends toward no. Actual tasks must nevertheless show that a predictive process can maintain a goal, cope with failure, and continue over longer horizons.

---

## 11. Thought, self, and consciousness

Three questions have to be kept separate: what feeding predicted content back might do; how such feedback might occur in a human body; and what the author infers about self-awareness. Establishing the first does not establish the other two.

### 11.1 Thinking as the continued participation of internal states in prediction

A system need not wait for the external world to take one step before it takes each internal step.

Self-generated language, expectations of images or actions, and the influence of memories can all become part of a subsequent state and give rise to further content.

On this account thought is not an additional layer outside prediction. It can be the ongoing process in which internal products become conditions for subsequent prediction.

### 11.2 An inner-speech hypothesis: returning predicted content to the sensory side can sustain a thought

Silently say “My keys are still in ...” No one has spoken into your ear, yet the fragment can bring forth “my coat pocket” as if you had just heard it. The author's conjecture is that a sound, image, or bodily sensation predicted in the brain is passed out of the activity that generated it and back to the sensory side. There it can participate like new input in the next prediction. A line of thought can thus continue beyond the brief activity that initiated it. Calling the inner voice an “illusion” does not deny the experience of seeming to hear. What need not exist is another “I” inside who speaks and listens.

Two questions are not interchangeable. **Functionally**, can the product of a previous step be used again as input to extend thought? One can investigate this in an artificial system by connecting and interrupting a loop. **Physiologically**, does the proposed return actually pass through the corresponding receptors? Returning to sensory pathways in the brain is different from reaching peripheral receptors in the eyes or ears and then re-entering the nervous system. The latter claim requires physiological evidence that distinguishes the routes. A recurrent model that can continue a sentence does not show that humans use either route.

### 11.3 Self-related content can itself be predicted

“Who am I?”, “What did I just do?”, and “What will happen to me?” can all occur as linguistic or nonlinguistic content in the current state.

The theory accordingly does not posit an extra subject detached from the transitions of its own state. Descriptions of the self, expectations of one's behavior, and statements about the reliability of one's predictions can arise within one system.

The author takes a stronger position: **“Human beings do not possess self-awareness.”** In his account, sounds, images, and sensations ordinarily attributed to an inner “I” are the contents of prediction and feedback. There is no additional self that receives and directs them. Representing “this is my hand” or “this is what I just did,” or reporting inner speech, is distinct from granting the existence of an independent subject.

The boundary holds in the other direction too. Explaining an action without an inner decision maker does not prove that self-awareness does not exist. The strong thesis has not been quietly replaced with “a control system does not need a central controller,” nor is it the result of an experiment on humans. An action model's performance cannot tell us why pain or first-person experience exists.

### 11.4 Do not put the explanatory subject back in

It is easy to say “the system knows what to do, so it chooses the best future.” That can summarize a behavior, but it cannot take the place of a mechanism.

The question throughout is how the present state, the structure left by learning, and continuing input give rise to the next words and actions.

Introducing a self that “really understands and decides” whenever the explanation gets difficult would reintroduce exactly what the theory set out to explain.

---

## 12. Science and reality

### 12.1 In what sense “language is the world”

“Language is the world” here means neither that language creates reality at will nor that people can experience nothing outside language.

The point is that when people explicitly state what an object, relation, condition, inference, or theory is, much of the world they can discuss together and continue working with has already entered an activity of language and symbols.

Language is not a container suspended above the world. It originates in activity and remains constrained by what happens in subsequent practice.

Both statements can therefore hold:

```text
Language affects how people can go on describing and predicting the world.
Feedback from reality affects which expressions remain useful.
```

### 12.2 Internal structure need not be a replica of reality

A predictive system can work on a limited task without reproducing, component by component, everything of which its target is made.

The theory does not require an exact structural identity between an implicit state and reality, or a corresponding elementary entity in reality for every concept in a language.

Not requiring complete isomorphism is not the same as saying no structure in reality can be retained internally. Consistent performance on some tasks may require the system to retain relevant differences and constraints.

The mistake to avoid is equating successful prediction with discovery of the ultimate constitution of reality.

### 12.3 Science is not an arbitrary language game

When two accounts give different expectations about one question, both can be brought up against the same observations, operations, and results.

Words can be combined freely. Reality does not have to cooperate.

Science, in this framework, keeps bringing predictions made in language back into public practice. Statements, conditions, operations, outcomes, and revisions can be taken up by other people.

```text
actual experience
    → continued work in language and symbols
    → expectations that can be brought to the world
    → observation, action, and feedback
    → earlier expressions and predictive methods are retained, restricted, or revised
```

This explains where scientific work fits in a cycle of prediction and language. It does not dispense with experiments, measurement, and proof in favor of fluent expression.

### 12.4 Mathematics and science answer to different constraints

Within given definitions and rules one can ask whether a mathematical derivation holds. An empirical scientific statement also answers to observations.

The theory can try to explain how people learn to respect these constraints in symbolic prediction. It cannot rewrite both as “it occurs often in the world, therefore it is correct.”

A mathematical argument takes place in a symbolic space; its correctness is not determined by the frequency of its sentences. Likewise, telling a coherent scientific story does not establish that its expectations about the world have been met.

### 12.5 The theory must allow itself to be revised

JSTP is itself an account in language. Offering an explanation of the origins of language does not exempt it from clear definitions, empirical comparisons, or counterexamples.

Its value should lie in whether it distinguishes models, helps organize training, suggests experiments, and explains phenomena previously hard to bring together.

If every possible result can be renamed “prediction after all,” with nothing that could change a practical judgment, the account remains a broad descriptive framework rather than a theory that distinguishes among possibilities.

---

## 13. Architectural implications for unified AI

This section develops engineering ideas from the theory. It does not claim that any existing model has validated the theory in full.

### 13.1 Different models can predict states in different forms

On this account, the first differences among language, image, video, 3D, and action models lie in what data they encounter and which subsequent content training constrains.

| Training material | Its place in the theory | What this training alone does not establish |
| --- | --- | --- |
| Language, code, and formal symbols | Learning how symbols continue and constrain one another in context | Mastery of physical operations never encountered |
| Still images and corresponding conditions | Learning the structure of visible states and conditional associations | Learning motion and feedback in real time |
| Video and sequential perception | Learning how observations and latent states change with time | Reliable prediction of the consequences of an action |
| 3D scenes, depth, or Gaussian representations | Providing spatial states and observable forms of those states | A static representation is not itself a transition model |
| Screen image, task, and next executed action | Learning to predict one's action directly from the situation, without a next-frame target | Action success alone cannot establish expectation of an external consequence |
| Interactive trajectories containing actions and their outcomes | Also learning what a given operation will move after different histories | Predicting an outcome does not guarantee the action will use it |
| Continuing multimodal interaction | Allowing different kinds of content to inform predictions of one another | Simply mixing data does not establish effective unification |

This is not a fixed ranking of technical approaches. One unifying principle may lead to different architectural choices for different tasks.

### 13.2 Work backward from the ability to the data

The author's later design principle is to specify the desired ability first and then ask which changes of state the system should experience.

To study reasoning in language, give it experience of continuing, challenging, and revising statements. To learn the next click in a familiar window, screen images and actual clicks supply the relevant signal. To know whether the model expects a particular window to open, one also needs outcomes against which to check that expectation. The same applies to machine operation: imitating a joystick movement and anticipating resistance at a bucket are different training problems.

“They are all prediction” does not make the data interchangeable or every algorithm equally suitable for every task.

> The explanation of the mechanism is unified. The actual differences in data, representations, and training conditions remain.

### 13.3 What a shared latent space means

The unified model envisaged here is more than a language model sending text commands to a separate world model.

A deeper form of unification lets visual, linguistic, bodily, memory, and action information change subsequent predictions together. No kind of content need be permanently isolated in another system accessible only through a one-way interface.

```text
information available now + effects retained from earlier interactions
                              ↓
                       shared latent state
                              ↓
this prediction: an action, a sentence, a sensation,
or several of them together with changes in the world
                              ↓
executable outputs act on the world; new observations enter the next prediction
```

“Shared” means that experience can change how the same network works later. It does not require a fixed-length vector or a hidden film from which all modalities could be recovered. A probe can leave the expectation “these two parts will move together.” If in the next episode it is used only to predict which handle to pull, the system need not also output images of both compartments moving.

### 13.4 Unified does not mean homogeneous

Vision, language, and control signals may have different encoders and decoders. A model may use multiple scales, local subspaces, and different interfaces.

These engineering distinctions need not imply fundamentally different intelligences. The question is whether the components participate in a jointly learnable predictive process and whether information from one really affects subsequent states in another.

An action head is not in itself a failure to unify. Conversely, putting everything into tokens does not by itself achieve unity.

Neither module names nor the way data are packaged is a sufficient criterion.

### 13.5 Prediction is not limited to the immediately following step

“What follows” can cover different timescales. Training may constrain the next observation, a later phase of a task, missing content, or a later correspondence across modalities.

```text
current screen and operation → next interface change
current task and history → subsequent action sequence
current fragment of language → subsequent words and expectations about a scene
partial observation → what further observation might reveal
```

These possibilities let training cover the abilities claimed by the theory. They do not add a mysterious long-range faculty to its basic mechanism.

Long-horizon prediction also faces accumulating error and uncertainty. It should not be described as necessarily unfolding a single precise future.

### 13.6 Real tasks must constrain a latent state

If the only objective is to predict an internal vector the model generates itself, a system that always emits the same state may do well on its own criterion.

“Predict the latent state” cannot be a sufficient objective by itself. Actual observations, continuations in language, results of actions, or other explicit tasks must require the state to retain useful information.

Different objectives can impose these constraints. They should all serve to distinguish what actually follows, rather than reward a loop that is merely easy for the model to satisfy.

### 13.7 Where 3D and Gaussian representations fit

A 3D representation, a Gaussian representation, an image, or a depth map can be an input, a supervisory target, or an output describing the world.

None of these is intelligence itself, and none need be the sole format of a shared latent state.

Requiring a model to represent present and future scenes during training may constrain its internal predictions. Whether the whole scene must be reconstructed at run time depends on the task.

Nor can a plausible-looking future scene be taken as an accurate mechanical or control outcome on appearance alone. The required prediction—image, position, force, or task success—must be specified in training and evaluation.

### 13.8 Steps of a generative algorithm need not be time in the world

Different computations can implement prediction. This does not make every internal iteration a step in world time.

A generator might progressively refine a coarse result into a sharp one. Its refinement steps need not correspond to successive real events.

“The algorithm changes state,” “the algorithm generates a state,” and “the algorithm predicts what will happen next in an actual interaction” should therefore be distinguished.

This distinction strengthens a unified account: it makes clear which training constrains changes in the world and which chiefly constrains the form in which a state is represented.

---

## 14. Two worked examples

These are hypothetical scenarios to illustrate the theory, not completed or validated experiments.

### 14.1 Desktop operation: a screen image may predict only the next click

The user says, “Move this file to the destination folder.”

The current situation includes the words of the task, the window on the screen, the cursor position, the history of operations, and the system's experience of the interface.

The theory need not posit an independent planner that first translates the request into a complete plan and passes it to a fundamentally different action intelligence. The system can predict its own subsequent operations from the joint state while continuing to receive feedback.

```text
task wording + current interface + interaction history
    → predict the next drag of the file (possibly the only output) and execute it
    → the actual interface changes
    → the new screen and the original task jointly enter the present state
    → predict what the system itself will do next
```

An unexpected dialog should change subsequent predictions in time to matter. The system should not blindly execute an initially generated sequence.

If two folders look much alike, the system might instead expect “when I release the button, the file will appear on the left” and use that expectation to adjust where it drops the file. Both routes fit the theory. The first does not cease to be prediction because it draws less of the future. The second does not become more reliable merely because it draws more.

Whether the file was moved correctly must still be judged from the result. Describing the right steps is not the same as completing the task in a working interface.

### 14.2 Loading material: a latent space includes more than images

Consider the task “collect material from the pile and deliver it to the designated location.”

The current state may include the task wording, camera view, machine posture, bucket position, control input, force feedback, and preceding motion.

As resistance rises, the operator might predict only “my hand will ease back a little,” or might expect the bucket to be forced upward. Neither requires calculating the path of every stone. A coarse expectation will not always suffice, however. To distinguish a cutting edge striking a hard base from a more tightly packed pile, experience must contain cues that distinguish the two.

A model need not start with an explicit “law of granular causation” in order to learn what follows insertion of the bucket. Nor does exposure to similar-looking video suffice to establish mastery of the actual forces and operating conditions.

Training on appearances alone may principally yield expectations about appearances. To bring force, resistance, payload, and subsequent control into the same predictive system requires corresponding observations, interaction, or supervision.

**Unification at the level of the proposed mechanism does not remove a task's need for specific information.**

---

## 15. Testable claims and experiments that distinguish them

Sections 15.2–15.6 pose five research questions, not five completed experimental proofs. Section 15.8 separately reports existing frozen-model experiments and indicates which small parts of those questions they address.

### 15.1 Turning a claim into a comparison

“Every ability is just a state transition” is too broad to support a useful comparison.

More discriminating questions ask which abilities a predictive system can acquire without a particular prespecified higher cognitive module, under a stated dataset, resource budget, and task; which abilities depend on linguistic interaction; which benefit from a shared latent state; and under what conditions they fail.

Criteria for an ability should be specified before the result. We cannot call every success “prediction” after seeing it and attribute every failure to insufficient scale.

### 15.2 Question 1: Can signals acquire a symbolic function without prespecified meanings?

Let several predictive systems operate in one environment and emit signals the others can receive. Do not specify which signal must denote which object or give them a complete dictionary.

Observe whether signals acquire a stable function in repeated interaction, whether they elicit appropriate subsequent behavior when an object is temporarily out of view, and whether their use transfers to new scene combinations.

Compare random signals, signals whose temporal correspondences have been shuffled, and signals with normal interactive feedback.

If a signal triggers a fixed response only in training scenes and does not acquire a reusable function across contexts, the result does not support the full account of language formation.

Even stable communication would show only that symbolic functions can form under these training conditions, not establish the entire historical origin of natural language.

### 15.3 Question 2: Does judgment require language?

Compare systems with similar worldly experience and comparable training resources. One group can receive and produce sequences of symbols that can be fed back; the other lacks that condition.

In addition to new combinations, retention of conditions, and multistep tasks, a more discriminating problem is possible. A system initially chooses between two explanations on the strength of a clue. New evidence undermines its reason without changing the immediate payoff of either operation. Does it retract only conclusions that depended on the discredited reason, preserve unrelated ones, and transfer that distinction to a new situation? It can respond by pointing or acting. It need not speak, and imitating a sentence containing a negation should not earn credit.

An advantage for the group with continuing language supports the usefulness of language under those conditions, not its necessity. If the other group receives no linguistic input, acquires no identifiable linguistic system, and nevertheless makes these distinctions reliably, the author's necessity claim faces a counterexample. Before the experiment, specify how linguistic activity will be identified and how a memorized action sequence will be ruled out. Do not wait for the control group to succeed and then call everything inside it language.

### 15.4 Question 3: Does a unified state outperform loosely connected systems?

Compare a jointly trained system in which information can enter prediction in both directions with a system whose parts chiefly exchange finished outputs through fixed interfaces.

Match access to data, compute budget, and output requirements as closely as possible. Examine correction across modalities, missing observations, adjustments after a change of verbal instruction, and transfer to new situations.

An advantage would support that particular unified design under those conditions. Without an advantage, examine whether sharing created competition for capacity, information loss, or conflicting training signals.

The theory should declare neither that every scheme with shared parameters is optimal nor that all modular designs are excluded.

### 15.5 Question 4: Is succeeding at a task the same as actually using consequences to act?

Specify at least two conditions that can vary independently: **whether training requires prediction of external consequences, and whether a decision at execution time can use such an expectation.** Learning the next action from a screen, goal, and history is an action-prediction-only control, not a “non-JSTP” system. A second system can learn both actions and consequences within the same weights while retaining a direct action path. Match visible experience, capacity, and training budget as closely as practicable.

Freeze the weights. First fix a single probe history, change only the previously unseen goal, and execute and score the choices separately. That tests whether one experience can serve different tasks; it does not show that consequences entered the action. Next fix the real experience, goal, and available actions. Change only activity independently calibrated to correspond to a particular physical consequence, leaving the direct path untouched. If the choice switches in the predicted direction specifically where that consequence calls for a different action, and an unrelated change of equal magnitude rarely does so, there is evidence about the pathway. Deliberately supplying a wrong expectation may turn a correct action into an error. That is task failure but informative about mechanism.

Conversely, an intervention that leaves the action unchanged shows only that the implementation did not exhibit dependence on **the part of the expectation that was changed**. It does not mean the network failed to predict its own action, let alone rule out every implicit expectation.

### 15.6 Question 5: Are separate logical or causal modules necessary?

Under specified tasks and resources, compare predictive learning alone with designs that add explicit logical rules, causal graphs, or specialized planning structures.

If the former achieves comparable results with unseen symbols, novel combinations, and changes of intervention, the explicit modules are not necessary **within that range**.

This does not directly show that the network lacks any organization describable as logical or causal computation. Stronger claims about internal structure require explicit identification criteria and interventions on the system itself.

Likewise, better results with an explicit module do not establish that it is a basic primitive of all intelligence. It may simply suit the available data and task better.

### 15.7 How to avoid indefinite retreat

For each experiment, specify the training conditions, the range over which resources may be increased, the evaluation task, and what outcome would change the conclusion.

A finite experiment cannot prove that an architecture will fail at every conceivable scale. It can refute a prediction made under particular resource limits and substantially reduce confidence in a practical approach.

It is useful to state two kinds of claims at once: a broad mechanistic hypothesis and specific engineering predictions tied to explicit conditions. The latter can receive meaningful feedback on a shorter research cycle.

### 15.8 Which distinctions have already been tested?

The following records come from different tasks. They do not form a single leaderboard.

| Question | What the frozen model did | Inference warranted |
| --- | --- | --- |
| **Does grouping visible regions tell the model which part will move if another part is pushed?** | In ordinary multi-object and motion scenes, V6's direct grouping IoU (intersection over union of visible regions) was 36.28%, versus 23.49% for the strongest nonlearned baseline. It missed the prespecified 50% threshold. On paired trials with identical final images but either one rigid body or two independent pieces, it always predicted “does not follow.” In none of 144 pairs was it right about both cases. [Visible-region measurement](records/v6-grouping.md#可见区域); [paired push trials](records/v6-grouping.md#同一末帧的操作). | There is a local grouping signal. This model did not pass the history-dependent test of action consequences. IoU is not a percentage of human ability. |
| **Without object masks, can a model learn which parts move with a handle?** | In a later motor and symbolic feedback study the main model scored 100% on strict co-motion with familiar wiring and new layouts. With a previously unseen middle-single-load wiring, this fell to 54.28%. [Frozen metrics](../evidence/mechanism/analysis/metrics.csv). | A grouping relation became readable in a particular interaction. Its decline under new wiring limits the claim; general object identity was not established. |
| **Does switching handles for a new goal show that action depends on an expected consequence?** | The late-goal study used 192 independent test layouts and three training seeds. After the same probe the model achieved both opposite new goals on 99.83% of pairs; the direct-only control achieved 100%. In a model retaining its direct path, real experience and goal were fixed, and only four latent directions aligned with future physical readings were replaced. When a switch was called for, the model switched in the donor direction on 59.93% of trials; an equal-magnitude control that preserved physical readings yielded 0.88%. [Protocol and counts](../studies/03-late-goals/README.md); [primary metrics](../evidence/late-goals/analysis/metrics.csv). | Success after a goal change does not identify the route. The directed intervention shows that implicit consequences influenced some choices in this model. No advantage over direct action in task completion was observed. |
| **Does sustained use of internal feedback require language?** | In a finite circuit task, training covered at most four steps and frozen testing eight. Three preset predicates, a holistic code, and a learned three-bit code all achieved 100%; clearing feedback reduced all three to 50%. [Record and code definitions](../studies/02-motor-symbol/README.md). | This task required information to be retained, not those three symbols to be given in advance. It did not test reasons, right and wrong, or argument, still less measure human inner speech. |

These records supply existence evidence or concrete counterexamples. One network can learn a kind of local grouping, and an implicit consequence can affect an action. Neither follows from task success alone. These experiments do not decide claims about the ontology of the world, the necessity of language, or self-awareness.

---

## 16. Present limits and open questions

### 16.1 A transition alone does not explain prediction

Ordinary dynamical systems also undergo transitions. Criteria are still needed: when is an internal state a prediction of information not yet obtained, and what feedback demonstrates that the system learned something that generalizes?

This is a problem of operationalizing the central concept, not a terminological detail that can be put aside.

### 16.2 No separate module does not mean no internal structure

The theory denies that each ability needs a machine bearing its name installed in advance. It does not deny that the same network can acquire different computations. Suppose prediction training without segmentation labels produced a circuit sensitive to border ownership, and intervening on that circuit selectively harmed object prediction. That would exemplify what the theory hopes to explain, not refute it.

Conversely, if tracking can be learned under specified training and resource conditions only when object identities are given in advance, then “the predictive experience in this experiment suffices to produce grouping” has failed. A neural network's ability to represent an object in principle does not make that experiment a success.

### 16.3 Language being helpful and judgment requiring it are different claims

The claim is not that all intelligence needs language. Reaching, evasive movement, tracking, and anticipating an intervention's outcome may proceed without it. The contested question is whether a system without linguistic activity can treat a reason as a reason, object to it, or use it to revise another judgment. Better performance on one task with linguistic feedback does not settle necessity.

“Language” must not expand to mean “any structured internal activity.” Otherwise “higher intelligence requires language” is a definitional repetition rather than a claim about actual systems.

### 16.4 Effective action needs a source of learning

Placing action within prediction provides a unified formulation. Making predicted actions reliable requires an account of how experience distinguishes success, failure, and irrelevant continuations.

The theory need not posit an independent will. It does still have to explain how behavioral preferences, exploration, and the sustained pursuit of a task form.

### 16.5 Evidence for local worlds and evidence about intelligence are not interchangeable

A useful unified predictive system would not establish that heterogeneous local worlds must constitute reality at its deepest level.

Likewise, the philosophical appeal of local worlds would not establish that predictive mechanisms alone suffice for all higher cognition.

The accounts may be compatible and inform each other. Each nevertheless needs its own claims and supporting evidence.

### 16.6 The present status of the theory

This is a hypothesis about intelligence with a definite unifying aim: reduce the number of prespecified cognitive mechanisms and place language, world, and action back in one ongoing process.

Its value does not lie in sounding as though it can explain everything. It lies in whether it guides clearer designs, makes predictions that distinguish among possibilities, and allows feedback to change it.

---

## 17. A minimal formalization and a register of claims

### 17.1 Purpose of the formalization

What follows is an implementation-neutral draft, not the uniquely correct model architecture.

It keeps four notions apart: the actual change in the world, the information the system receives, the update of its internal state, and a prediction of content not yet obtained.

### 17.2 Reality and partial observation

```text
Actual change:
S_t → S_(t+Δ)

What the system receives:
O_t = the partial information available at the present time
```

The system need not possess the whole of `S_t`, and `O_t` need not suffice on its own to determine the future.

### 17.3 Internal state update

```text
Z_t = Update_θ(Z_(t-1), new observation, effects of the system's earlier outputs)
```

`Update` is notation for the system's ongoing activity, not a new higher cognitive module.

Updating can be continuous and need not wait for all types of input to arrive simultaneously. An internal state can retain historical influence without rereading the entire past at every step.

### 17.4 Which part of a change is predicted from a joint state?

```text
Z_t: the internal state formed jointly by current observations,
     the agent's own state, and the influence of history
Y:   the subsequent content predicted on this occasion
q_θ(Y | Z_t): the expectation concerning Y formed from the present joint state
```

Over an interval Δ, `Y` might be only “my hand begins to rise,” or “the cup moves roughly downward,” or “the hand's movement and the cup's position change together.” An image can also be `Y` if there is in fact an expectation of an image. Testing whether the next observation supports “downward” is not testing pixelwise reconstruction of a frame.

The notation does not add a “module for choosing Y.” It requires the researcher to say which prediction is being discussed. Quantities outside `Y` may never have been predicted at all: the system need not generate a complete world distribution and discard the other components. Nor does `q` require an explicit probability table. An action, a rough direction, or a latent expectation that later feedback can check are possible implementations.

### 17.5 Output and re-entry

```text
linguistic output → becomes part of a later linguistic or perceptual state

prediction of one's own next action, connected to the motor path
    → bodily or interface action → fresh feedback enters the state

expectation of someone else's action, a hypothetical action, or an external change
    → need not be executed; later observations can support or revise it
```

This is what “language runs on” and “action is part of the model” have in common: what the system produces does not stay outside it, but enters subsequent states.

### 17.6 Learning

```text
the difference between a prediction and what actually follows,
together with task-related feedback
    → changes the longer-term structure θ
    → changes later predictions in similar and novel situations
```

Parameter optimization is one way of implementing learning in an artificial system. No particular optimizer, loss function, or ordering of training phases is the essence of intelligence on this account.

Nor must engineering use a single numerical loss simply because the explanation has a unifying aim. Multiple constraints on outputs can serve useful joint prediction.

### 17.7 Register of core claims

| ID | Claim | Status within the theory |
| --- | --- | --- |
| H1 | Things are, in their nature, undergoing transitions together. To study intelligence it suffices that an observer can experience parts changing together or separately. | Worldview; the weaker empirical premise can be adopted independently. |
| H2 | Network structure provides the means to distinguish local activity. Connections, ongoing activity, and experience can give rise to object grouping without a prespecified independent segmenter. | Structural and learning hypothesis. |
| H3 | The basic function of intelligence is prediction. Input can be incomplete, and the prediction itself may concern just one change, without first generating an entire future. | Core mechanistic claim. |
| H4 | The crucial product of learning is an implicit organization of transitions that generalizes, not enumeration of situations. | Claim about learning. |
| H5 | Compression need not be installed as a separate basic cognitive mechanism. | Constraint on the proposed mechanism. |
| H6 | Sounds and symbols begin as elements of shared situations and acquire a role in subsequent prediction through practice. | Claim about language formation. |
| H7 | A word need not summarize or replace an entire event. As a local element it can prompt further connections. | Claim about concepts and meaning. |
| H8 | Linguistic products enter prediction again, allowing symbolic activity to continue unfolding on its own. | Claim about higher cognition. |
| H9 | Causation, logic, and right and wrong **as judgments** arise only at the level of language. Nonlinguistic prediction can nevertheless get things right. | Strong hypothesis about higher intelligence. |
| H10 | The system can predict and execute only its own next action, or allow an expectation of an external consequence to affect it. Judgment and planning are not required intermediate steps. | Claim about action. |
| H11 | World, language, memory, and action can inform predictions of one another in a unified but not necessarily homogeneous latent system. | Architectural direction. |
| H12 | The theory must earn support or undergo revision through investigation of specific abilities, resource limits, and internal mechanisms. | Research principle. |
| H13 | The author holds that human beings do not possess self-awareness. Inner speech and related phenomena arise as predicted contents return to the sensory side and prompt further predictions, not because another “I” is listening. | Strong hypothesis about the self and sensory feedback; the physiological route is unconfirmed. |

The claims have different standing. Evidence for one cannot automatically be counted as evidence for them all.

---

## 18. The theory in brief

### 18.1 In one sentence

> **Intelligence predicts some of what comes next from the joint state of the agent and the world: it may predict only its own reaching, or anticipate how hand and cup will move together; on this theory, higher intelligence arises only when prediction continues in language and a reason can bear on a further judgment.**

### 18.2 In a little more detail

The author understands the world in terms of the transitions of things together. A cup, hand, and tabletop participate in one process without always moving as one. Units of a neural network that can vary separately yet remain connected give the system a way to express this difference. After experiences of grasping, releasing, and occlusion, the network may learn which parts to track together and which to update separately. Object grouping is a way of working the network can acquire, not a finished product to be delivered before prediction begins.

Prediction needs neither a complete input nor a complete future. When catching a cup, a prediction can stop at one's own action and become a movement directly through a motor pathway. On another occasion there may be a vague sense of its fall, a precise landing point, or even a sequence of images. There is no compulsory order and no need for an inner “I” to inspect the alternatives. Whether an action relied on an expectation about the outside world or only on the history already available is a question for experiments that alter the expectation. It cannot be decided from the appearance of the action.

Language too begins within experience. Someone saying “watch out” is another event actually taking place beside the cup, not an abbreviation of the cup's fall. Later, one word can prompt another and a sentence can invite an objection; the first half of one's own sentence can supply the conditions for the second. The author proposes that causal explanations, right and wrong, and logical judgments form in activity of this kind. The basic function has not changed from prediction into another intelligence. He further proposes that feeding sounds, images, and sensations back internally can sustain thought and give the impression of a self inside that listens and speaks. This explanation of human experience goes beyond showing that an artificial network can use the preceding step's signal for its next step. It requires separate evidence.

### 18.3 An outline of the whole process

```text
Agent and world change together. Current input and past experience enter one network.
                                      ↓
                         Expectation of some change
                  ┌───────────────────┼───────────────────┐
          How will I move?     What will move with it?    What will be said next?
                  ↓                   ↓                   ↓
           Action executed      Later observation       Condition for more words
                  ↓                   ↓                   ↓
           Actual feedback      Support or revision     Explanation, objection, judgment
                  └───────────────────┴───────────────────┘
                                      ↓
                           Re-entry into the joint state
```

These columns are possible contents of one process, not three machines. One alone may occur, or they may influence each other. Language need not interpose itself between perception and every action.

### 18.4 Four distinctions to retain

**Joint does not mean exhaustive.** One's own action may be the only predicted content, not a clip that has to be cut from an invisible film of the future.

**Language arises from shared activity.** There is no designer who must first encode the world in symbols. In practice, sounds and symbols gradually acquire the capacity to lead to other states.

**Implicit mechanism and explicit description are different.** Saying “because” or completing an inference does not establish a separate internal mechanism with the same name. Conversely, a predictive account must itself face evidence about mechanisms.

**Unity does not erase the requirements of implementation.** Predicting the world, language, and one's own actions can belong to one theory. A working model still needs appropriate data, training, memory, feedback, and routes to execution.

---

## Appendix A: Glossary

| Term | Use in this manuscript |
| --- | --- |
| State | A description of what is the case at a particular stage or within a given scope. “Transition” in the worldview should not be read as the claim that everything jumps frame by frame. |
| Joint state | The shared situation constituted by the agent and its surroundings, or the state constituted by multiple activities in a network. It need not exhaust all available information. |
| Segmentation / grouping | Distinguishing parts of a joint state that can vary separately and bringing together those that should be tracked together. More than detecting edges or outputting masks. |
| State transition | One state follows another. The arrow alone does not establish causation. |
| State transition prediction | Forming an expectation of some subsequent change from the present situation and history. It may be coarse, implicit, and restricted to one's own action, or detailed and span several kinds of content. |
| Implicit structure | Internal organization supporting reusable predictions, without prespecifying names drawn from human concepts. |
| Generalization | Appropriate prediction and behavior in situations or combinations not directly memorized. |
| Language formation | The process by which signals acquire stable continuations in shared practice. |
| Language running on its own | Linguistic products continue to condition prediction, so activity can unfold without the corresponding scene being present. |
| Causation and logic | Higher-level relations and constraints that this theory seeks to explain as arising in linguistic activity. |
| Action | A prediction of the agent's subsequent behavior realized through a motor or control pathway. Not every imagined action is executed. |
| Local world | A system or region with local state contents and transition descriptions that can be discussed within a certain scope. |
| Unified latent space | A system of states in which different kinds of content can jointly influence prediction. The modalities need not all have the same form. |

## Appendix B: Relation to the early English draft

The Chinese manuscript from which this edition is translated drew on an earlier file, `joint_state_transition_prediction_hypothesis.md`. It was not a paragraph-by-paragraph translation of that early file. After the early draft, the author revised the theory, and the Chinese manuscript was reorganized accordingly. This English edition follows the revised Chinese text.

| Possible reading of the earlier account | Formulation adopted here |
| --- | --- |
| First obtain a complete state, then predict the next one. | A little information is enough to begin. Observation and prediction continue to overlap. |
| “Joint” requires a background prediction of the entire future before deciding what to output. | Prediction itself may be incomplete. An image may prompt only the next action, a local external change, or a much more detailed prediction. |
| Many neurons mean that objects have already been segmented. | Structure makes local differences possible. Which parts form an individual can be learned in prediction and interaction. |
| First compress reality, then form concepts and language. | No independent compression mechanism is assumed; the relevant organization forms in prediction. |
| A word replaces a whole event. | The word was already in the shared experience. It is an element that can give rise to further connections. |
| First understand the structure of reality, then symbolize it. | Signals acquire symbolic functions in use, without requiring a design laid down in advance. |
| A world model supplies implicit causation; a language model supplies logic. | Both are discussed in terms of implicit transitions at the underlying level. Causation and logic belong to the level of linguistic explanation. |
| Direct action is the opposite of JSTP; only evaluating a consequence before acting counts as prediction. | Predicting one's own action already fits the theory. The use of an external consequence is a separate, testable mechanistic question. |
| A unified model cannot have different encoders or output heads. | It is the predictive system that is unified. Heterogeneous interfaces and local implementations are compatible with it. |

**Basis of the manuscript.** The author's successive discussions of joint state transitions, the formation of language, and intelligence based on prediction, together with later clarifications about segmentation, action prediction, predictive granularity, judgment at the level of language, and self-awareness. The cosmological and neuroscientific conjectures in the author's earlier *Law of Causation* have not been carried over as factual premises of this theory.

**Scope of the evidence.** This is a theoretical manuscript, not a survey of the external literature. The numbers in Section 15.8 come from the project records linked there. The hypothetical scenarios, draft formalization, and proposed studies are not experimental results. The references in the early English draft are not proof of the theory as a whole.

**English edition.** Translated with AI assistance from Ke Yue's Chinese revision of 19 November 2026. Section numbers, the register H1–H13, and the original distinctions between claims, examples, completed experiments, and proposed tests have been retained. Copyright © 2026 Ke Yue.
