# TODOs

## 06.06. 2025
Dataset:
    - control topic, genre, register
    - $neq$ PAN23: Cross-Discourse Type AV, e.g. essay vs. email
    [ ] no format (paragraphs, layout, title), plain text
    [ ] no confounders (topic, genre, register, text length influence style)
    - constructed situation with manually rephrased texts is okay
    - small dataset is okay
    [ ] compare Gutenberg books (same and different author case) for long text quality example
        [ ] Imposter
        [ ] Unmasking
    [ ] Fanfiction dataset (PAN)
        [ ] Imposter
        [ ] Unmasking
    [ ] Koppel et. Al. (2014) dataset
        [ ] Imposter
        [ ] Unmasking
    [ ] Paraphrasing approaches at PAN and ELOQUENT


Written work
[ ] Add dataset discussion to written theses
    [ ] (PAN23)[https://pan.webis.de/clef23/pan23-web/]
        - too difficult
        - what does PAN test/ research?
    [ ] Describe optimal dataset(s)
        - requirements, quality
        - similarity to Koppel et. Al. 2014
        - similarity to PAN fanfiction
[ ] Add task to written theses
[ ] Add 
[ ] "Klassische Situation (Koppel et. Al. 2014) vs. Our Task"
    - what is given, what is searched for (i.e. imposter methods: Koppel had issues reducing confounders, but easy with LLMs)
    - compare both situations
    [ ] but, same premise: Imposter approach for lineup of difficult opponents using good imposters, if candidate picked it is good evidence that correct

Implementation
[ ] Work is precomputed, no demo, no application!
[ ] Imposter is an "Einzelfalllösung"
    - costly/ effort for inference and experiments is okay
    - create controlled scenario
[ ] Implement Imposter via similar content
[ ] Implement Imposter from Koppel et Al. (2014)
    - baseline(s) for our modern interpretation of Koppel's imposter approach
[ ] Implement Upsampling for shorter than 500 word texts
[ ] Only use exactly 500 words per texts since text length heavily influences text quality (confounder)
[ ] Koppel et. Al. (2014) datasets
    - find and use, maybe on ceph
- if paper unclear, decide on own 
- extend paper
