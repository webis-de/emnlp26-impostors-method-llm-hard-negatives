# ✅ TODOs

## 📅 06.06.2025

---

### 📚 Dataset

- Control:
  - topic, genre, register  
  - $\neq$ PAN23: Cross-Discourse Type AV (e.g. essay vs. email)

- Requirements:
  - [ ] No format (paragraphs, layout, title) — plain text only  
  - [ ] No confounders (topic, genre, register, text length influence style)  
  - Constructed situations with manually rephrased texts are **okay**  
  - Small dataset is **okay**

- Sources:
  - [x] **Gutenberg books** (same vs. different author) — long text quality example  
    - Koppel Webis dataset contains texts also present in Gutenberg
    - [ ] Imposter  
    - [x] Unmasking  
  - [ ] **Fanfiction dataset (PAN)**  
    - [ ] Imposter  
    - [ ] Unmasking  
  - [ ] **Koppel et al. (2014) dataset**  
    - [x] (Blog posts)[https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download]
      - [ ] Imposter  
      - [x] Unmasking  
    - [ ] Student essays 
      - [ ] Imposter  
      - [ ] Unmasking  
  - [ ] **Paraphrasing approaches** at PAN and ELOQUENT  

---

### ✍️ Written Work

- [ ] Add **dataset discussion** to thesis
  - [ ] [PAN23](https://pan.webis.de/clef23/pan23-web/)
    - Too difficult  
    - What does PAN actually test/research?
  - [ ] Describe **optimal dataset(s)**:
    - Requirements & quality
    - Similarity to Koppel et al. (2014)
    - Similarity to PAN Fanfiction

- [ ] Add **task description** to thesis

- [ ] Add comparison:  
  **"Klassische Situation (Koppel et al. 2014) vs. Our Task"** 
  - [ ] Compare both situations  
    - Imposter method: Koppel had issues reducing confounders, easier with LLMs  
    - What is given, what is searched for  
    - [ ] Same premise:  
        Imposter approach for lineup of difficult opponents.  
        If candidate picked → strong evidence of correct authorship  

---

### 🛠️ Implementation

- [ ] Work is **precomputed** — no demo, no application  
- [ ] Imposter is an **"Einzelfalllösung"**
  - High inference/experiment cost is okay  
  - Controlled scenario is the goal  
- [ ] Implement:
  - Imposter via **similar content**  
  - Koppel et al. (2014) imposter approaches (baseline for modern version)
- [ ] Implement **upsampling** for texts < 500 words (cf. [Bevendorff Paper](https://aclanthology.org/N19-1068/)) 
- [ ] Use **exactly 500 words** per text (text length is a confounder)  
- [ ] Use Koppel et al. (2014) datasets
  - Find & use (maybe on Ceph)
- If paper unclear → make informed decisions
- Extend the paper if needed  
