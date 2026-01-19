# Research-Thinking-Agent

[![License](https://img.shields.io/github/license/Sakuya4/Research-Thinking-Agent)](https://github.com/Sakuya4/Research-Thinking-Agent/blob/6512cd55afdce891a4e553e77f8fe6d2684da0bc/LICENSE)

<img width="691" height="230" alt="image" src="https://github.com/user-attachments/assets/fd8b2be3-acba-453b-b8a2-49b4379f1557" />

## Inspiration

Academic research rarely starts with a clear problem statement.  
More often, it begins with a vague idea, a few keywords, or an intuition that something is worth exploring.

However, most existing AI tools assume that users already know *what* to search for. In reality, researchers often struggle with an earlier and more fundamental challenge: **figuring out the right research direction and framing the problem itself**.

This project was inspired by my own experience of getting stuck at the beginning of research—knowing the topic, but not knowing how to systematically think through the space or identify meaningful literature to read.

---

## What It Does

This project is an **AI research thinking agent** designed to support researchers *before* they start reading papers.

Given a small set of keywords or a high-level research idea, the agent:
- Decomposes the idea into structured sub-problems
- Identifies common methodologies and learning paradigms for each sub-problem
- Generates well-scoped academic search queries and research directions
- Mimics the reasoning process of an experienced research advisor

Instead of returning a list of papers, the agent focuses on **helping users think**, explore unfamiliar research spaces, and build a clearer mental model of the field.

---

## How We Built It

The system is implemented as a **command-line AI agent** using the Google Gemini API.

The agent operates in multiple reasoning stages:
1. **Concept Expansion** – Breaking down vague research ideas into concrete sub-problems  
2. **Method Mapping** – Connecting each sub-problem to commonly used approaches in academia  
3. **Search Strategy Generation** – Producing structured, high-quality search keywords and venues  

Gemini models are prompted to act as a *research advisor*, producing structured JSON outputs that represent intermediate reasoning steps. These outputs are then transformed into human-readable Markdown summaries for research planning.

This agent-based design keeps the reasoning process transparent, inspectable, and reusable.

---

## Challenges We Ran Into

One major challenge was preventing the model from skipping reasoning and immediately listing papers or buzzwords.

To address this, I had to carefully design prompts that enforce step-by-step decomposition, role constraints, and structured outputs.

Another challenge was finding the right balance between flexibility and structure—ensuring the agent remains useful across different research domains while still providing consistent, high-quality guidance.

---

## Accomplishments That We Proud Of

- Designing an AI agent that focuses on **research thinking**, not just information retrieval  
- Creating a reusable, structured reasoning pipeline instead of a single-pass response  
- Successfully using Gemini models to perform multi-stage, research-oriented reasoning  
- Building a tool that reflects how real researchers actually think and explore ideas  

---

## What's Next for the Agent

Future directions for this agent include:
- Integrating direct connections to academic databases (e.g., arXiv or Semantic Scholar)
- Adding iterative refinement, where users can guide or constrain the agent’s reasoning
- Supporting visual research maps and citation graphs
- Expanding the agent to assist with experiment design and hypothesis formulation

The long-term goal is to evolve this agent into a true **AI research collaborator**.

---

## Requirements / Prerequisites

### 1) Python
- Python **3.10+** recommended

### 2) Install Dependencies

From project root:

```
pip install -e .
```

### 3) Google AI Studio API Key
You need a Gemini API key from Google AI Studio.

Create a .env file in the project root:

```
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-flash-latest
```

---

## How To Run

Interactive CLI Mode
```
rta 
```

Then type a topic, for example:
  - ultrasound heart failure
  - LLM agent benchmarks
  - bio-impedance pulse wave modeling

The system will generate outputs and save them under runs/<run_id>/.

--- 
## References & Inspirations

This project is inspired by recent research on large language model (LLM) agents, 
automated research workflows, and agentic systems for scientific reasoning and synthesis.

In particular, the overall system design and long-term vision are influenced by the following works:

- **Liu et al., 2025**  
  *A Vision for Auto Research with LLM Agents*  
  arXiv preprint arXiv:2504.18765  

- **Cao et al., 2025**  
  *Large Language Models for Planning: A Comprehensive and Systematic Survey*  
  arXiv preprint arXiv:2505.19683  

- **Shittu & Quaye, 2025**  
  *AI-Assisted Handheld Echocardiography by Nonexpert Operators: A Narrative Review of Prospective Studies*  
  Cureus, 17(11): e97050. doi:10.7759/cureus.97050  


- **Liu et al., 2025**  
  *Agentic AutoSurvey: Let LLMs Survey LLMs*  
  arXiv preprint arXiv:2509.18661  
